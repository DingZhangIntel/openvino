// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>

#include "embedding/embedding_infer_request.hpp"
#include "executor.hpp"
#include "infer_request_utils.hpp"
#include "llm_test_helpers.hpp"
#include "openvino/openvino.hpp"

namespace ov::test::npuw {

struct EmbeddingInferRequestTestAccess {
    static ov::SoPtr<ov::ITensor> prefill_input(ov::npuw::EmbeddingInferRequest& request,
                                                const std::string& name) {
        return request.m_prefill_request->get_tensor(request.m_prefill_in_ports.at(name));
    }
};

}  // namespace ov::test::npuw

namespace {

using ov::test::npuw::EmbeddingInferRequestTestAccess;
using ov::test::npuw::NullPlugin;

class FakeEmbeddingCompiledModel;

class FakeEmbeddingInferRequest final : public ov::ISyncInferRequest {
public:
    explicit FakeEmbeddingInferRequest(std::shared_ptr<const FakeEmbeddingCompiledModel> compiled_model);

    void infer() override;
    void check_tensors() const override {}
    std::vector<ov::ProfilingInfo> get_profiling_info() const override {
        return {};
    }
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override {
        return {};
    }
};

class FakeEmbeddingCompiledModel final : public ov::npuw::ICompiledModel_v0 {
public:
    FakeEmbeddingCompiledModel(const std::shared_ptr<ov::Model>& model,
                               const std::shared_ptr<const ov::IPlugin>& plugin,
                               const ov::AnyMap&)
        : ov::npuw::ICompiledModel_v0(model, plugin), m_model(model) {}

    void export_model(std::ostream&) const override {}
    std::shared_ptr<const ov::Model> get_runtime_model() const override {
        return m_model;
    }
    void set_property(const ov::AnyMap&) override {}
    ov::Any get_property(const std::string&) const override {
        return {};
    }
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override {
        auto self = std::static_pointer_cast<const FakeEmbeddingCompiledModel>(shared_from_this());
        return std::make_shared<FakeEmbeddingInferRequest>(std::move(self));
    }
    std::shared_ptr<ov::npuw::IBaseInferRequest> create_base_infer_request() const override {
        return {};
    }
    std::shared_ptr<ov::IAsyncInferRequest> wrap_async_infer_request(
        std::shared_ptr<ov::npuw::IBaseInferRequest>) const override {
        return std::make_shared<ov::IAsyncInferRequest>(create_sync_infer_request(),
                                                        intel_npu::make_executor("embedding_test_task", 1),
                                                        intel_npu::make_executor("embedding_test_callback", 1));
    }
    std::string submodel_device(std::size_t) const override {
        return "CPU";
    }
    std::size_t num_submodels() const override {
        return 1;
    }
    std::shared_ptr<ov::npuw::weights::Bank> get_weights_bank() const override {
        return {};
    }
    void set_weights_bank(std::shared_ptr<ov::npuw::weights::Bank>) override {}
    void finalize_weights_bank() override {}
    void reconstruct_closure() override {}
    void serialize(std::ostream&, const ov::npuw::s11n::CompiledContext&) const override {}

private:
    std::shared_ptr<ov::Model> m_model;
};

FakeEmbeddingInferRequest::FakeEmbeddingInferRequest(
    std::shared_ptr<const FakeEmbeddingCompiledModel> compiled_model)
    : ov::ISyncInferRequest(std::move(compiled_model)) {
    for (const auto& input : get_compiled_model()->inputs()) {
        set_tensor(input, ov::get_tensor_impl(ov::Tensor(input.get_element_type(), input.get_shape())));
    }
    for (const auto& output : get_compiled_model()->outputs()) {
        set_tensor(output, ov::get_tensor_impl(ov::Tensor(output.get_element_type(), output.get_shape())));
    }
}

void FakeEmbeddingInferRequest::infer() {
    const auto input_ids_port = ov::npuw::util::find_port_by_name(get_compiled_model()->inputs(), "input_ids");
    ASSERT_TRUE(input_ids_port.has_value());
    const auto input_ids = get_tensor(input_ids_port.value());
    const auto output = get_tensor(get_compiled_model()->outputs().front());
    ASSERT_EQ(output->get_element_type(), ov::element::f32);
    ASSERT_GE(output->get_shape().size(), 2u);
    std::memset(output->data(), 0, output->get_byte_size());

    const auto seq_len = input_ids->get_shape()[1];
    const auto row_size = output->get_size() / output->get_shape()[1];
    for (size_t token = 0; token < seq_len; ++token) {
        std::fill_n(output->data<float>() + token * row_size,
                    row_size,
                    static_cast<float>(input_ids->data<int64_t>()[token]));
    }

    for (size_t index = 1; index < get_compiled_model()->outputs().size(); ++index) {
        auto tensor = get_tensor(get_compiled_model()->outputs()[index]);
        std::memset(tensor->data(), 0, tensor->get_byte_size());
    }
}

TEST(EmbeddingInferRequestTest, PartialFinalChunkUsesLeftAlignedBuffersAndOutputRows) {
    auto plugin = std::make_shared<NullPlugin>();
    ov::npuw::LLMCompiledModel::CompiledModelFactory factory =
        [](const std::shared_ptr<ov::Model>& model,
           const std::shared_ptr<const ov::IPlugin>& subplugin,
           const ov::AnyMap& properties) -> std::shared_ptr<ov::npuw::ICompiledModel_v0> {
        return std::make_shared<FakeEmbeddingCompiledModel>(model, subplugin, properties);
    };
    const ov::AnyMap properties{{"NPUW_LLM", "YES"},
                                {"NPUW_DEVICES", "CPU"},
                                {"NPUW_TEXT_EMBED", "YES"},
                                {"NPUW_LLM_SHARED_HEAD", "NO"},
                                {"NPUW_LLM_MAX_PROMPT_LEN", "128"},
                                {"NPUW_LLM_MIN_RESPONSE_LEN", "64"},
                                {"NPUW_LLM_PREFILL_HINT", "DYNAMIC"},
                                {"NPUW_LLM_PREFILL_CHUNK_SIZE", "32"}};
    auto compiled = std::make_shared<ov::npuw::LLMCompiledModel>(
        ov::test::npuw::build_embedding_decoder_test_model(), plugin, properties, factory);
    ov::npuw::EmbeddingInferRequest request(compiled);

    constexpr size_t prompt_len = 65u;
    ov::Tensor input_ids(ov::element::i64, ov::Shape{1, prompt_len});
    std::iota(input_ids.data<int64_t>(), input_ids.data<int64_t>() + prompt_len, int64_t{1});
    ov::Tensor attention_mask(ov::element::i64, ov::Shape{1, prompt_len});
    std::fill_n(attention_mask.data<int64_t>(), prompt_len, int64_t{1});

    const auto input_ids_port = ov::npuw::util::find_port_by_name(compiled->inputs(), "input_ids");
    const auto attention_mask_port = ov::npuw::util::find_port_by_name(compiled->inputs(), "attention_mask");
    ASSERT_TRUE(input_ids_port.has_value());
    ASSERT_TRUE(attention_mask_port.has_value());
    request.set_tensor(input_ids_port.value(), ov::get_tensor_impl(input_ids));
    request.set_tensor(attention_mask_port.value(), ov::get_tensor_impl(attention_mask));
    ASSERT_NO_THROW(request.infer());

    auto staged_ids = EmbeddingInferRequestTestAccess::prefill_input(request, "input_ids");
    ASSERT_EQ(staged_ids->get_size(), 32u);
    EXPECT_EQ(staged_ids->data<int64_t>()[0], 65);
    EXPECT_TRUE(std::all_of(staged_ids->data<int64_t>() + 1,
                            staged_ids->data<int64_t>() + staged_ids->get_size(),
                            [](int64_t value) {
                                return value == 0;
                            }));

    auto staged_positions = EmbeddingInferRequestTestAccess::prefill_input(request, "position_ids");
    EXPECT_EQ(staged_positions->data<int64_t>()[0], 64);
    EXPECT_TRUE(std::all_of(staged_positions->data<int64_t>() + 1,
                            staged_positions->data<int64_t>() + staged_positions->get_size(),
                            [](int64_t value) {
                                return value == 0;
                            }));

    auto staged_mask = EmbeddingInferRequestTestAccess::prefill_input(request, "attention_mask");
    EXPECT_TRUE(std::all_of(staged_mask->data<int64_t>(), staged_mask->data<int64_t>() + prompt_len, [](int64_t value) {
        return value == 1;
    }));
    EXPECT_TRUE(std::all_of(staged_mask->data<int64_t>() + prompt_len,
                            staged_mask->data<int64_t>() + staged_mask->get_size(),
                            [](int64_t value) {
                                return value == 0;
                            }));

    auto output = request.get_tensor(compiled->outputs().front());
    const auto output_row_size = output->get_size() / output->get_shape()[1];
    for (size_t token = 0; token < prompt_len; ++token) {
        EXPECT_FLOAT_EQ(output->data<float>()[token * output_row_size], static_cast<float>(token + 1));
    }
}

}  // namespace