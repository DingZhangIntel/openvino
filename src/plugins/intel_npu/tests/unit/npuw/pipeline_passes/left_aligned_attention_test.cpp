// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npuw_transformations/left_aligned_attention.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "npuw_transformations/detect_causal_mask.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/openvino.hpp"

namespace {

constexpr size_t kPastCapacity = 5u;
constexpr size_t kContextCapacity = 8u;

std::shared_ptr<ov::op::v0::Parameter> make_parameter(const std::string& name,
                                                      const ov::element::Type& type,
                                                      const ov::Shape& shape) {
    auto parameter = std::make_shared<ov::op::v0::Parameter>(type, shape);
    parameter->set_friendly_name(name);
    parameter->output(0).set_names({name});
    return parameter;
}

std::shared_ptr<ov::Model> make_static_attention_model(size_t query_length, int64_t mask_kind) {
    auto query = make_parameter("query", ov::element::f32, {1, 1, query_length, 1});
    auto past_key = make_parameter("past_key_values.0.key", ov::element::f32, {1, 1, kPastCapacity, 1});
    auto present_key = make_parameter("present_key", ov::element::f32, {1, 1, query_length, 1});
    auto past_value = make_parameter("past_key_values.0.value", ov::element::f32, {1, 1, kPastCapacity, 1});
    auto present_value = make_parameter("present_value", ov::element::f32, {1, 1, query_length, 1});
    auto attention_mask = make_parameter("attention_mask", ov::element::i64, {1, kContextCapacity});

    auto key = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{past_key, present_key}, 2);
    auto value = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{past_value, present_value}, 2);
    auto mask_shape = ov::op::v0::Constant::create(
        ov::element::i64,
        ov::Shape{4},
        std::vector<int64_t>{1, 1, static_cast<int64_t>(query_length), static_cast<int64_t>(kContextCapacity)});
    auto original_mask =
        std::make_shared<ov::op::v3::Broadcast>(std::make_shared<ov::op::v0::Convert>(attention_mask, ov::element::f32),
                                                mask_shape);
    auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(query, key, value, original_mask, false);
    sdpa->get_rt_info()[ov::npuw::NPUW_SDPA_MASK_RT_KEY] = mask_kind;

    return std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(sdpa),
                         std::make_shared<ov::op::v0::Result>(key),
                         std::make_shared<ov::op::v0::Result>(value)},
        ov::ParameterVector{query, past_key, present_key, past_value, present_value, attention_mask});
}

ov::TensorVector evaluate(const std::shared_ptr<ov::Model>& model, size_t valid_present_rows) {
    ov::TensorVector inputs;
    inputs.reserve(model->get_parameters().size());
    for (const auto& parameter : model->get_parameters()) {
        ov::Tensor input(parameter->get_element_type(), parameter->get_shape());
        std::memset(input.data(), 0, input.get_byte_size());
        const auto& name = parameter->get_friendly_name();
        if (name == "past_key_values.0.key") {
            const float values[] = {1.f, 2.f, 3.f, 4.f, 5.f};
            std::copy(std::begin(values), std::end(values), input.data<float>());
        } else if (name == "present_key") {
            const float values[] = {10.f, 11.f, 12.f};
            std::copy_n(values, input.get_size(), input.data<float>());
        } else if (name == "past_key_values.0.value") {
            const float values[] = {10.f, 20.f, 30.f, 40.f, 50.f};
            std::copy(std::begin(values), std::end(values), input.data<float>());
        } else if (name == "present_value") {
            const float values[] = {100.f, 200.f, 300.f};
            std::copy_n(values, input.get_size(), input.data<float>());
        } else if (name == "attention_mask") {
            std::fill_n(input.data<int64_t>(), 2u + valid_present_rows, int64_t{1});
        } else if (name == "past_len") {
            input.data<int64_t>()[0] = 2;
        }
        inputs.push_back(input);
    }

    ov::TensorVector outputs;
    outputs.reserve(model->get_results().size());
    for (const auto& result : model->get_results()) {
        outputs.emplace_back(result->input_value(0).get_element_type(), result->input_value(0).get_shape());
    }
    EXPECT_TRUE(model->evaluate(outputs, inputs));
    return outputs;
}

void expect_values(const ov::Tensor& tensor, const std::vector<float>& expected) {
    ASSERT_EQ(tensor.get_size(), expected.size());
    for (size_t index = 0; index < expected.size(); ++index) {
        EXPECT_NEAR(tensor.data<const float>()[index], expected[index], 1e-5f) << "index=" << index;
    }
}

struct LeftAlignedAttentionCase {
    size_t query_length;
    size_t valid_present_rows;
    std::vector<float> full_expected;
    std::vector<float> sliding_expected;
};

class LeftAlignedAttentionTest : public testing::TestWithParam<LeftAlignedAttentionCase> {};

TEST_P(LeftAlignedAttentionTest, BuildsContiguousFullAndSlidingMasksWithPastHistory) {
    const auto& test_case = GetParam();
    for (const auto& [mask_kind, expected] :
         std::vector<std::pair<int64_t, std::vector<float>>>{{ov::npuw::NPUW_SDPA_MASK_CAUSAL, test_case.full_expected},
                                                             {2, test_case.sliding_expected}}) {
        auto model = make_static_attention_model(test_case.query_length, mask_kind);
        ASSERT_TRUE(ov::npuw::LeftAlignedAttention().run_on_model(model));

        const auto& parameters = model->get_parameters();
        const auto past_len = std::find_if(parameters.begin(), parameters.end(), [](const auto& parameter) {
            return parameter->get_friendly_name() == "past_len";
        });
        ASSERT_NE(past_len, model->get_parameters().end());
        EXPECT_EQ((*past_len)->get_element_type(), ov::element::i64);
        EXPECT_TRUE((*past_len)->get_shape().empty());

        const auto outputs = evaluate(model, test_case.valid_present_rows);
        expect_values(outputs.at(0), expected);

        // The SDPA branch consumes ScatterUpdate rows [past_len, past_len + L),
        // while the cache result keeps its original fixed-shape Concat route.
        expect_values(outputs.at(1), {1.f, 2.f, 3.f, 4.f, 5.f, 10.f, 11.f, 12.f});
        expect_values(outputs.at(2), {10.f, 20.f, 30.f, 40.f, 50.f, 100.f, 200.f, 300.f});
    }
}

INSTANTIATE_TEST_SUITE_P(
    StaticChunkShape,
    LeftAlignedAttentionTest,
    testing::Values(LeftAlignedAttentionCase{3u, 2u, {130.f / 3.f, 82.5f, 82.5f}, {60.f, 150.f, 200.f}},
                    LeftAlignedAttentionCase{1u, 1u, {130.f / 3.f}, {60.f}}));

}  // namespace