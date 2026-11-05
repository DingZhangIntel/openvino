// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npuw_transformations/left_aligned_attention.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "npuw_transformations/detect_causal_mask.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/openvino.hpp"

namespace {

constexpr size_t kContextCapacity = 8u;

std::shared_ptr<ov::op::v0::Parameter> make_parameter(const std::string& name,
                                                      const ov::element::Type& type,
                                                      const ov::Shape& shape) {
    auto parameter = std::make_shared<ov::op::v0::Parameter>(type, shape);
    parameter->set_friendly_name(name);
    parameter->output(0).set_names({name});
    return parameter;
}

std::shared_ptr<ov::Model> make_static_attention_model(size_t query_length,
                                                       int64_t mask_kind,
                                                       ov::element::Type mask_type = ov::element::f32) {
    const auto past_capacity = kContextCapacity - query_length;
    auto query = make_parameter("query", mask_type, {1, 1, query_length, 1});
    auto past_key = make_parameter("past_key_values.0.key", mask_type, {1, 1, past_capacity, 1});
    auto present_key = make_parameter("present_key", mask_type, {1, 1, query_length, 1});
    auto past_value = make_parameter("past_key_values.0.value", mask_type, {1, 1, past_capacity, 1});
    auto present_value = make_parameter("present_value", mask_type, {1, 1, query_length, 1});
    auto attention_mask = make_parameter("attention_mask", ov::element::i64, {1, kContextCapacity});

    auto key = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{past_key, present_key}, 2);
    auto value = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{past_value, present_value}, 2);
    auto mask_shape = ov::op::v0::Constant::create(
        ov::element::i64,
        ov::Shape{4},
        std::vector<int64_t>{1, 1, static_cast<int64_t>(query_length), static_cast<int64_t>(kContextCapacity)});
    auto original_mask =
        std::make_shared<ov::op::v3::Broadcast>(std::make_shared<ov::op::v0::Convert>(attention_mask, mask_type),
                                                mask_shape);
    auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(query, key, value, original_mask, false);
    sdpa->get_rt_info()[ov::npuw::NPUW_SDPA_MASK_RT_KEY] = mask_kind;

    return std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(sdpa),
                         std::make_shared<ov::op::v0::Result>(key),
                         std::make_shared<ov::op::v0::Result>(value)},
        ov::ParameterVector{query, past_key, present_key, past_value, present_value, attention_mask});
}

ov::Tensor evaluate_mask(const std::shared_ptr<ov::Model>& model, size_t valid_present_rows) {
    ov::TensorVector inputs;
    inputs.reserve(model->get_parameters().size());
    for (const auto& parameter : model->get_parameters()) {
        ov::Tensor input(parameter->get_element_type(), parameter->get_shape());
        std::memset(input.data(), 0, input.get_byte_size());
        const auto& name = parameter->get_friendly_name();
        if (name == "attention_mask") {
            std::fill_n(input.data<int64_t>(), 2u + valid_present_rows, int64_t{1});
        } else if (name == "past_len") {
            input.data<int64_t>()[0] = 2;
        }
        inputs.push_back(input);
    }

    const auto ops = model->get_ops();
    const auto sdpa = std::find_if(ops.begin(), ops.end(), [](const auto& node) {
        return ov::is_type<ov::op::v13::ScaledDotProductAttention>(node);
    });
    OPENVINO_ASSERT(sdpa != ops.end());
    const auto mask = (*sdpa)->input_value(3);
    auto mask_model = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(mask)},
                                                  model->get_parameters());
    ov::TensorVector outputs = {ov::Tensor(mask.get_element_type(), mask.get_shape())};
    EXPECT_TRUE(mask_model->evaluate(outputs, inputs));
    return outputs.front();
}

void expect_contiguous_mask(const ov::Tensor& mask, size_t query_length, size_t valid_present_rows, int64_t mask_kind) {
    ASSERT_EQ(mask.get_shape(), (ov::Shape{1u, 1u, query_length, kContextCapacity}));
    constexpr size_t past_len = 2u;
    const auto* values = mask.data<const float>();
    for (size_t query_index = 0; query_index < query_length; ++query_index) {
        const auto query_position = past_len + query_index;
        for (size_t key_index = 0; key_index < kContextCapacity; ++key_index) {
            bool expected_valid = key_index <= query_position &&
                                  (key_index < past_len + valid_present_rows || key_index == query_position);
            if (mask_kind >= 0) {
                expected_valid = expected_valid && key_index > query_position - static_cast<size_t>(mask_kind);
            }
            EXPECT_EQ(values[query_index * kContextCapacity + key_index],
                      expected_valid ? 0.f : std::numeric_limits<float>::lowest())
                << "query=" << query_index << ", key=" << key_index;
        }
    }
}

struct LeftAlignedAttentionCase {
    size_t query_length;
    size_t valid_present_rows;
};

class LeftAlignedAttentionTest : public testing::TestWithParam<LeftAlignedAttentionCase> {};

TEST_P(LeftAlignedAttentionTest, BuildsContiguousFullAndSlidingMasksWithPastHistory) {
    const auto& test_case = GetParam();
    for (const auto mask_kind : {ov::npuw::NPUW_SDPA_MASK_CAUSAL, int64_t{2}}) {
        auto model = make_static_attention_model(test_case.query_length, mask_kind);
        ASSERT_TRUE(ov::npuw::LeftAlignedAttention().run_on_model(model));

        const auto& parameters = model->get_parameters();
        const auto past_len = std::find_if(parameters.begin(), parameters.end(), [](const auto& parameter) {
            return parameter->get_friendly_name() == "past_len";
        });
        ASSERT_NE(past_len, model->get_parameters().end());
        EXPECT_EQ((*past_len)->get_element_type(), ov::element::i64);
        EXPECT_TRUE((*past_len)->get_shape().empty());

        expect_contiguous_mask(evaluate_mask(model, test_case.valid_present_rows),
                               test_case.query_length,
                               test_case.valid_present_rows,
                               mask_kind);
    }
}

INSTANTIATE_TEST_SUITE_P(StaticChunkShape,
                         LeftAlignedAttentionTest,
                         testing::Values(LeftAlignedAttentionCase{3u, 2u}, LeftAlignedAttentionCase{1u, 1u}));

TEST(LeftAlignedAttentionTest, KeepsTheDiagonalValidForPaddedRows) {
    auto model = make_static_attention_model(3u, ov::npuw::NPUW_SDPA_MASK_CAUSAL);
    ASSERT_TRUE(ov::npuw::LeftAlignedAttention().run_on_model(model));

    const auto mask = evaluate_mask(model, 2u);
    EXPECT_EQ(mask.data<const float>()[2u * kContextCapacity + 4u], 0.f);
}

TEST(LeftAlignedAttentionTest, UsesFiniteLowestAdditiveMaskValue) {
    auto model = make_static_attention_model(1u, ov::npuw::NPUW_SDPA_MASK_CAUSAL, ov::element::f16);
    ASSERT_TRUE(ov::npuw::LeftAlignedAttention().run_on_model(model));

    const auto ops = model->get_ops();
    const auto masked_value = std::find_if(ops.begin(), ops.end(), [](const auto& node) {
        const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
        return constant && constant->get_element_type() == ov::element::f16 && constant->get_shape().empty() &&
               constant->cast_vector<float>().front() < -60000.f;
    });
    ASSERT_NE(masked_value, ops.end());
    const auto masked_constant = ov::as_type_ptr<ov::op::v0::Constant>(*masked_value);
    ASSERT_NE(masked_constant, nullptr);
    EXPECT_TRUE(std::isfinite(masked_constant->cast_vector<float>().front()));
}

TEST(LeftAlignedAttentionTest, RewritesMasksWithoutScatteringKvForIsolation) {
    auto model = make_static_attention_model(1u, ov::npuw::NPUW_SDPA_MASK_CAUSAL);
    ASSERT_TRUE(ov::npuw::LeftAlignedAttention(false).run_on_model(model));

    const auto ops = model->get_ops();
    EXPECT_FALSE(std::any_of(ops.begin(), ops.end(), [](const auto& node) {
        return ov::is_type<ov::op::v3::ScatterUpdate>(node);
    }));
}

TEST(LeftAlignedAttentionTest, RejectsPastKvAttentionWithUnknownMask) {
    auto model = make_static_attention_model(1u, ov::npuw::NPUW_SDPA_MASK_CAUSAL);
    for (const auto& node : model->get_ops()) {
        if (const auto sdpa = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(node)) {
            sdpa->get_rt_info().erase(ov::npuw::NPUW_SDPA_MASK_RT_KEY);
        }
    }
    EXPECT_THROW(ov::npuw::LeftAlignedAttention().run_on_model(model), ov::Exception);
}

TEST(LeftAlignedAttentionTest, RejectsPastKvAttentionWithUnsupportedLayoutPath) {
    auto model = make_static_attention_model(1u, ov::npuw::NPUW_SDPA_MASK_CAUSAL);
    for (const auto& node : model->get_ops()) {
        if (const auto sdpa = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(node)) {
            sdpa->input(1).replace_source_output(std::make_shared<ov::op::v0::Abs>(sdpa->input_value(1)));
        }
    }
    model->validate_nodes_and_infer_types();

    EXPECT_THROW(ov::npuw::LeftAlignedAttention().run_on_model(model), ov::Exception);
}

TEST(LeftAlignedAttentionTest, RejectsMasksThatDependOnAnotherModelInput) {
    auto model = make_static_attention_model(1u, ov::npuw::NPUW_SDPA_MASK_CAUSAL);
    auto token_type_ids = make_parameter("token_type_ids", ov::element::f32, {1, kContextCapacity});
    auto token_type_mask = std::make_shared<ov::op::v3::Broadcast>(
        token_type_ids,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, 1, 1, 8}));
    for (const auto& node : model->get_ops()) {
        if (const auto sdpa = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(node)) {
            auto mask = std::make_shared<ov::op::v1::Add>(sdpa->input_value(3), token_type_mask);
            sdpa->input(3).replace_source_output(mask);
        }
    }
    model->add_parameters({token_type_ids});
    model->validate_nodes_and_infer_types();

    EXPECT_THROW(ov::npuw::LeftAlignedAttention().run_on_model(model), ov::Exception);
}

}  // namespace