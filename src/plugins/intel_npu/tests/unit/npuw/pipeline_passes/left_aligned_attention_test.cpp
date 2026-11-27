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
#include "openvino/op/gather.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/openvino.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/op_conversions/convert_convertlike.hpp"
#include "transformations/op_conversions/scaled_dot_product_attention_decomposition.hpp"

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
                                                       ov::element::Type mask_type = ov::element::f32,
                                                       bool grouped = false) {
    const auto past_capacity = kContextCapacity - query_length;
    auto query = make_parameter("query", mask_type, {1, grouped ? 2u : 1u, query_length, 1});
    auto past_key = make_parameter("past_key_values.0.key", mask_type, {1, 1, past_capacity, 1});
    auto present_key = make_parameter("present_key", mask_type, {1, 1, query_length, 1});
    auto past_value = make_parameter("past_key_values.0.value", mask_type, {1, 1, past_capacity, 1});
    auto present_value = make_parameter("present_value", mask_type, {1, 1, query_length, 1});
    auto attention_mask = make_parameter("attention_mask", ov::element::i64, {1, kContextCapacity});

    auto key = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{past_key, present_key}, grouped ? -2 : 2);
    auto value = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{past_value, present_value}, grouped ? -2 : 2);
    auto mask_shape = ov::op::v0::Constant::create(
        ov::element::i64,
        ov::Shape{4},
        std::vector<int64_t>{1, 1, static_cast<int64_t>(query_length), static_cast<int64_t>(kContextCapacity)});
    auto original_mask =
        std::make_shared<ov::op::v3::Broadcast>(std::make_shared<ov::op::v0::Convert>(attention_mask, mask_type),
                                                mask_shape);
    ov::Output<ov::Node> key_input = key;
    ov::Output<ov::Node> value_input = value;
    if (grouped) {
        const auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
        const auto broadcast_shape =
            ov::op::v0::Constant::create(ov::element::i64,
                                         ov::Shape{5},
                                         std::vector<int64_t>{1, 1, 2, static_cast<int64_t>(kContextCapacity), 1});
        const auto reshape_shape =
            ov::op::v0::Constant::create(ov::element::i64,
                                         ov::Shape{4},
                                         std::vector<int64_t>{1, 2, static_cast<int64_t>(kContextCapacity), 1});
        const auto repeat_heads = [&](const ov::Output<ov::Node>& input) {
            const auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(input, axis);
            const auto broadcast = std::make_shared<ov::op::v3::Broadcast>(unsqueeze,
                                                                           broadcast_shape,
                                                                           ov::op::BroadcastType::BIDIRECTIONAL);
            return std::make_shared<ov::op::v1::Reshape>(broadcast, reshape_shape, false);
        };
        key_input = repeat_heads(key);
        value_input = repeat_heads(value);
    }
    auto sdpa =
        std::make_shared<ov::op::v13::ScaledDotProductAttention>(query, key_input, value_input, original_mask, false);
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

TEST(LeftAlignedAttentionTest, SingleTokenAtEndMatchesScatteredAttention) {
    for (const bool grouped : {false, true}) {
        for (const auto mask_kind : {ov::npuw::NPUW_SDPA_MASK_CAUSAL, int64_t{2}}) {
            SCOPED_TRACE(testing::Message() << "grouped=" << grouped << ", mask_kind=" << mask_kind);
            auto scattered = make_static_attention_model(1u, mask_kind, ov::element::f32, grouped);
            auto at_end = scattered->clone();
            ASSERT_TRUE(ov::npuw::LeftAlignedAttention().run_on_model(scattered));
            ASSERT_TRUE(
                ov::npuw::LeftAlignedAttention(true, ov::npuw::LeftAlignedAttention::SingleTokenPlacement::AT_END)
                    .run_on_model(at_end));
            const auto at_end_ops = at_end->get_ops();
            EXPECT_FALSE(std::any_of(at_end_ops.begin(), at_end_ops.end(), [](const auto& node) {
                return ov::is_type<ov::op::v3::ScatterUpdate>(node);
            }));
            for (const auto& model : {scattered, at_end}) {
                ov::pass::Manager manager;
                manager.register_pass<ov::pass::ScaledDotProductAttentionDecomposition>();
                manager.register_pass<ov::pass::ConvertConvertLike>();
                manager.run_passes(model);
            }
            for (const int64_t past_length : {int64_t{0}, int64_t{1}, int64_t{2}, int64_t{6}, int64_t{7}}) {
                std::vector<ov::TensorVector> results;
                for (const auto& model : {scattered, at_end}) {
                    ov::TensorVector inputs;
                    for (const auto& parameter : model->get_parameters()) {
                        const auto& name = parameter->get_friendly_name();
                        ov::Tensor tensor(parameter->get_element_type(), parameter->get_shape());
                        if (name == "past_len") {
                            tensor.data<int64_t>()[0] = past_length;
                        } else if (name == "attention_mask") {
                            for (size_t index = 0; index < tensor.get_size(); ++index) {
                                tensor.data<int64_t>()[index] = index % 3u != 0u ? 1 : 0;
                            }
                        } else {
                            for (size_t index = 0; index < tensor.get_size(); ++index) {
                                tensor.data<float>()[index] = static_cast<float>(index + 1u) / 4.f;
                            }
                        }
                        inputs.push_back(tensor);
                    }
                    ov::TensorVector outputs;
                    for (const auto& output : model->outputs()) {
                        outputs.emplace_back(output.get_element_type(), output.get_shape());
                    }
                    ASSERT_TRUE(model->evaluate(outputs, inputs));
                    results.push_back(std::move(outputs));
                }
                const auto& expected = results[0][0];
                const auto& actual = results[1][0];
                ASSERT_EQ(actual.get_size(), expected.get_size());
                for (size_t index = 0; index < actual.get_size(); ++index) {
                    EXPECT_NEAR(actual.data<const float>()[index], expected.data<const float>()[index], 1e-5f)
                        << "past_length=" << past_length << ", index=" << index;
                }
            }
        }
    }
}

TEST(LeftAlignedAttentionTest, SingleTokenAtEndFallsBackForUnsupportedLayouts) {
    auto model = make_static_attention_model(1u, ov::npuw::NPUW_SDPA_MASK_CAUSAL);
    for (const auto& node : model->get_ops()) {
        if (const auto sdpa = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(node)) {
            const auto factor = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {2.f});
            sdpa->input(1).replace_source_output(std::make_shared<ov::op::v1::Multiply>(sdpa->input_value(1), factor));
        }
    }
    model->validate_nodes_and_infer_types();
    ASSERT_TRUE(ov::npuw::LeftAlignedAttention(true, ov::npuw::LeftAlignedAttention::SingleTokenPlacement::AT_END)
                    .run_on_model(model));
    const auto ops = model->get_ops();
    EXPECT_EQ(std::count_if(ops.begin(),
                            ops.end(),
                            [](const auto& node) {
                                return ov::is_type<ov::op::v3::ScatterUpdate>(node);
                            }),
              2);
}

TEST(LeftAlignedAttentionTest, SingleTokenAtEndFallsBackForPresentBeforePastAndZeroWindow) {
    for (const bool zero_window : {false, true}) {
        auto model = make_static_attention_model(1u, zero_window ? 0 : ov::npuw::NPUW_SDPA_MASK_CAUSAL);
        if (!zero_window) {
            for (const auto& node : model->get_ops()) {
                const auto concat = ov::as_type_ptr<ov::op::v0::Concat>(node);
                if (concat && concat->get_input_size() == 2u) {
                    const auto past = concat->input_value(0);
                    const auto present = concat->input_value(1);
                    concat->set_arguments({present, past});
                }
            }
            model->validate_nodes_and_infer_types();
        }
        ASSERT_TRUE(ov::npuw::LeftAlignedAttention(true, ov::npuw::LeftAlignedAttention::SingleTokenPlacement::AT_END)
                        .run_on_model(model));
        const auto ops = model->get_ops();
        EXPECT_EQ(std::count_if(ops.begin(),
                                ops.end(),
                                [](const auto& node) {
                                    return ov::is_type<ov::op::v3::ScatterUpdate>(node);
                                }),
                  2);
    }
}

TEST(LeftAlignedAttentionTest, SingleTokenAtEndRejectsMultiplePresentRows) {
    auto model = make_static_attention_model(1u, ov::npuw::NPUW_SDPA_MASK_CAUSAL);
    for (const auto& parameter : model->get_parameters()) {
        const auto& name = parameter->get_friendly_name();
        if (name.find("past_key_values") != std::string::npos) {
            parameter->set_partial_shape({1, 1, kContextCapacity - 2u, 1});
        } else if (name == "present_key" || name == "present_value") {
            parameter->set_partial_shape({1, 1, 2, 1});
        }
    }
    model->validate_nodes_and_infer_types();
    EXPECT_THROW(ov::npuw::LeftAlignedAttention(true, ov::npuw::LeftAlignedAttention::SingleTokenPlacement::AT_END)
                     .run_on_model(model),
                 ov::Exception);
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

TEST(LeftAlignedAttentionTest, RewritesLfm3MaskUsingStaticPastKvShape) {
    constexpr size_t query_length = 3u;
    auto model = make_static_attention_model(query_length, int64_t{2});
    const auto& parameters = model->get_parameters();
    const auto past_key = *std::find_if(parameters.begin(), parameters.end(), [](const auto& parameter) {
        return parameter->get_friendly_name() == "past_key_values.0.key";
    });
    const auto attention_mask = *std::find_if(parameters.begin(), parameters.end(), [](const auto& parameter) {
        return parameter->get_friendly_name() == "attention_mask";
    });
    const auto scalar = [](int64_t value) {
        return ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {value});
    };
    const auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {0, 1});
    const auto padding_axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 2});
    const auto past_shape = std::make_shared<ov::op::v3::ShapeOf>(past_key);
    const auto past_length = std::make_shared<ov::op::v8::Gather>(past_shape, scalar(2), scalar(0));
    const auto total_length = std::make_shared<ov::op::v1::Add>(past_length, scalar(query_length));
    const auto query_range =
        std::make_shared<ov::op::v4::Range>(past_length, total_length, scalar(1), ov::element::i64);
    const auto key_range = std::make_shared<ov::op::v4::Range>(scalar(0), total_length, scalar(1), ov::element::i64);
    const auto query_positions = std::make_shared<ov::op::v0::Unsqueeze>(query_range, scalar(1));
    const auto key_positions = std::make_shared<ov::op::v0::Unsqueeze>(key_range, scalar(0));
    const auto causal = std::make_shared<ov::op::v1::LessEqual>(key_positions, query_positions);
    const auto causal_4d = std::make_shared<ov::op::v0::Unsqueeze>(causal, axes);
    const auto padding = std::make_shared<ov::op::v0::Unsqueeze>(
        std::make_shared<ov::op::v0::Convert>(attention_mask, ov::element::boolean),
        padding_axes);
    const auto valid = std::make_shared<ov::op::v1::LogicalAnd>(causal_4d, padding);
    const auto distance = std::make_shared<ov::op::v1::Subtract>(query_positions, key_positions);
    const auto outside_window = std::make_shared<ov::op::v1::GreaterEqual>(distance, scalar(2));
    const auto outside_window_4d = std::make_shared<ov::op::v0::Unsqueeze>(outside_window, axes);
    const auto fill = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {-65504.0f});
    const auto zero = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.0f});
    const auto window_mask = std::make_shared<ov::op::v1::Select>(outside_window_4d, fill, zero);
    const auto mask = std::make_shared<ov::op::v1::Select>(valid, window_mask, fill);
    for (const auto& node : model->get_ops()) {
        if (const auto sdpa = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(node)) {
            sdpa->input(3).replace_source_output(mask);
            sdpa->get_rt_info().erase(ov::npuw::NPUW_SDPA_MASK_RT_KEY);
        }
    }
    model->validate_nodes_and_infer_types();
    ov::npuw::DetectAttentionMask().run_on_model(model);

    ASSERT_TRUE(ov::npuw::LeftAlignedAttention().run_on_model(model));
    expect_contiguous_mask(evaluate_mask(model, 2u), query_length, 2u, int64_t{2});
}

TEST(LeftAlignedAttentionTest, RejectsMasksThatDependOnPastKvValues) {
    auto model = make_static_attention_model(1u, ov::npuw::NPUW_SDPA_MASK_CAUSAL);
    const auto& parameters = model->get_parameters();
    const auto past_key = *std::find_if(parameters.begin(), parameters.end(), [](const auto& parameter) {
        return parameter->get_friendly_name() == "past_key_values.0.key";
    });
    const auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 1, 2, 3});
    const auto past_sum = std::make_shared<ov::op::v1::ReduceSum>(past_key, axes, false);
    const auto past_shape = std::make_shared<ov::op::v3::ShapeOf>(past_key);
    const auto shape_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2});
    const auto gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    const auto past_length = std::make_shared<ov::op::v8::Gather>(past_shape, shape_axis, gather_axis);
    const auto combined =
        std::make_shared<ov::op::v1::Add>(past_sum,
                                          std::make_shared<ov::op::v0::Convert>(past_length, ov::element::f32));
    for (const auto& node : model->get_ops()) {
        if (const auto sdpa = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(node)) {
            sdpa->input(3).replace_source_output(std::make_shared<ov::op::v1::Add>(sdpa->input_value(3), combined));
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