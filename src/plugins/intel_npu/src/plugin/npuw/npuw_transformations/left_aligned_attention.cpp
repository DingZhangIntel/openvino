// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "left_aligned_attention.hpp"

#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../llm_compiled_model_utils.hpp"
#include "detect_causal_mask.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/scatter_update.hpp"

namespace {

namespace opv0 = ov::op::v0;
namespace opv1 = ov::op::v1;
namespace opv3 = ov::op::v3;
namespace opv4 = ov::op::v4;
namespace opv13 = ov::op::v13;

constexpr size_t kSdpaKeyInput = 1u;
constexpr size_t kSdpaValueInput = 2u;
constexpr size_t kSdpaMaskInput = 3u;
constexpr size_t kSequenceAxis = 2u;

struct ConcatPath {
    std::shared_ptr<opv0::Concat> concat;
    std::vector<std::shared_ptr<ov::Node>> nodes;
};

std::shared_ptr<opv0::Constant> i64_scalar(int64_t value) {
    return opv0::Constant::create(ov::element::i64, ov::Shape{}, {value});
}

std::shared_ptr<opv0::Constant> i64_vector(std::initializer_list<int64_t> values) {
    return opv0::Constant::create(ov::element::i64, ov::Shape{values.size()}, values);
}

bool is_layout_op(const std::shared_ptr<ov::Node>& node) {
    return ov::is_type<opv0::Convert>(node) || ov::is_type<opv0::Unsqueeze>(node) || ov::is_type<opv1::Reshape>(node) ||
           ov::is_type<opv1::Transpose>(node) || ov::is_type<opv1::Multiply>(node) ||
           ov::is_type<opv3::Broadcast>(node);
}

std::optional<ConcatPath> find_concat_path(const ov::Output<ov::Node>& sdpa_input) {
    std::vector<std::shared_ptr<ov::Node>> reverse_path;
    auto node = sdpa_input.get_node_shared_ptr();
    while (node && !ov::is_type<opv0::Concat>(node)) {
        if (!is_layout_op(node) || node->get_input_size() == 0u) {
            return std::nullopt;
        }
        reverse_path.push_back(node);
        node = node->input_value(0).get_node_shared_ptr();
    }

    const auto concat = ov::as_type_ptr<opv0::Concat>(node);
    if (!concat) {
        return std::nullopt;
    }
    std::reverse(reverse_path.begin(), reverse_path.end());
    return ConcatPath{concat, std::move(reverse_path)};
}

bool traces_to_past_kv(const ov::Output<ov::Node>& value) {
    auto node = value.get_node_shared_ptr();
    while (node && is_layout_op(node) && node->get_input_size() > 0u) {
        node = node->input_value(0).get_node_shared_ptr();
    }
    const auto parameter = ov::as_type_ptr<opv0::Parameter>(node);
    return parameter && parameter->get_friendly_name().find("past_key_values") != std::string::npos;
}

bool depends_on_past_kv(const ov::Output<ov::Node>& value) {
    std::unordered_set<ov::Node*> visited;
    std::vector<std::shared_ptr<ov::Node>> pending = {value.get_node_shared_ptr()};
    while (!pending.empty()) {
        const auto node = pending.back();
        pending.pop_back();
        if (!node || !visited.insert(node.get()).second) {
            continue;
        }
        if (const auto parameter = ov::as_type_ptr<opv0::Parameter>(node)) {
            if (parameter->get_friendly_name().find("past_key_values") != std::string::npos) {
                return true;
            }
            continue;
        }
        for (const auto& input : node->input_values()) {
            pending.push_back(input.get_node_shared_ptr());
        }
    }
    return false;
}

bool has_past_kv_input(const std::shared_ptr<opv0::Concat>& concat) {
    return std::any_of(concat->inputs().begin(), concat->inputs().end(), [](const ov::Input<ov::Node>& input) {
        return traces_to_past_kv(input.get_source_output());
    });
}

void assert_supported_mask_dependencies(const ov::Output<ov::Node>& mask,
                                        const std::shared_ptr<opv0::Parameter>& attention_mask,
                                        const std::string& sdpa_name) {
    std::unordered_set<ov::Node*> shape_only_visited;
    std::unordered_set<ov::Node*> value_visited;
    std::vector<std::pair<std::shared_ptr<ov::Node>, bool>> pending = {{mask.get_node_shared_ptr(), false}};
    while (!pending.empty()) {
        const auto [node, shape_only] = pending.back();
        pending.pop_back();
        if (!node ||
            !(shape_only ? shape_only_visited.insert(node.get()).second : value_visited.insert(node.get()).second)) {
            continue;
        }
        if (const auto parameter = ov::as_type_ptr<opv0::Parameter>(node)) {
            const auto& parameter_name = parameter->get_friendly_name();
            const bool is_static_query_shape =
                shape_only && (parameter_name == "input_ids" || parameter_name == "inputs_embeds");
            const bool is_static_position_ids =
                parameter_name == "position_ids" && parameter->get_partial_shape().is_static();
            OPENVINO_ASSERT(parameter == attention_mask || parameter_name == ov::npuw::util::kPastLenParamName ||
                                is_static_query_shape || is_static_position_ids,
                            "LeftAlignedAttention: SDPA mask depends on unsupported model input ",
                            parameter_name,
                            ": ",
                            sdpa_name);
            continue;
        }
        const bool input_is_shape_only = shape_only || ov::is_type<opv3::ShapeOf>(node);
        for (const auto& input : node->input_values()) {
            pending.emplace_back(input.get_node_shared_ptr(), input_is_shape_only);
        }
    }
}

size_t find_present_concat_input(const std::shared_ptr<opv0::Concat>& concat) {
    OPENVINO_ASSERT(concat->get_input_size() == 2u,
                    "LeftAlignedAttention: static KV Concat must have exactly two inputs: ",
                    concat->get_friendly_name());

    const bool first_is_past = traces_to_past_kv(concat->input_value(0));
    const bool second_is_past = traces_to_past_kv(concat->input_value(1));
    OPENVINO_ASSERT(first_is_past != second_is_past,
                    "LeftAlignedAttention: static KV Concat must have exactly one past-KV input: ",
                    concat->get_friendly_name());
    return first_is_past ? 1u : 0u;
}

ov::Output<ov::Node> make_sdpa_kv_input(const ConcatPath& path,
                                        size_t present_input,
                                        size_t query_length,
                                        const std::shared_ptr<opv0::Parameter>& past_len) {
    const auto& concat_shape = path.concat->get_output_partial_shape(0);
    OPENVINO_ASSERT(concat_shape.is_static(),
                    "LeftAlignedAttention: KV Concat output must have static shape: ",
                    path.concat->get_friendly_name());
    const auto rank = concat_shape.rank().get_length();
    const auto axis = ov::util::normalize_axis(path.concat->get_axis(), rank);
    const auto& present_shape = path.concat->get_input_partial_shape(present_input);
    OPENVINO_ASSERT(present_shape.is_static() && static_cast<size_t>(axis) < present_shape.size() &&
                        present_shape[static_cast<size_t>(axis)].get_length() == static_cast<int64_t>(query_length),
                    "LeftAlignedAttention: current KV rows must match the static query length: ",
                    path.concat->get_friendly_name());

    const auto static_indices = std::make_shared<opv4::Range>(i64_scalar(0),
                                                              i64_scalar(static_cast<int64_t>(query_length)),
                                                              i64_scalar(1),
                                                              ov::element::i64);
    const auto indices = std::make_shared<opv1::Add>(past_len, static_indices);
    const auto scatter = std::make_shared<opv3::ScatterUpdate>(path.concat,
                                                               indices,
                                                               path.concat->input_value(present_input),
                                                               i64_scalar(axis));
    scatter->set_friendly_name(path.concat->get_friendly_name() + "/left_aligned_sdpa");
    ov::copy_runtime_info(path.concat, {static_indices, indices, scatter});

    ov::Output<ov::Node> result = scatter;
    for (const auto& layout_node : path.nodes) {
        auto inputs = layout_node->input_values();
        inputs.at(0) = result;
        const auto clone = layout_node->clone_with_new_inputs(inputs);
        clone->set_friendly_name(layout_node->get_friendly_name() + "/left_aligned_sdpa");
        ov::copy_runtime_info(layout_node, clone);
        result = clone;
    }
    return result;
}

std::shared_ptr<opv0::Parameter> find_attention_mask(const std::shared_ptr<ov::Model>& model) {
    for (const auto& parameter : model->get_parameters()) {
        if (parameter->get_output_tensor(0).get_names().count("attention_mask") != 0u ||
            parameter->get_friendly_name() == "attention_mask") {
            return parameter;
        }
    }
    return nullptr;
}

std::shared_ptr<opv0::Parameter> find_or_add_past_len(const std::shared_ptr<ov::Model>& model) {
    for (const auto& parameter : model->get_parameters()) {
        if (parameter->get_friendly_name() != ov::npuw::util::kPastLenParamName) {
            continue;
        }
        OPENVINO_ASSERT(parameter->get_element_type() == ov::element::i64 && parameter->get_partial_shape().rank() == 0,
                        "LeftAlignedAttention: past_len must be a scalar i64 Parameter.");
        return parameter;
    }

    const auto past_len = std::make_shared<opv0::Parameter>(ov::element::i64, ov::Shape{});
    past_len->set_friendly_name(ov::npuw::util::kPastLenParamName);
    past_len->output(0).set_names({ov::npuw::util::kPastLenParamName});
    model->add_parameters({past_len});
    return past_len;
}

std::shared_ptr<opv0::Constant> make_lowest_finite_mask_value(const ov::element::Type& mask_type) {
    if (mask_type == ov::element::f16) {
        return opv0::Constant::create(mask_type, ov::Shape{}, {std::numeric_limits<ov::float16>::lowest()});
    }
    if (mask_type == ov::element::bf16) {
        return opv0::Constant::create(mask_type, ov::Shape{}, {std::numeric_limits<ov::bfloat16>::lowest()});
    }
    if (mask_type == ov::element::f32) {
        return opv0::Constant::create(mask_type, ov::Shape{}, {std::numeric_limits<float>::lowest()});
    }
    if (mask_type == ov::element::f64) {
        return opv0::Constant::create(mask_type, ov::Shape{}, {std::numeric_limits<double>::lowest()});
    }
    OPENVINO_THROW("LeftAlignedAttention: expected a supported real-valued SDPA mask, got ", mask_type);
}

ov::Output<ov::Node> make_contiguous_mask(const std::shared_ptr<opv0::Parameter>& attention_mask,
                                          const std::shared_ptr<opv0::Parameter>& past_len,
                                          const ov::Shape& mask_shape,
                                          const ov::element::Type& mask_type,
                                          int64_t sliding_window,
                                          ov::NodeVector& created_nodes) {
    OPENVINO_ASSERT(mask_shape.size() == 4u && mask_shape[0] == 1u && mask_shape[1] == 1u,
                    "LeftAlignedAttention: expected a [1, 1, query, key] SDPA mask.");
    const auto query_length = mask_shape[2];
    const auto key_length = mask_shape[3];
    const auto& attention_mask_shape = attention_mask->get_shape();
    OPENVINO_ASSERT(
        attention_mask_shape.size() == 2u && attention_mask_shape[0] == 1u && attention_mask_shape[1] == key_length,
        "LeftAlignedAttention: attention_mask must cover the static KV capacity.");

    const auto key_positions = std::make_shared<opv4::Range>(i64_scalar(0),
                                                             i64_scalar(static_cast<int64_t>(key_length)),
                                                             i64_scalar(1),
                                                             ov::element::i64);
    const auto query_offsets = std::make_shared<opv4::Range>(i64_scalar(0),
                                                             i64_scalar(static_cast<int64_t>(query_length)),
                                                             i64_scalar(1),
                                                             ov::element::i64);
    const auto query_positions = std::make_shared<opv1::Add>(past_len, query_offsets);
    const auto query_positions_col = std::make_shared<opv0::Unsqueeze>(query_positions, i64_vector({1}));
    std::shared_ptr<ov::Node> valid = std::make_shared<opv1::LessEqual>(key_positions, query_positions_col);
    if (sliding_window >= 0) {
        const auto lower_bound = std::make_shared<opv1::Subtract>(query_positions_col, i64_scalar(sliding_window));
        const auto within_window = std::make_shared<opv1::Greater>(key_positions, lower_bound);
        valid = std::make_shared<opv1::LogicalAnd>(valid, within_window);
        created_nodes.insert(created_nodes.end(), {lower_bound, within_window});
    }

    const auto valid_4d = std::make_shared<opv0::Unsqueeze>(valid, i64_vector({0, 1}));
    const auto diagonal = std::make_shared<opv1::Equal>(key_positions, query_positions_col);
    const auto diagonal_4d = std::make_shared<opv0::Unsqueeze>(diagonal, i64_vector({0, 1}));
    const auto attention_mask_bool = std::make_shared<opv0::Convert>(attention_mask, ov::element::boolean);
    const auto attention_mask_4d = std::make_shared<opv0::Unsqueeze>(attention_mask_bool, i64_vector({1, 2}));
    const auto padding_or_diagonal = std::make_shared<opv1::LogicalOr>(attention_mask_4d, diagonal_4d);
    const auto combined = std::make_shared<opv1::LogicalAnd>(valid_4d, padding_or_diagonal);
    created_nodes.insert(created_nodes.end(),
                         {key_positions,
                          query_offsets,
                          query_positions,
                          query_positions_col,
                          valid,
                          valid_4d,
                          diagonal,
                          diagonal_4d,
                          attention_mask_bool,
                          attention_mask_4d,
                          padding_or_diagonal,
                          combined});

    if (mask_type == ov::element::boolean) {
        return combined;
    }
    OPENVINO_ASSERT(mask_type.is_real(),
                    "LeftAlignedAttention: expected a boolean or real-valued SDPA mask, got ",
                    mask_type);
    const auto zero = opv0::Constant::create(mask_type, ov::Shape{}, {0.0f});
    const auto masked = make_lowest_finite_mask_value(mask_type);
    const auto additive = std::make_shared<opv1::Select>(combined, zero, masked);
    created_nodes.insert(created_nodes.end(), {zero, masked, additive});
    return additive;
}

}  // namespace

bool ov::npuw::LeftAlignedAttention::run_on_model(const std::shared_ptr<ov::Model>& model) {
    const auto attention_mask = find_attention_mask(model);

    bool rewritten = false;
    std::shared_ptr<opv0::Parameter> past_len;
    for (const auto& node : model->get_ordered_ops()) {
        const auto sdpa = ov::as_type_ptr<opv13::ScaledDotProductAttention>(node);
        if (!sdpa) {
            continue;
        }

        const auto key_path = find_concat_path(sdpa->input_value(kSdpaKeyInput));
        const auto value_path = find_concat_path(sdpa->input_value(kSdpaValueInput));
        const bool key_uses_past_kv = key_path && has_past_kv_input(key_path->concat);
        const bool value_uses_past_kv = value_path && has_past_kv_input(value_path->concat);
        const bool key_depends_on_past_kv = depends_on_past_kv(sdpa->input_value(kSdpaKeyInput));
        const bool value_depends_on_past_kv = depends_on_past_kv(sdpa->input_value(kSdpaValueInput));
        if (!key_depends_on_past_kv && !value_depends_on_past_kv) {
            continue;
        }

        OPENVINO_ASSERT(sdpa->get_input_size() > kSdpaMaskInput,
                        "LeftAlignedAttention: past-KV SDPA has no explicit mask: ",
                        sdpa->get_friendly_name());
        OPENVINO_ASSERT(key_depends_on_past_kv && value_depends_on_past_kv && key_uses_past_kv && value_uses_past_kv,
                        "LeftAlignedAttention: past-KV SDPA must expose both K and V through supported Concats: ",
                        sdpa->get_friendly_name());
        OPENVINO_ASSERT(attention_mask,
                        "LeftAlignedAttention: past-KV SDPA requires an attention_mask Parameter: ",
                        sdpa->get_friendly_name());
        const auto mask_kind = sdpa->get_rt_info().find(ov::npuw::NPUW_SDPA_MASK_RT_KEY);
        OPENVINO_ASSERT(mask_kind != sdpa->get_rt_info().end(),
                        "LeftAlignedAttention: past-KV SDPA has an unsupported attention mask: ",
                        sdpa->get_friendly_name());
        assert_supported_mask_dependencies(sdpa->input_value(kSdpaMaskInput),
                                           attention_mask,
                                           sdpa->get_friendly_name());

        const auto& query_shape = sdpa->get_input_partial_shape(0);
        const auto& key_shape = sdpa->get_input_partial_shape(kSdpaKeyInput);
        const auto& mask_shape = sdpa->get_input_partial_shape(kSdpaMaskInput);
        OPENVINO_ASSERT(query_shape.is_static() && key_shape.is_static() && mask_shape.is_static() &&
                            query_shape.size() == 4u && key_shape.size() == 4u && mask_shape.size() == 4u &&
                            query_shape[kSequenceAxis] == mask_shape[2] && key_shape[kSequenceAxis] == mask_shape[3],
                        "LeftAlignedAttention: recognized static SDPA has unsupported Q/K/mask shapes: ",
                        sdpa->get_friendly_name());

        const auto query_length = static_cast<size_t>(query_shape[kSequenceAxis].get_length());
        if (!past_len) {
            past_len = find_or_add_past_len(model);
        }
        if (m_rewrite_kv_cache) {
            const auto present_key_input = find_present_concat_input(key_path->concat);
            const auto present_value_input = find_present_concat_input(value_path->concat);
            sdpa->input(kSdpaKeyInput)
                .replace_source_output(make_sdpa_kv_input(*key_path, present_key_input, query_length, past_len));
            sdpa->input(kSdpaValueInput)
                .replace_source_output(make_sdpa_kv_input(*value_path, present_value_input, query_length, past_len));
        }

        ov::NodeVector mask_nodes;
        const auto contiguous_mask = make_contiguous_mask(attention_mask,
                                                          past_len,
                                                          mask_shape.to_shape(),
                                                          sdpa->get_input_element_type(kSdpaMaskInput),
                                                          mask_kind->second.as<int64_t>(),
                                                          mask_nodes);
        ov::copy_runtime_info({sdpa, sdpa->input_value(kSdpaMaskInput).get_node_shared_ptr()}, mask_nodes);
        sdpa->input(kSdpaMaskInput).replace_source_output(contiguous_mask);
        rewritten = true;
    }

    model->validate_nodes_and_infer_types();
    return rewritten;
}