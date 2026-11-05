// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov::npuw {

// Rewrites static SDPA masks for a runtime past length. KV layout rewriting is
// optional because attention isolation modes retain the original K/V graph.
class LeftAlignedAttention : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::LeftAlignedAttention");
    explicit LeftAlignedAttention(bool rewrite_kv_cache = true) : m_rewrite_kv_cache(rewrite_kv_cache) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    bool m_rewrite_kv_cache;
};

}  // namespace ov::npuw