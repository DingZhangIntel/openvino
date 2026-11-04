// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov::npuw {

// Replaces tail-layout static attention with a runtime-past-length layout for
// chunked prefill and generation. The original KV Concat remains on the cache
// output route; only its SDPA-facing branch is rewritten.
class LeftAlignedAttention : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::LeftAlignedAttention");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::npuw