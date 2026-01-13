// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <hipdnn_data_sdk/data_objects/block_scale_quantize_attributes_generated.h>
#include <hipdnn_data_sdk/utilities/json/Common.hpp>

namespace hipdnn_data_sdk::data_objects
{
// NOLINTNEXTLINE(readability-identifier-naming)
inline void to_json(nlohmann::json& bq_json, const BlockScaleQuantizeAttributes& bq)
{
    auto& inputs = bq_json["inputs"] = {};
    auto& outputs = bq_json["outputs"] = {};

    inputs["x_tensor_uid"] = bq.x_tensor_uid();
    inputs["block_size"] = bq.block_size();
    inputs["axis"] = bq.axis();
    inputs["transpose"] = bq.transpose();

    outputs["y_tensor_uid"] = bq.y_tensor_uid();
    outputs["scale_tensor_uid"] = bq.scale_tensor_uid();
}

}
namespace hipdnn_data_sdk::json
{
template <>
inline auto to<data_objects::BlockScaleQuantizeAttributes>(flatbuffers::FlatBufferBuilder& builder,
                                                           const nlohmann::json& entry)
{
    auto& inputs = entry.at("inputs");
    auto& outputs = entry.at("outputs");

    return data_objects::CreateBlockScaleQuantizeAttributes(
        builder,
        inputs.at("x_tensor_uid").get<int64_t>(),
        outputs.at("y_tensor_uid").get<int64_t>(),
        outputs.at("scale_tensor_uid").get<int64_t>(),
        inputs.at("block_size").get<std::optional<int32_t>>(),
        inputs.at("axis").get<std::optional<int64_t>>(),
        inputs.at("transpose").get<bool>());
}

}
