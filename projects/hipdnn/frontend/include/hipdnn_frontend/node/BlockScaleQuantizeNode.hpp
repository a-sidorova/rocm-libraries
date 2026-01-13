// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Node.hpp"
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/BlockScaleQuantizeAttributes.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>

namespace hipdnn_frontend::graph
{
class BlockScaleQuantizeNode : public BaseNode<BlockScaleQuantizeNode>
{

public:
    BlockScaleQuantizeAttributes attributes;

    BlockScaleQuantizeNode(BlockScaleQuantizeAttributes&& blockScaleQuantizeAttributes,
                           const GraphAttributes& graphAttrs)
        : BaseNode(graphAttrs)
        , attributes(std::move(blockScaleQuantizeAttributes))
    {
    }

    Error pre_validate_node() const override
    {
        // Validate tensor pointers
        const auto x = attributes.get_x();
        const auto y = attributes.get_y();
        const auto scale = attributes.get_scale();

        HIPDNN_CHECK_ERROR(validateTensors(x, y, scale));

        return {};
    }

    Error infer_properties_node() override
    {
        // Validate tensor pointers
        const auto x = attributes.get_x();
        const auto y = attributes.get_y();
        const auto scale = attributes.get_scale();

        HIPDNN_CHECK_ERROR(validateTensors(x, y, scale));
        HIPDNN_CHECK_ERROR(attributes.fill_from_context(graph_attributes));

        return {};
    }

    flatbuffers::Offset<hipdnn_data_sdk::data_objects::Node>
        pack_node(flatbuffers::FlatBufferBuilder& builder) const override
    {
        return hipdnn_data_sdk::data_objects::CreateNodeDirect(
            builder,
            attributes.get_name().c_str(),
            toSdkType(attributes.compute_data_type),
            hipdnn_data_sdk::data_objects::NodeAttributes::BlockScaleQuantizeAttributes,
            attributes.pack_attributes(builder).Union());
    }

private:
    static Error validateTensors(const std::shared_ptr<TensorAttributes>& x,
                                 const std::shared_ptr<TensorAttributes>& y,
                                 const std::shared_ptr<TensorAttributes>& scale)
    {
        HIPDNN_RETURN_IF_FALSE(
            x, ErrorCode::ATTRIBUTE_NOT_SET, std::string("BlockScaleQuantizeNode missing X input"));
        HIPDNN_RETURN_IF_FALSE(
            y, ErrorCode::ATTRIBUTE_NOT_SET, std::string("BlockScaleQuantizeNode missing Y input"));
        HIPDNN_RETURN_IF_FALSE(scale,
                               ErrorCode::ATTRIBUTE_NOT_SET,
                               std::string("BlockScaleQuantizeNode missing Scale output"));

        return {};
    }
};
}
