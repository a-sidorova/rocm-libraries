// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Attributes.hpp"
#include "TensorAttributes.hpp"
#include <hipdnn_data_sdk/data_objects/block_scale_quantize_attributes_generated.h>
#include <memory>
#include <unordered_map>
#include <vector>

namespace hipdnn_frontend::graph
{
class BlockScaleQuantizeAttributes : public Attributes<BlockScaleQuantizeAttributes>
{
public:
    enum class InputNames
    {
        X = 0 // Input tensor
    };
    typedef InputNames input_names; // NOLINT(readability-identifier-naming)

    enum class OutputNames
    {
        Y = 0, // Output tensor
        SCALE = 1 // Scale tensor
    };
    typedef OutputNames output_names; // NOLINT(readability-identifier-naming)

    std::unordered_map<InputNames, std::shared_ptr<TensorAttributes>> inputs;
    std::unordered_map<OutputNames, std::shared_ptr<TensorAttributes>> outputs;

    // NOLINTBEGIN(readability-identifier-naming)
    std::optional<int32_t> block_size = std::nullopt;
    std::optional<int64_t> axis = std::nullopt;
    bool transpose = false;
    // NOLINTEND(readability-identifier-naming)

    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_x() const
    {
        return getInput(InputNames::X);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_y() const
    {
        return getOutput(OutputNames::Y);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_scale() const
    {
        return getOutput(OutputNames::SCALE);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::optional<int32_t> get_block_size() const
    {
        return block_size;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::optional<int64_t> get_axis() const
    {
        return axis;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    bool get_transpose() const
    {
        return transpose;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    BlockScaleQuantizeAttributes& set_x(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::X, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BlockScaleQuantizeAttributes& set_x(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::X, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BlockScaleQuantizeAttributes& set_y(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::Y, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BlockScaleQuantizeAttributes& set_y(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::Y, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BlockScaleQuantizeAttributes& set_scale(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::SCALE, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BlockScaleQuantizeAttributes& set_scale(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::SCALE, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BlockScaleQuantizeAttributes& set_block_size(int32_t value)
    {
        block_size = value;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BlockScaleQuantizeAttributes& set_axis(int64_t value)
    {
        axis = value;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BlockScaleQuantizeAttributes& set_transpose(bool value)
    {
        transpose = value;
        return *this;
    }

    flatbuffers::Offset<hipdnn_data_sdk::data_objects::BlockScaleQuantizeAttributes>
        pack_attributes(flatbuffers::FlatBufferBuilder& builder) const // NOLINT
    {
        return hipdnn_data_sdk::data_objects::CreateBlockScaleQuantizeAttributes(
            builder,
            get_x()->get_uid(),
            get_y()->get_uid(),
            get_scale()->get_uid(),
            block_size,
            axis,
            transpose);
    }
};

typedef BlockScaleQuantizeAttributes Block_scale_quantize_attributes;
} // namespace hipdnn_frontend::graph
