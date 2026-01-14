// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <functional>
#include <numeric>

#include "HipblasltMatrixLayout.hpp"
#include "HipblasltUtils.hpp"
#include <hipdnn_plugin_sdk/PluginException.hpp>

namespace hipblaslt_plugin
{

HipblasltMatrixLayout::HipblasltMatrixLayout(
    const hipdnn_data_sdk::data_objects::TensorAttributes& tensor)
    : _uid(tensor.uid())
{
    PLUGIN_THROW_IF_NULL(tensor.dims(),
                         HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                         "Tensor dims pointer is null for tensor UID: " + std::to_string(_uid));
    PLUGIN_THROW_IF_NULL(tensor.strides(),
                         HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                         "Tensor strides pointer is null for tensor UID: " + std::to_string(_uid));

    const auto& dims = *tensor.dims();
    const auto& strides = *tensor.strides();
    const auto rank = dims.size();

    PLUGIN_THROW_IF_TRUE(rank < 2,
                         HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                         "Tensor rank must be at least 2 for matrix layout. UID: "
                             + std::to_string(_uid));

    const auto rows = static_cast<uint64_t>(dims[rank - 2]);
    const auto cols = static_cast<uint64_t>(dims[rank - 1]);
    const auto ld = strides[rank - 2];

    THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatrixLayoutCreate(
        &_matrix_layout,
        hipblaslt_utils::tensorDataTypeToHipDataType(tensor.data_type()),
        rows,
        cols,
        ld));

    if(rank > 2)
    {
        const int64_t batch
            = std::accumulate(dims.begin(), dims.end() - 2, int64_t{1}, std::multiplies<int64_t>());

        if(batch > 1)
        {
            const int64_t batch_stride = strides[rank - 3];
            THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatrixLayoutSetAttribute(
                _matrix_layout, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
            THROW_ON_HIPBLASLT_FAILURE(
                hipblasLtMatrixLayoutSetAttribute(_matrix_layout,
                                                  HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                  &batch_stride,
                                                  sizeof(batch_stride)));
        }
    }
}

HipblasltMatrixLayout::HipblasltMatrixLayout(HipblasltMatrixLayout&& other) noexcept
    : _uid(other._uid)
    , _matrix_layout(other._matrix_layout)
{
    other._matrix_layout = nullptr;
}

HipblasltMatrixLayout& HipblasltMatrixLayout::operator=(HipblasltMatrixLayout&& other) noexcept
{
    if(this != &other)
    {
        if(_matrix_layout != nullptr)
        {
            LOG_ON_HIPBLASLT_FAILURE(hipblasLtMatrixLayoutDestroy(_matrix_layout));
        }

        _uid = other._uid;
        _matrix_layout = other._matrix_layout;

        other._matrix_layout = nullptr;
    }
    return *this;
}

HipblasltMatrixLayout::~HipblasltMatrixLayout()
{
    if(_matrix_layout != nullptr)
    {
        LOG_ON_HIPBLASLT_FAILURE(hipblasLtMatrixLayoutDestroy(_matrix_layout));
    }
}

int64_t HipblasltMatrixLayout::uid() const
{
    return _uid;
}

hipblasLtMatrixLayout_t HipblasltMatrixLayout::matrixLayout() const
{
    return _matrix_layout;
}

} // namespace hipblaslt_plugin
