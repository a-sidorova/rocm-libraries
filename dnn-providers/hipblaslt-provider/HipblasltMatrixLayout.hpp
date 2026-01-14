// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipblaslt/hipblaslt.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>

namespace hipblaslt_plugin
{

class HipblasltMatrixLayout
{
public:
    HipblasltMatrixLayout(const hipdnn_data_sdk::data_objects::TensorAttributes& tensor);

    HipblasltMatrixLayout(const HipblasltMatrixLayout&) = delete;
    HipblasltMatrixLayout& operator=(const HipblasltMatrixLayout&) = delete;

    HipblasltMatrixLayout(HipblasltMatrixLayout&& other) noexcept;
    HipblasltMatrixLayout& operator=(HipblasltMatrixLayout&& other) noexcept;

    ~HipblasltMatrixLayout();

    int64_t uid() const;

    hipblasLtMatrixLayout_t matrixLayout() const;

private:
    int64_t _uid;
    hipblasLtMatrixLayout_t _matrix_layout{nullptr};
};

} // namespace hipblaslt_plugin
