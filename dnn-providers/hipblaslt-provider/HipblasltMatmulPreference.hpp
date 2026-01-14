// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipblaslt/hipblaslt.h>

namespace hipblaslt_plugin
{

class HipblasltMatmulPreference
{
public:
    HipblasltMatmulPreference();

    HipblasltMatmulPreference(const HipblasltMatmulPreference&) = delete;
    HipblasltMatmulPreference& operator=(const HipblasltMatmulPreference&) = delete;

    HipblasltMatmulPreference(HipblasltMatmulPreference&& other) noexcept;
    HipblasltMatmulPreference& operator=(HipblasltMatmulPreference&& other) noexcept;

    ~HipblasltMatmulPreference();

    hipblasLtMatmulPreference_t matmulPreference() const;

private:
    hipblasLtMatmulPreference_t _pref = nullptr;
};

} // namespace hipblaslt_plugin
