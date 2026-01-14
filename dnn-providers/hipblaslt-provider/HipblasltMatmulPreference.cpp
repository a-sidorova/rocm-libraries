// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <limits>
#include <vector>

#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "HipblasltMatmulPreference.hpp"
#include "HipblasltUtils.hpp"

namespace hipblaslt_plugin
{

HipblasltMatmulPreference::HipblasltMatmulPreference()
{
    THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatmulPreferenceCreate(&_pref));
}

HipblasltMatmulPreference::HipblasltMatmulPreference(HipblasltMatmulPreference&& other) noexcept
    : _pref(other._pref)
{
    other._pref = nullptr;
}

HipblasltMatmulPreference&
    HipblasltMatmulPreference::operator=(HipblasltMatmulPreference&& other) noexcept
{
    if(this != &other)
    {
        if(_pref != nullptr)
        {
            LOG_ON_HIPBLASLT_FAILURE(hipblasLtMatmulPreferenceDestroy(_pref));
        }

        _pref = other._pref;
        other._pref = nullptr;
    }
    return *this;
}

HipblasltMatmulPreference::~HipblasltMatmulPreference()
{
    if(_pref != nullptr)
    {
        LOG_ON_HIPBLASLT_FAILURE(hipblasLtMatmulPreferenceDestroy(_pref));
    }
}

hipblasLtMatmulPreference_t HipblasltMatmulPreference::matmulPreference() const
{
    return _pref;
}

}
