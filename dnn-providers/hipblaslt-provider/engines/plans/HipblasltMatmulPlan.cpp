// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>
#include <string>

#include <hipdnn_data_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_data_sdk/utilities/ScopedResource.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "HipblasltMatmulPlan.hpp"
#include "HipdnnEnginePluginHandle.hpp"

namespace hipblaslt_plugin
{

hipblasOperation_t MatmulParams::getTrans(const hipdnn_data_sdk::data_objects::TensorAttributes& t)
{
    const auto& strides = *t.strides();
    const std::string tensorName = t.name()->str();
    PLUGIN_THROW_IF_FALSE(strides.size() > 1,
                          HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                          "Unsupported stride for input matrix: " + tensorName);
    if(strides[strides.size() - 2] == 1)
    {
        return HIPBLAS_OP_T;
    }
    else if(strides[strides.size() - 1] == 1)
    {
        return HIPBLAS_OP_N;
    }
    else
    {
        PLUGIN_THROW_IF_TRUE(true,
                             HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                             "Unsupported stride for input matrix: " + tensorName);
    }
    return HIPBLAS_OP_N; // Unreachable, placates compiler
}

hipblasComputeType_t MatmulParams::getComputeDataType(
    [[maybe_unused]] const hipdnn_data_sdk::data_objects::MatmulAttributes& attributes)
{
    // TODO: Add more generic logic for compute data type
    return HIPBLAS_COMPUTE_32F;
}

MatmulParams::MatmulParams(
    [[maybe_unused]] const hipdnn_data_sdk::data_objects::MatmulAttributes& attributes,
    [[maybe_unused]] const std::
        unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>& tensorMap)
    : _matrixLayoutA(hipblaslt_utils::createMatrixLayout(tensorMap, attributes.a_tensor_uid()))
    , _matrixLayoutB(hipblaslt_utils::createMatrixLayout(tensorMap, attributes.b_tensor_uid()))
    , _matrixLayoutC(hipblaslt_utils::createMatrixLayout(tensorMap, attributes.c_tensor_uid()))
    , _matrixLayoutD(hipblaslt_utils::createMatrixLayout(tensorMap, attributes.c_tensor_uid()))
{
    const auto& attrA = hipblaslt_utils::findTensorAttributes(tensorMap, attributes.a_tensor_uid());
    const auto& attrB = hipblaslt_utils::findTensorAttributes(tensorMap, attributes.b_tensor_uid());
    const auto& attrC = hipblaslt_utils::findTensorAttributes(tensorMap, attributes.c_tensor_uid());

    _matmulDesc = HipblasltMatmulDesc(
        getTrans(attrA), getTrans(attrB), getComputeDataType(attributes), HIP_R_32F);

    _tensorsValid = (!attrA.virtual_() && !attrB.virtual_() && !attrC.virtual_());
}

const HipblasltMatrixLayout& MatmulParams::a() const
{
    return _matrixLayoutA;
}

const HipblasltMatrixLayout& MatmulParams::b() const
{
    return _matrixLayoutB;
}

const HipblasltMatrixLayout& MatmulParams::c() const
{
    return _matrixLayoutC;
}

const HipblasltMatrixLayout& MatmulParams::d() const
{
    return _matrixLayoutD;
}

const HipblasltMatmulDesc& MatmulParams::desc() const
{
    return _matmulDesc;
}

bool MatmulParams::validTensors() const
{
    return _tensorsValid;
}

MatmulPlan::MatmulPlan(const HipdnnEnginePluginHandle& handle, MatmulParams&& params)
    : _params(std::move(params))
{
    size_t max_workspace_size = 0;
    hipblasLtMatmulPreference_t pref;
    THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatmulPreferenceCreate(&pref));
    THROW_ON_HIPBLASLT_FAILURE(
        hipblasLtMatmulPreferenceSetAttribute(pref,
                                              HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &max_workspace_size,
                                              sizeof(max_workspace_size)));

    constexpr int request_solutions = 1;
    hipblasLtMatmulHeuristicResult_t heuristicResult[request_solutions];
    int returnedAlgoCount = 0;
    THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatmulAlgoGetHeuristic(handle.hipblasltHandle,
                                                               _params.desc().matmulDesc(),
                                                               _params.a().matrixLayout(),
                                                               _params.b().matrixLayout(),
                                                               _params.c().matrixLayout(),
                                                               _params.d().matrixLayout(),
                                                               pref,
                                                               request_solutions,
                                                               heuristicResult,
                                                               &returnedAlgoCount));

    PLUGIN_THROW_IF_FALSE(returnedAlgoCount > 1,
                          HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                          "hipBLASLt has not found algorithm!");

    _heuristicResult = heuristicResult[0];
    _workspaceSize = _heuristicResult.workspaceSize;
}

size_t MatmulPlan::getWorkspaceSize([[maybe_unused]] const HipdnnEnginePluginHandle& handle) const
{
    return _workspaceSize;
}

void MatmulPlan::execute([[maybe_unused]] const HipdnnEnginePluginHandle& handle,
                         [[maybe_unused]] const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                         [[maybe_unused]] uint32_t numDeviceBuffers,
                         [[maybe_unused]] void* workspace) const
{
    auto aBuffer
        = hipblaslt_utils::findDeviceBuffer(_params.a().uid(), deviceBuffers, numDeviceBuffers);
    auto bBuffer
        = hipblaslt_utils::findDeviceBuffer(_params.b().uid(), deviceBuffers, numDeviceBuffers);
    auto cBuffer
        = hipblaslt_utils::findDeviceBuffer(_params.c().uid(), deviceBuffers, numDeviceBuffers);
    auto dBuffer = cBuffer;

    THROW_ON_HIPBLASLT_FAILURE(hipblasLtMatmul(handle.hipblasltHandle,
                                               _params.desc().matmulDesc(),
                                               &_alpha,
                                               aBuffer.ptr,
                                               _params.a().matrixLayout(),
                                               bBuffer.ptr,
                                               _params.b().matrixLayout(),
                                               &_beta,
                                               cBuffer.ptr,
                                               _params.c().matrixLayout(),
                                               dBuffer.ptr,
                                               _params.d().matrixLayout(),
                                               &_heuristicResult.algo,
                                               workspace,
                                               _workspaceSize,
                                               handle.getStream()));
}

} // namespace hipblaslt_plugin
