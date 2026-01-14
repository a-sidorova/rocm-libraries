// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HipblasltMatrixLayout.hpp"
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

using namespace hipblaslt_plugin;

TEST(TestHipblasltMatrixLayout, CanCreateAndDestroy)
{
    auto builder = hipdnn_test_sdk::utilities::createValidMatmulGraph();
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& tensorMap = graph.getTensorMap();
    ASSERT_FALSE(tensorMap.empty());
    const auto* tensorAttr = tensorMap.begin()->second;
    ASSERT_NE(tensorAttr, nullptr);

    EXPECT_NO_THROW({
        HipblasltMatrixLayout matLayout(*tensorAttr);
        EXPECT_EQ(matLayout.uid(), tensorAttr->uid());
        EXPECT_NE(matLayout.matrixLayout(), nullptr);
    });
}

TEST(TestHipblasltMatrixLayout, TensorDescriptorIsValid)
{
    auto builder = hipdnn_test_sdk::utilities::createValidMatmulGraph();
    hipdnn_plugin_sdk::GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const auto& tensorMap = graph.getTensorMap();
    ASSERT_FALSE(tensorMap.empty());
    const auto* tensorAttr = tensorMap.begin()->second;
    HipblasltMatrixLayout matLayout(*tensorAttr);

    // The descriptor should be non-null and can be used in HipBLASLt API calls
    EXPECT_NE(matLayout.matrixLayout(), nullptr);
}
