// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include "hipdnn_frontend/attributes/BlockScaleQuantizeAttributes.hpp"
#include <gtest/gtest.h>

using hipdnn_frontend::DataType;
using hipdnn_frontend::graph::BlockScaleQuantizeAttributes;
using hipdnn_frontend::graph::TensorAttributes;

TEST(TestBlockScaleQuantizeAttributes, CreateAttributes)
{
    BlockScaleQuantizeAttributes bqAttributes;

    bqAttributes.set_x(std::make_shared<TensorAttributes>());
    bqAttributes.set_y(std::make_shared<TensorAttributes>());
    bqAttributes.set_scale(std::make_shared<TensorAttributes>());
    bqAttributes.set_block_size(8).set_axis(1).set_transpose(false);

    auto xTensor = bqAttributes.get_x();
    xTensor->set_uid(1)
        .set_name("XTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({32, 128})
        .set_stride({128, 1});

    auto yTensor = bqAttributes.get_y();
    yTensor->set_uid(2)
        .set_name("YTensor")
        .set_data_type(DataType::FP8_E4M3)
        .set_dim({32, 128})
        .set_stride({128, 1});

    auto scaleTensor = bqAttributes.get_scale();
    scaleTensor->set_uid(3)
        .set_name("ScaleTensor")
        .set_data_type(DataType::FP8_E8M0)
        .set_dim({32, 16})
        .set_stride({16, 1});

    EXPECT_EQ(xTensor->get_uid(), 1);
    EXPECT_EQ(xTensor->get_name(), "XTensor");
    EXPECT_EQ(xTensor->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(xTensor->get_dim(), (std::vector<int64_t>{32, 128}));
    EXPECT_EQ(xTensor->get_stride(), (std::vector<int64_t>{128, 1}));

    EXPECT_EQ(yTensor->get_uid(), 2);
    EXPECT_EQ(yTensor->get_name(), "YTensor");
    EXPECT_EQ(yTensor->get_data_type(), DataType::FP8_E4M3);
    EXPECT_EQ(yTensor->get_dim(), (std::vector<int64_t>{32, 128}));
    EXPECT_EQ(yTensor->get_stride(), (std::vector<int64_t>{128, 1}));

    EXPECT_EQ(scaleTensor->get_uid(), 3);
    EXPECT_EQ(scaleTensor->get_name(), "ScaleTensor");
    EXPECT_EQ(scaleTensor->get_data_type(), DataType::FP8_E8M0);
    EXPECT_EQ(scaleTensor->get_dim(), (std::vector<int64_t>{32, 16}));
    EXPECT_EQ(scaleTensor->get_stride(), (std::vector<int64_t>{16, 1}));

    ASSERT_TRUE(bqAttributes.get_block_size().has_value());
    EXPECT_EQ(*bqAttributes.get_block_size(), 8);
    ASSERT_TRUE(bqAttributes.get_axis().has_value());
    EXPECT_EQ(*bqAttributes.get_axis(), 1);
    EXPECT_FALSE(bqAttributes.get_transpose());
}

TEST(TestBlockScaleQuantizeAttributes, PackAttributes)
{
    BlockScaleQuantizeAttributes bqAttributes;
    bqAttributes.set_block_size(16).set_axis(0).set_transpose(false);

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(10);
    bqAttributes.set_x(xTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_uid(20);
    bqAttributes.set_y(yTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_uid(30);
    bqAttributes.set_scale(scaleTensor);

    // Populate tensor metadata and ensure getters alias the same objects
    xTensor->set_name("XPack")
        .set_data_type(DataType::FLOAT)
        .set_dim({32, 128})
        .set_stride({128, 1});
    yTensor->set_name("YPack")
        .set_data_type(DataType::FP8_E4M3)
        .set_dim({32, 128})
        .set_stride({128, 1});
    scaleTensor->set_name("ScalePack")
        .set_data_type(DataType::FP8_E8M0)
        .set_dim({32, 16})
        .set_stride({16, 1});

    EXPECT_EQ(bqAttributes.get_x(), xTensor);
    EXPECT_EQ(bqAttributes.get_y(), yTensor);
    EXPECT_EQ(bqAttributes.get_scale(), scaleTensor);

    EXPECT_EQ(xTensor->get_name(), "XPack");
    EXPECT_EQ(xTensor->get_data_type(), DataType::FLOAT);
    EXPECT_EQ(xTensor->get_dim(), (std::vector<int64_t>{32, 128}));
    EXPECT_EQ(xTensor->get_stride(), (std::vector<int64_t>{128, 1}));

    EXPECT_EQ(yTensor->get_name(), "YPack");
    EXPECT_EQ(yTensor->get_data_type(), DataType::FP8_E4M3);
    EXPECT_EQ(yTensor->get_dim(), (std::vector<int64_t>{32, 128}));
    EXPECT_EQ(yTensor->get_stride(), (std::vector<int64_t>{128, 1}));

    EXPECT_EQ(scaleTensor->get_name(), "ScalePack");
    EXPECT_EQ(scaleTensor->get_data_type(), DataType::FP8_E8M0);
    EXPECT_EQ(scaleTensor->get_dim(), (std::vector<int64_t>{32, 16}));
    EXPECT_EQ(scaleTensor->get_stride(), (std::vector<int64_t>{16, 1}));

    flatbuffers::FlatBufferBuilder builder;
    auto packedAttributes = bqAttributes.pack_attributes(builder);
    builder.Finish(packedAttributes);

    auto buffer = builder.GetBufferPointer();
    auto bqFb
        = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::BlockScaleQuantizeAttributes>(buffer);

    EXPECT_EQ(bqFb->x_tensor_uid(), 10);
    EXPECT_EQ(bqFb->y_tensor_uid(), 20);
    EXPECT_EQ(bqFb->scale_tensor_uid(), 30);
    ASSERT_TRUE(bqFb->block_size().has_value());
    EXPECT_EQ(*bqFb->block_size(), 16);
    ASSERT_TRUE(bqFb->axis().has_value());
    EXPECT_EQ(*bqFb->axis(), 0);
    EXPECT_FALSE(bqFb->transpose());
}

TEST(TestBlockScaleQuantizeAttributes, DefaultValues)
{
    BlockScaleQuantizeAttributes bqAttributes;

    EXPECT_EQ(bqAttributes.get_x(), nullptr);
    EXPECT_EQ(bqAttributes.get_y(), nullptr);
    EXPECT_EQ(bqAttributes.get_scale(), nullptr);
    EXPECT_FALSE(bqAttributes.get_block_size().has_value());
    EXPECT_FALSE(bqAttributes.get_axis().has_value());
    EXPECT_FALSE(bqAttributes.get_transpose());
}

TEST(TestBlockScaleQuantizeAttributes, SetXMove)
{
    BlockScaleQuantizeAttributes bqAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(100).set_name("MovedX").set_data_type(DataType::FLOAT);

    auto rawPtr = xTensor.get();
    bqAttributes.set_x(std::move(xTensor));
    EXPECT_EQ(xTensor, nullptr);

    auto retrievedTensor = bqAttributes.get_x();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestBlockScaleQuantizeAttributes, SetYMove)
{
    BlockScaleQuantizeAttributes bqAttributes;

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_uid(110).set_name("MovedY").set_data_type(DataType::FP8_E4M3);

    auto rawPtr = yTensor.get();
    bqAttributes.set_y(std::move(yTensor));
    EXPECT_EQ(yTensor, nullptr);

    auto retrievedTensor = bqAttributes.get_y();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestBlockScaleQuantizeAttributes, SetScaleMove)
{
    BlockScaleQuantizeAttributes bqAttributes;

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_uid(120).set_name("MovedScale").set_data_type(DataType::FP8_E8M0);

    auto rawPtr = scaleTensor.get();
    bqAttributes.set_scale(std::move(scaleTensor));
    EXPECT_EQ(scaleTensor, nullptr);

    auto retrievedTensor = bqAttributes.get_scale();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestBlockScaleQuantizeAttributes, SetTensorsConstRef)
{
    BlockScaleQuantizeAttributes bqAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(400).set_name("XConstRef");

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_uid(200).set_name("YConstRef");

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_uid(300).set_name("ScaleConstRef");

    bqAttributes.set_x(xTensor);
    bqAttributes.set_y(yTensor);
    bqAttributes.set_scale(scaleTensor);

    EXPECT_NE(xTensor, nullptr);
    EXPECT_NE(yTensor, nullptr);
    EXPECT_NE(scaleTensor, nullptr);
    EXPECT_EQ(bqAttributes.get_x(), xTensor);
    EXPECT_EQ(bqAttributes.get_y(), yTensor);
    EXPECT_EQ(bqAttributes.get_scale(), scaleTensor);
}
