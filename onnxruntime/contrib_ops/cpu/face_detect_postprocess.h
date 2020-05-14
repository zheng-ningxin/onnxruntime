// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

struct BoundingBox {
  BoundingBox(float left, float top, float right, float bottom)
      : left_(left), top_(top), right_(right), bottom_(bottom) {}
  BoundingBox() = default;
  float left_;
  float top_;
  float right_;
  float bottom_;
};

struct Point {
  Point(float x, float y)
      : x_(x), y_(y) {}
  Point() = default;
  float x_;
  float y_;
};

class FaceDetectPostProcess final : public OpKernel {
 public:
  explicit FaceDetectPostProcess(const OpKernelInfo& info) : OpKernel(info) {
    score_threshold = info.GetAttrOrDefault<float>("score_threshold", 0);
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  float score_threshold;
};

}  // namespace contrib
}  // namespace onnxruntime
