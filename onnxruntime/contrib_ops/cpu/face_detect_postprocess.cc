/* Copyright Microsoft.
==============================================================================*/

#include "face_detect_postprocess.h"
#include <queue>

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    FaceDetectPostProcess,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder(),
    FaceDetectPostProcess);

Status FaceDetectPostProcess::Compute(OpKernelContext* ctx) const {

  const auto& rect_delta = ctx->Input<Tensor>(0);
  const auto& rect_delta_shape = rect_delta->Shape();
  const float* rect_delta_data = rect_delta->Data<float>();

  const auto& landmark_delta = ctx->Input<Tensor>(1);
  const auto& landmark_delta_shape = landmark_delta->Shape();
  const float* landmark_delta_data = landmark_delta->Data<float>();

  const auto& score_input = ctx->Input<Tensor>(2);
  const auto& score_shape = score_input->Shape();
  const float* score_data = score_input->Data<float>();

  const int64_t hight = rect_delta_shape[1];
  const int64_t width = rect_delta_shape[2];
  const int64_t anchor_num = score_shape[3];
  const int64_t rect_len = static_cast<int64_t>(rect_delta_shape[3] / anchor_num);
  const int64_t landmark_len = static_cast<int64_t>(landmark_delta_shape[3] / anchor_num);

  std::vector<float> score_list;
  std::vector<BoundingBox> rect_list;
  std::vector<Point> landmarks_list;

  int feature_map_stride = 4;
  std::vector<int> anchor_proposals{ 16, 32, 64, 128 };

  // TODO: parallel for hight * width
  for (int64_t h = 0; h < hight; ++h) {
    for (int64_t w = 0; w < width; ++w) {
      for (int64_t k = 0; k < anchor_num; ++k) {
        auto offset = h * width * anchor_num + w * anchor_num;
        auto score = score_data[offset + k];
        if (score < score_threshold) {
          continue;
        }

        int prop_w_h = anchor_proposals[k];
        int64_t prop_center_x = w * feature_map_stride;
        int64_t prop_center_y = h * feature_map_stride;

        float rect_center_x = rect_delta_data[offset * rect_len + k * rect_len] * prop_w_h + prop_center_x;
        float rect_center_y = rect_delta_data[offset * rect_len + k * rect_len + 1] * prop_w_h + prop_center_y;
        float rect_w = std::exp(rect_delta_data[offset * rect_len + k * rect_len + 2]) * prop_w_h;
        float rect_h = std::exp(rect_delta_data[offset * rect_len + k * rect_len + 3]) * prop_w_h;
        float left = rect_center_x - static_cast<float>(rect_w / 2.0);
        float top = rect_center_y - static_cast<float>(rect_h / 2.0);
        float right = rect_center_x + static_cast<float>(rect_w / 2.0);
        float bottom = rect_center_y + static_cast<float>(rect_h / 2.0);

        for (int64_t i = 0; i < static_cast<int64_t>(landmark_len / 2); ++i) {
          float x = landmark_delta_data[offset * landmark_len + k * landmark_len + 2 * i] * prop_w_h + prop_center_x;
          float y = landmark_delta_data[offset * landmark_len + k * landmark_len + 2 * i + 1] * prop_w_h + prop_center_y;
          landmarks_list.emplace_back(x, y);
        }

        score_list.emplace_back(score);
        rect_list.emplace_back(top, left, bottom, right);
      }
    }
  }

  //copy to output
  auto output_count = score_list.size();

  Tensor* boxes_output = ctx->Output(0, {static_cast<int64_t>(output_count), 4});
  ORT_ENFORCE(boxes_output != nullptr);
  memcpy(boxes_output->MutableData<float>(), rect_list.data(), output_count * sizeof(BoundingBox));

  Tensor* scores_output = ctx->Output(1, {static_cast<int64_t>(output_count)});
  ORT_ENFORCE(scores_output != nullptr);
  memcpy(scores_output->MutableData<float>(), score_list.data(), output_count * sizeof(float));

  Tensor* landmarks_output = ctx->Output(2, {static_cast<int64_t>(output_count), 10});
  ORT_ENFORCE(landmarks_output != nullptr);
  memcpy(landmarks_output->MutableData<float>(), landmarks_list.data(), output_count * 10 * sizeof(float));

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
