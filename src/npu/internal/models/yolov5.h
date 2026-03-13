// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef _RKNN_DEMO_YOLOV5_H_
#define _RKNN_DEMO_YOLOV5_H_

#include "npu/internal/npu_common.h"
#include "rknn_api.h"

#include <stdint.h>
#include <vector>

#define OBJ_NAME_MAX_SIZE 64

struct YoloV5PostProcessCtx;

// Infer class count from RKNN YOLOv5 output channels. / Infer 类别 数量 from RKNN YOLOv5 输出 channels.
int get_yolov5_model_num_classes(rknn_app_context_t* app_ctx);
YoloV5PostProcessCtx* create_yolov5_post_process_ctx(const char* label_path, float box_thresh, float nms_thresh,
                                                     int required_num_classes);
void destroy_yolov5_post_process_ctx(YoloV5PostProcessCtx* ctx);
const char* coco_cls_to_name(const YoloV5PostProcessCtx* ctx, int cls_id);
int post_process(rknn_app_context_t* app_ctx, void* outputs, const YoloV5PostProcessCtx* ctx,
                 object_detect_result_list* od_results);
int init_yolov5_model(const char* model_path, rknn_app_context_t* app_ctx);
int release_yolov5_model(rknn_app_context_t* app_ctx);
int inference_yolov5_model(rknn_app_context_t* app_ctx, const YoloV5PostProcessCtx* ctx,
                           object_detect_result_list* od_results);

#endif  // _RKNN_DEMO_YOLOV5_H_

