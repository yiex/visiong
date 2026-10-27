// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef _RKNN_DEMO_YOLO26_H_
#define _RKNN_DEMO_YOLO26_H_

#include "npu/internal/npu_common.h"

struct Yolo26PostProcessCtx;

Yolo26PostProcessCtx* create_yolo26_post_process_ctx(rknn_app_context_t* app_ctx, const char* label_path,
                                                     float box_thresh);
void destroy_yolo26_post_process_ctx(Yolo26PostProcessCtx* ctx);
const char* coco_cls_to_name_yolo26(const Yolo26PostProcessCtx* ctx, int cls_id);

int init_yolo26_model(const char* model_path, rknn_app_context_t* app_ctx);
int release_yolo26_model(rknn_app_context_t* app_ctx);

int post_process_yolo26(rknn_app_context_t* app_ctx, Yolo26PostProcessCtx* ctx, float letterbox_scale,
                        int letterbox_pad_x, int letterbox_pad_y, object_detect_result_list* od_results);
int inference_yolo26_model(rknn_app_context_t* app_ctx, Yolo26PostProcessCtx* ctx, float letterbox_scale,
                           int letterbox_pad_x, int letterbox_pad_y, object_detect_result_list* od_results);

#endif // _RKNN_DEMO_YOLO26_H_
