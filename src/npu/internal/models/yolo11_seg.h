// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef _RKNN_DEMO_YOLO11_SEG_H_
#define _RKNN_DEMO_YOLO11_SEG_H_

#include "npu/internal/npu_common.h"

struct Yolo11SegPostProcessCtx;

int get_yolo11_seg_model_num_classes(rknn_app_context_t* app_ctx);
Yolo11SegPostProcessCtx* create_yolo11_seg_post_process_ctx(const char* label_path,
                                                             float box_thresh,
                                                             float nms_thresh,
                                                             int required_num_classes);
void destroy_yolo11_seg_post_process_ctx(Yolo11SegPostProcessCtx* ctx);
const char* coco_cls_to_name_yolo11_seg(const Yolo11SegPostProcessCtx* ctx, int cls_id);

int init_yolo11_seg_model(const char* model_path, rknn_app_context_t* app_ctx);
int release_yolo11_seg_model(rknn_app_context_t* app_ctx);

int inference_yolo11_seg_model(rknn_app_context_t* app_ctx,
                               const Yolo11SegPostProcessCtx* ctx,
                               float letterbox_scale,
                               int letterbox_pad_x,
                               int letterbox_pad_y,
                               object_detect_result_list* od_results);

#endif  // _RKNN_DEMO_YOLO11_SEG_H_

