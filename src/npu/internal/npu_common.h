// SPDX-License-Identifier: LGPL-3.0-or-later
// npu_common.h / 详见英文原注释。
#ifndef __NPU_COMMON_H__
#define __NPU_COMMON_H__

#include "rknn_api.h"

#define OBJ_NUMB_MAX_SIZE 128
#define FACE_LANDMARK_MAX_SIZE 5
#define POSE_KEYPOINT_MAX_SIZE 17
#define SEG_MASK_POINT_MAX_SIZE 128

typedef struct {
    int left;
    int top;
    int right;
    int bottom;
} image_rect_t;

typedef struct {
    int x;
    int y;
} point_t;

typedef struct {
    char* dma_buf_virt_addr;
    int dma_buf_fd;
    int size;
} rknn_dma_buf;

typedef struct {
    image_rect_t box;
    float prop;
    int cls_id;
    point_t point[FACE_LANDMARK_MAX_SIZE];       // RetinaFace: 5 landmarks
    int keypoint_count;                          // YOLO11-pose: number of valid keypoints
    float keypoints[POSE_KEYPOINT_MAX_SIZE][3]; // [x, y, score]
    int mask_point_count;                        // YOLO11-seg: number of valid contour points
    float mask_points[SEG_MASK_POINT_MAX_SIZE][2]; // [x, y] contour points in image coordinates
} object_detect_result;

typedef struct {
    int id;
    int count;
    object_detect_result results[OBJ_NUMB_MAX_SIZE];
} object_detect_result_list;

typedef struct rknn_app_context_t {
    rknn_context rknn_ctx;
    rknn_tensor_mem* max_mem;
    rknn_tensor_mem* net_mem;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;
    rknn_tensor_mem* input_mems[1];
    rknn_tensor_mem* output_mems[9];
    int model_width;
    int model_height;
    int model_channel;
    bool is_quant;
} rknn_app_context_t;

#endif // __NPU_COMMON_H__

