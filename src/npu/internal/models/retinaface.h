// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (c) 2023 by Rockchip Electronics Co., Ltd. All Rights Reserved. / Copyright (c) 2023 由 Rockchip Electronics Co., Ltd. 全部 Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); / 详见英文原注释。
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at / You 可能 obtain 复制 的 License at
//
//     http://www.apache.org/licenses/LICENSE-2.0 / 详见英文原注释。
//
// Unless required by applicable law or agreed to in writing, software / Unless required 由 applicable law 或 agreed 以 在 writing, 软件
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. / WITHOUT WARRANTIES 或 CONDITIONS 的 ANY KIND, either express 或 implied.
// See the License for the specific language governing permissions and
// limitations under the License. / 详见英文原注释。

#ifndef _RKNN_DEMO_RETINAFACE_RV1106_H_
#define _RKNN_DEMO_RETINAFACE_RV1106_H_

#include "npu/internal/npu_common.h"

#include "rknn_api.h"
#include <stdint.h>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// retinaface / 详见英文原注释。
int init_retinaface_model(const char* model_path, rknn_app_context_t* app_ctx);
int release_retinaface_model(rknn_app_context_t* app_ctx);
int inference_retinaface_model(rknn_app_context_t* app_ctx, object_detect_result_list* od_results);

#endif
