// SPDX-License-Identifier: LGPL-3.0-or-later

// Provide exactly one translation unit for stb image load/write symbols.
// 为 stb 图像读写符号提供唯一的实现编译单元，避免链接阶段出现未定义符号。
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
