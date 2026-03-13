// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_C_COMPAT_BASE64_H
#define VISIONG_C_COMPAT_BASE64_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Returns the number of bytes written, excluding the trailing '\0'. / 返回 number 的 bytes written, excluding trailing '\0'.
size_t base64_encode(char* out, const uint8_t* data, size_t bytes);

// Returns the number of bytes written to the output buffer. / 返回 number 的 bytes written 以 输出 缓冲区.
size_t base64_decode(uint8_t* out, const char* in, size_t inlen);

#ifdef __cplusplus
}
#endif

#endif  // VISIONG_C_COMPAT_BASE64_H
