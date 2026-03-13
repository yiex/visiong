// SPDX-License-Identifier: LGPL-3.0-or-later
// Minimal base64 implementation for media-server/librtsp SDP helpers. / 最小化 base64 实现 用于 media-server/librtsp SDP 辅助函数.
// API matches src/c_compat/include/base64.h

#include "base64.h"

#include <string.h>

static const char k_b64_table[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static unsigned char b64_rev(unsigned char c) {
    if (c >= 'A' && c <= 'Z') return (unsigned char)(c - 'A');
    if (c >= 'a' && c <= 'z') return (unsigned char)(c - 'a' + 26);
    if (c >= '0' && c <= '9') return (unsigned char)(c - '0' + 52);
    if (c == '+') return 62;
    if (c == '/') return 63;
    return 0xFF;
}

size_t base64_encode(char* out, const uint8_t* data, size_t bytes) {
    size_t i = 0;
    size_t o = 0;
    if (!out || (!data && bytes != 0)) return 0;

    while (i + 2 < bytes) {
        uint32_t v = ((uint32_t)data[i] << 16) | ((uint32_t)data[i + 1] << 8) | (uint32_t)data[i + 2];
        out[o++] = k_b64_table[(v >> 18) & 0x3F];
        out[o++] = k_b64_table[(v >> 12) & 0x3F];
        out[o++] = k_b64_table[(v >> 6) & 0x3F];
        out[o++] = k_b64_table[v & 0x3F];
        i += 3;
    }

    if (i < bytes) {
        uint32_t v = (uint32_t)data[i] << 16;
        out[o++] = k_b64_table[(v >> 18) & 0x3F];
        if (i + 1 < bytes) {
            v |= (uint32_t)data[i + 1] << 8;
            out[o++] = k_b64_table[(v >> 12) & 0x3F];
            out[o++] = k_b64_table[(v >> 6) & 0x3F];
            out[o++] = '=';
        } else {
            out[o++] = k_b64_table[(v >> 12) & 0x3F];
            out[o++] = '=';
            out[o++] = '=';
        }
    }

    return o;
}

size_t base64_decode(uint8_t* out, const char* in, size_t inlen) {
    size_t i = 0;
    size_t o = 0;
    if (!out || (!in && inlen != 0)) return 0;

    while (i < inlen) {
        // collect 4 chars, skipping whitespace / 详见英文原注释。
        unsigned char a = 0, b = 0, c = 0, d = 0;
        int pad = 0;
        int got = 0;
        while (i < inlen && got < 4) {
            unsigned char ch = (unsigned char)in[i++];
            if (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n')
                continue;
            if (ch == '=') {
                // padding / 详见英文原注释。
                if (got < 2) return o; // invalid
                pad++;
                // treat as 0 for value / treat 作为 0 用于 value
                if (got == 2) c = 0;
                if (got == 3) d = 0;
                got++;
                continue;
            }

            unsigned char v = b64_rev(ch);
            if (v == 0xFF) return o; // invalid char
            if (got == 0) a = v;
            else if (got == 1) b = v;
            else if (got == 2) c = v;
            else d = v;
            got++;
        }

        if (got == 0) break;
        if (got < 4) break;

        uint32_t v = ((uint32_t)a << 18) | ((uint32_t)b << 12) | ((uint32_t)c << 6) | (uint32_t)d;
        out[o++] = (uint8_t)((v >> 16) & 0xFF);
        if (pad < 2) out[o++] = (uint8_t)((v >> 8) & 0xFF);
        if (pad < 1) out[o++] = (uint8_t)(v & 0xFF);
        if (pad) break;
    }

    return o;
}


