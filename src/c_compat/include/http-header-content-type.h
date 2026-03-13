// SPDX-License-Identifier: LGPL-3.0-or-later
// Minimal Content-Type header parser for media-server/librtsp. / 最小化 Content-Type header parser 用于 media-server/librtsp.

#ifndef _VISIONG_HTTP_HEADER_CONTENT_TYPE_H_
#define _VISIONG_HTTP_HEADER_CONTENT_TYPE_H_

#ifdef __cplusplus
extern "C" {
#endif

struct http_header_content_type_t {
    char media_type[32];    // e.g. "application"
    char media_subtype[64]; // e.g. "sdp"
};

/// Parse HTTP/RTSP Content-Type field value. / 详见英文原注释。
/// Example inputs:
/// - "application/sdp" / 详见英文原注释。
/// - "application/sdp; charset=utf-8"
/// @return 0-ok, <0-error / @返回 0-ok, <0-错误
int http_header_content_type(const char* field, struct http_header_content_type_t* ct);

#ifdef __cplusplus
}
#endif

#endif // _VISIONG_HTTP_HEADER_CONTENT_TYPE_H_

