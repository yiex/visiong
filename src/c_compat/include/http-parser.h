// SPDX-License-Identifier: LGPL-3.0-or-later
// Minimal HTTP/RTSP message parser for media-server/librtsp. / 最小化 HTTP/RTSP message parser 用于 media-server/librtsp.
// This project only needs a small subset of ireader/sdk http-parser.
//
// Supported: / 详见英文原注释。
// - RTSP request parsing: METHOD URI RTSP/x.y + headers + optional body(Content-Length)
// - RTSP response parsing (basic): RTSP/x.y code reason + headers + optional body / - RTSP response parsing (基础): RTSP/x.y code reason + headers + optional body
// - Accessors used by librtsp server/client modules

#ifndef _VISIONG_HTTP_PARSER_H_
#define _VISIONG_HTTP_PARSER_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    HTTP_PARSER_REQUEST = 0,
    HTTP_PARSER_RESPONSE = 1,
};

typedef struct http_parser_t http_parser_t;

http_parser_t* http_parser_create(int type, void* reserved1, void* reserved2);
void http_parser_destroy(http_parser_t* parser);

/// Reset parser state. If internal buffer already contains a complete message, / Reset parser 状态. 如果 内部 缓冲区 已经 contains complete message,
/// it will be consumed and any remaining bytes kept for next parsing.
void http_parser_clear(http_parser_t* parser);

/// Feed input bytes into parser. / Feed 输入 bytes into parser.
/// @param[in] data input bytes (can be NULL if *bytes==0)
/// @param[in,out] bytes input length; output remaining(unconsumed from input). (visiong impl always consumes / @param[在,out] bytes 输入 length; 输出 remaining(unconsumed from 输入). (visiong impl always consumes
/// input)
/// @return 0-ok(message complete), 1-need more data, <0-error / @返回 0-ok(message complete), 1-need 更 data, <0-错误
int http_parser_input(http_parser_t* parser, const void* data, size_t* bytes);

// ---- accessors (subset) ---- / 详见英文原注释。

/// Get protocol version (e.g. RTSP/1.0). / 详见英文原注释。
/// @return 0-ok, <0-error
int http_get_version(http_parser_t* parser, char protocol[8], int* major, int* minor);

/// Get request method string (e.g. "DESCRIBE"). Valid for request parser. / Get request method string (e.g. "DESCRIBE"). 有效 用于 request parser.
const char* http_get_request_method(http_parser_t* parser);

/// Get request URI string. Valid for request parser. / Get request URI string. 有效 用于 request parser.
const char* http_get_request_uri(http_parser_t* parser);

/// Get response status code. Valid for response parser. / Get response status code. 有效 用于 response parser.
int http_get_status_code(http_parser_t* parser);

/// Find header value by name (case-insensitive). Return NULL if not found. / Find header value 由 name (case-insensitive). 返回 NULL 如果 不 found.
const char* http_get_header_by_name(http_parser_t* parser, const char* name);

/// Find header and parse int64 value. Return 0 on success, <0 on error/not found. / Find header 与 parse int64 value. 返回 0 在 success, <0 在 错误/不 found.
int http_get_header_by_name2(http_parser_t* parser, const char* name, int64_t* value);

/// Get content pointer and length (body). Content is valid until http_parser_clear/destroy. / Get content pointer 与 length (body). Content 为 有效 until http_parser_clear/destroy.
const void* http_get_content(http_parser_t* parser);
size_t http_get_content_length(http_parser_t* parser);

#ifdef __cplusplus
}
#endif

#endif // _VISIONG_HTTP_PARSER_H_

