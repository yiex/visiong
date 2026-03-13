// SPDX-License-Identifier: LGPL-3.0-or-later
// Minimal HTTP/RTSP message parser for media-server/librtsp. / 最小化 HTTP/RTSP message parser 用于 media-server/librtsp.
// Implements a tiny subset of https://github.com/ireader/sdk http-parser.

#include "http-parser.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <ctype.h>

#if !defined(_WIN32)
#include <strings.h> // strcasecmp
#endif

#define HTTP_PARSER_MAX_HEADERS 64
#define HTTP_PARSER_INIT_CAP    (64 * 1024)

struct http_header_kv_t {
    const char* name;
    const char* value;
};

struct http_parser_t {
    int type; // HTTP_PARSER_REQUEST/RESPONSE

    char* buf;
    size_t len;
    size_t cap;

    // message framing / 详见英文原注释。
    size_t header_end_pos; // bytes including \r\n\r\n
    size_t content_len;
    size_t message_len; // header_end_pos + content_len
    int complete;

    // parsed fields (pointers into buf) / 详见英文原注释。
    char protocol[8];
    int major;
    int minor;

    char* method;
    char* uri;
    int status_code;

    struct http_header_kv_t headers[HTTP_PARSER_MAX_HEADERS];
    size_t header_count;

    const void* content;
};

static int http_ascii_tolower(int c) {
    if (c >= 'A' && c <= 'Z') return c - 'A' + 'a';
    return c;
}

static int http_match_token_ci(const char* p, const char* token) {
    // token must be NUL-terminated / token 必须 被 NUL-terminated
    while (*token) {
        if (!*p) return 0;
        if (http_ascii_tolower((unsigned char)*p) != http_ascii_tolower((unsigned char)*token))
            return 0;
        ++p;
        ++token;
    }
    return 1;
}

static size_t http_find_header_end(const char* p, size_t n) {
    // find "\r\n\r\n" / 详见英文原注释。
    if (n < 4) return (size_t)-1;
    for (size_t i = 0; i + 3 < n; ++i) {
        if (p[i] == '\r' && p[i + 1] == '\n' && p[i + 2] == '\r' && p[i + 3] == '\n')
            return i;
    }
    return (size_t)-1;
}

static size_t http_parse_content_length_raw(const char* p, size_t header_bytes) {
    // Scan headers for "Content-Length:" (case-insensitive), without modifying buffer. / Scan headers 用于 "Content-Length:" (case-insensitive), without modifying 缓冲区.
    // Return 0 if not found/invalid.
    const char* end = p + header_bytes;
    for (const char* line = p; line < end; ) {
        const char* line_end = (const char*)memchr(line, '\n', (size_t)(end - line));
        if (!line_end) line_end = end;

        // line start points to beginning of a header line (or request line) / line 启动 points 以 beginning 的 header line (或 request line)
        // Only check header lines with "Content-Length:" prefix.
        // Skip request/status line by requiring ':' to exist before line_end. / 跳过 request/status line 由 requiring ':' 以 exist before line_end.
        const char* colon = (const char*)memchr(line, ':', (size_t)(line_end - line));
        if (colon) {
            // trim leading spaces / 详见英文原注释。
            const char* name = line;
            while (name < colon && (*name == ' ' || *name == '\t'))
                ++name;

            if (http_match_token_ci(name, "Content-Length")) {
                const char* v = colon + 1;
                while (v < line_end && (*v == ' ' || *v == '\t'))
                    ++v;

                size_t value = 0;
                int any = 0;
                while (v < line_end && *v >= '0' && *v <= '9') {
                    any = 1;
                    value = value * 10 + (size_t)(*v - '0');
                    ++v;
                }
                return any ? value : 0;
            }
        }

        line = line_end + (line_end < end ? 1 : 0);
    }
    return 0;
}

static int http_parse_version(const char* ver, char protocol[8], int* major, int* minor) {
    const char* slash = strchr(ver, '/');
    if (!slash) return -EINVAL;
    size_t proto_len = (size_t)(slash - ver);
    if (proto_len == 0) return -EINVAL;
    if (proto_len >= 8) proto_len = 7;
    memcpy(protocol, ver, proto_len);
    protocol[proto_len] = '\0';

    const char* v = slash + 1;
    char* dot = strchr((char*)v, '.');
    if (!dot) return -EINVAL;
    *major = atoi(v);
    *minor = atoi(dot + 1);
    return 0;
}

static void http_rtrim_inplace(char* s) {
    size_t n = strlen(s);
    while (n > 0) {
        char c = s[n - 1];
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n')
            s[--n] = '\0';
        else
            break;
    }
}

static char* http_ltrim(char* s) {
    while (*s == ' ' || *s == '\t') ++s;
    return s;
}

static int http_parse_message(http_parser_t* parser) {
    if (!parser || !parser->buf || parser->header_end_pos < 4) return -EINVAL;

    // Reset parsed fields / 详见英文原注释。
    parser->method = NULL;
    parser->uri = NULL;
    parser->status_code = 0;
    parser->protocol[0] = '\0';
    parser->major = 0;
    parser->minor = 0;
    parser->header_count = 0;

    char* buf = parser->buf;
    char* header_end = buf + parser->header_end_pos;

    // Parse start-line (first CRLF) / Parse start-line (首个 CRLF)
    char* cr = (char*)memchr(buf, '\r', (size_t)(header_end - buf));
    if (!cr || cr + 1 >= header_end || cr[1] != '\n')
        return -EINVAL;
    *cr = '\0'; // terminate start line
    char* start = buf;

    if (parser->type == HTTP_PARSER_REQUEST) {
        char* sp1 = strchr(start, ' ');
        if (!sp1) return -EINVAL;
        *sp1 = '\0';
        parser->method = start;

        char* p2 = sp1 + 1;
        while (*p2 == ' ') ++p2;
        char* sp2 = strchr(p2, ' ');
        if (!sp2) return -EINVAL;
        *sp2 = '\0';
        parser->uri = p2;

        char* ver = sp2 + 1;
        while (*ver == ' ') ++ver;
        if (0 != http_parse_version(ver, parser->protocol, &parser->major, &parser->minor))
            return -EINVAL;
    } else {
        // Response: VERSION SP CODE SP REASON / 详见英文原注释。
        char* sp1 = strchr(start, ' ');
        if (!sp1) return -EINVAL;
        *sp1 = '\0';
        if (0 != http_parse_version(start, parser->protocol, &parser->major, &parser->minor))
            return -EINVAL;

        char* p2 = sp1 + 1;
        while (*p2 == ' ') ++p2;
        parser->status_code = atoi(p2);
    }

    // Parse headers line by line (NUL-terminate each line at '\r') / Parse headers line 由 line (NUL-terminate each line at '\r')
    char* line = cr + 2; // after CRLF
    while (line < header_end) {
        char* e = (char*)memchr(line, '\r', (size_t)(header_end - line));
        if (!e || e + 1 >= header_end || e[1] != '\n')
            break;
        *e = '\0'; // terminate this line

        if (line == e) {
            // empty line: end of headers / empty line: end 的 headers
            break;
        }

        char* colon = strchr(line, ':');
        if (colon) {
            *colon = '\0';
            http_rtrim_inplace(line);
            char* value = http_ltrim(colon + 1);
            http_rtrim_inplace(value);

            if (parser->header_count < HTTP_PARSER_MAX_HEADERS) {
                parser->headers[parser->header_count].name = line;
                parser->headers[parser->header_count].value = value;
                parser->header_count++;
            }
        }

        line = e + 2;
    }

    parser->content = (const void*)(buf + parser->header_end_pos);
    return 0;
}

static int http_ensure_capacity(http_parser_t* parser, size_t need) {
    if (parser->cap >= need) return 0;
    size_t cap = parser->cap ? parser->cap : HTTP_PARSER_INIT_CAP;
    while (cap < need) {
        cap *= 2;
        if (cap < parser->cap) return -ENOMEM; // overflow
    }
    char* p = (char*)realloc(parser->buf, cap);
    if (!p) return -ENOMEM;
    parser->buf = p;
    parser->cap = cap;
    return 0;
}

http_parser_t* http_parser_create(int type, void* reserved1, void* reserved2) {
    (void)reserved1;
    (void)reserved2;

    http_parser_t* p = (http_parser_t*)calloc(1, sizeof(http_parser_t));
    if (!p) return NULL;
    p->type = type;
    p->buf = NULL;
    p->len = 0;
    p->cap = 0;
    p->header_end_pos = 0;
    p->content_len = 0;
    p->message_len = 0;
    p->complete = 0;
    if (0 != http_ensure_capacity(p, HTTP_PARSER_INIT_CAP)) {
        // Initial buffer allocation failed — clean up and return NULL / Initial 缓冲区 allocation 失败 — clean up 与 返回 NULL
        free(p);
        return NULL;
    }
    return p;
}

void http_parser_destroy(http_parser_t* parser) {
    if (!parser) return;
    if (parser->buf) free(parser->buf);
    free(parser);
}

void http_parser_clear(http_parser_t* parser) {
    if (!parser) return;

    if (parser->complete && parser->message_len > 0 && parser->len >= parser->message_len) {
        size_t remain = parser->len - parser->message_len;
        if (remain > 0) {
            memmove(parser->buf, parser->buf + parser->message_len, remain);
        }
        parser->len = remain;
    } else {
        // discard buffered data on clear / discard 缓冲区ed data 在 clear
        parser->len = 0;
    }

    parser->header_end_pos = 0;
    parser->content_len = 0;
    parser->message_len = 0;
    parser->complete = 0;
    parser->method = NULL;
    parser->uri = NULL;
    parser->status_code = 0;
    parser->protocol[0] = '\0';
    parser->major = 0;
    parser->minor = 0;
    parser->header_count = 0;
    parser->content = NULL;
}

int http_parser_input(http_parser_t* parser, const void* data, size_t* bytes) {
    if (!parser || !bytes) return -EINVAL;

    size_t n = *bytes;
    *bytes = 0; // visiong impl always consumes input into internal buffer

    if (n > 0 && data) {
        int r = http_ensure_capacity(parser, parser->len + n + 1);
        if (r != 0) return r;
        memcpy(parser->buf + parser->len, data, n);
        parser->len += n;
        parser->buf[parser->len] = '\0'; // convenience NUL
    }

    if (parser->complete) return 0;

    // If we already know message length, just wait for enough bytes. / 如果 we 已经 know message length, just 等待 用于 enough bytes.
    if (parser->message_len > 0) {
        if (parser->len < parser->message_len) return 1;
        // parse now (safe: no more appends before clear in librtsp) / parse now (安全: 不 更 appends before clear 在 librtsp)
        if (0 != http_parse_message(parser)) return -EINVAL;
        parser->content = (const void*)(parser->buf + parser->header_end_pos);
        parser->complete = 1;
        return 0;
    }

    // Find header end / 详见英文原注释。
    size_t header_end = http_find_header_end(parser->buf, parser->len);
    if ((size_t)-1 == header_end) return 1; // need more data

    parser->header_end_pos = header_end + 4;
    parser->content_len = http_parse_content_length_raw(parser->buf, parser->header_end_pos);
    parser->message_len = parser->header_end_pos + parser->content_len;

    if (parser->len < parser->message_len) return 1; // need more body data

    if (0 != http_parse_message(parser)) return -EINVAL;
    parser->content = (const void*)(parser->buf + parser->header_end_pos);
    parser->complete = 1;
    return 0;
}

int http_get_version(http_parser_t* parser, char protocol[8], int* major, int* minor) {
    if (!parser || !protocol || !major || !minor) return -EINVAL;
    snprintf(protocol, 8, "%s", parser->protocol[0] ? parser->protocol : "");
    *major = parser->major;
    *minor = parser->minor;
    return 0;
}

const char* http_get_request_method(http_parser_t* parser) {
    return (parser && parser->method) ? parser->method : NULL;
}

const char* http_get_request_uri(http_parser_t* parser) {
    return (parser && parser->uri) ? parser->uri : NULL;
}

int http_get_status_code(http_parser_t* parser) {
    return parser ? parser->status_code : 0;
}

const char* http_get_header_by_name(http_parser_t* parser, const char* name) {
    if (!parser || !name) return NULL;
    for (size_t i = 0; i < parser->header_count; ++i) {
        const char* n = parser->headers[i].name;
        if (!n) continue;
#if defined(_WIN32)
        if (_stricmp(n, name) == 0)
            return parser->headers[i].value;
#else
        if (strcasecmp(n, name) == 0)
            return parser->headers[i].value;
#endif
    }
    return NULL;
}

int http_get_header_by_name2(http_parser_t* parser, const char* name, int64_t* value) {
    if (!value) return -EINVAL;
    const char* v = http_get_header_by_name(parser, name);
    if (!v) return -ENOENT;
    char* end = NULL;
    long long n = strtoll(v, &end, 10);
    if (end == v) return -EINVAL;
    *value = (int64_t)n;
    return 0;
}

const void* http_get_content(http_parser_t* parser) {
    return parser ? parser->content : NULL;
}

size_t http_get_content_length(http_parser_t* parser) {
    return parser ? parser->content_len : 0;
}


