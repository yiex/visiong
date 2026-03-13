// SPDX-License-Identifier: LGPL-3.0-or-later
// Minimal Content-Type header parser for media-server/librtsp. / 最小化 Content-Type header parser 用于 media-server/librtsp.

#include "http-header-content-type.h"

#include <string.h>
#include <ctype.h>

static const char* skip_lws(const char* s) {
    while (*s == ' ' || *s == '\t') ++s;
    return s;
}

static void copy_token(char* out, size_t outcap, const char* begin, const char* end) {
    if (!out || outcap == 0) return;
    size_t n = (size_t)(end > begin ? (end - begin) : 0);
    if (n >= outcap) n = outcap - 1;
    if (n > 0) memcpy(out, begin, n);
    out[n] = '\0';
}

int http_header_content_type(const char* field, struct http_header_content_type_t* ct) {
    if (!field || !ct) return -1;
    ct->media_type[0] = '\0';
    ct->media_subtype[0] = '\0';

    const char* s = skip_lws(field);
    const char* semi = strchr(s, ';');
    const char* end = semi ? semi : (s + strlen(s));

    // trim trailing spaces / 详见英文原注释。
    while (end > s && (end[-1] == ' ' || end[-1] == '\t')) --end;

    const char* slash = memchr(s, '/', (size_t)(end - s));
    if (!slash) return -1;

    const char* type_end = slash;
    while (type_end > s && (type_end[-1] == ' ' || type_end[-1] == '\t')) --type_end;

    const char* sub_begin = slash + 1;
    sub_begin = skip_lws(sub_begin);
    const char* sub_end = end;
    while (sub_end > sub_begin && (sub_end[-1] == ' ' || sub_end[-1] == '\t')) --sub_end;

    if (type_end <= s || sub_end <= sub_begin) return -1;

    copy_token(ct->media_type, sizeof(ct->media_type), s, type_end);
    copy_token(ct->media_subtype, sizeof(ct->media_subtype), sub_begin, sub_end);
    return 0;
}


