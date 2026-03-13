// SPDX-License-Identifier: LGPL-3.0-or-later
// Minimal RFC822/RFC1123 datetime formatter for media-server/librtsp. / 最小化 RFC822/RFC1123 datetime 格式ter 用于 media-server/librtsp.

#include "rfc822-datetime.h"

#include <string.h>

#if !defined(_WIN32)
#include <time.h>
#endif

const char* rfc822_datetime_format(time_t t, rfc822_datetime_t datetime) {
    if (!datetime) return NULL;

    struct tm tmv;
#if defined(_WIN32)
    gmtime_s(&tmv, &t);
#else
    gmtime_r(&t, &tmv);
#endif

    // RFC1123 date format used by RTSP/HTTP Date header / RFC1123 date 格式 used 由 RTSP/HTTP Date header
    // Example: "Sun, 06 Nov 1994 08:49:37 GMT"
    // Keep buffer small and stable (32 bytes is enough). / 保持 缓冲区 小 与 stable (32 bytes 为 enough).
    // Use strftime (locale-independent for %a/%b on most systems).
    size_t n = strftime(datetime, sizeof(rfc822_datetime_t), "%a, %d %b %Y %H:%M:%S GMT", &tmv);
    if (n == 0) {
        // Fallback / 回退
        strncpy(datetime, "Thu, 01 Jan 1970 00:00:00 GMT", sizeof(rfc822_datetime_t) - 1);
        datetime[sizeof(rfc822_datetime_t) - 1] = '\0';
    }
    return datetime;
}


