// SPDX-License-Identifier: LGPL-3.0-or-later
// Minimal RFC822/RFC1123 datetime formatter for media-server/librtsp. / 最小化 RFC822/RFC1123 datetime 格式ter 用于 media-server/librtsp.

#ifndef _VISIONG_RFC822_DATETIME_H_
#define _VISIONG_RFC822_DATETIME_H_

#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef char rfc822_datetime_t[32];

/// Format UTC time to RFC822 date string, e.g. "Sun, 06 Nov 1994 08:49:37 GMT". / 格式 UTC time 以 RFC822 date string, e.g. "Sun, 06 Nov 1994 08:49:37 GMT".
/// @return datetime (same pointer as input)
const char* rfc822_datetime_format(time_t t, rfc822_datetime_t datetime);

#ifdef __cplusplus
}
#endif

#endif // _VISIONG_RFC822_DATETIME_H_

