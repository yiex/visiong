// SPDX-License-Identifier: LGPL-3.0-or-later
// Minimal HTTP reason-phrase helper for media-server/librtsp. / 最小化 HTTP reason-phrase 辅助函数 用于 media-server/librtsp.
// Original dependency: ireader/sdk http-reason.h

#ifndef _VISIONG_HTTP_REASON_H_
#define _VISIONG_HTTP_REASON_H_

#ifdef __cplusplus
extern "C" {
#endif

/// Map status code -> reason phrase (ASCII). / 映射 status code -> reason phrase (ASCII).
const char* http_reason_phrase(int code);

#ifdef __cplusplus
}
#endif

#endif // _VISIONG_HTTP_REASON_H_

