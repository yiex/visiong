// SPDX-License-Identifier: LGPL-3.0-or-later
// Minimal HTTP reason-phrase helper for media-server/librtsp. / 最小化 HTTP reason-phrase 辅助函数 用于 media-server/librtsp.

#include "http-reason.h"

const char* http_reason_phrase(int code) {
    switch (code) {
        // 2xx / 详见英文原注释。
        case 200: return "OK";
        case 201: return "Created";
        case 204: return "No Content";

        // 3xx / 详见英文原注释。
        case 301: return "Moved Permanently";
        case 302: return "Found";
        case 304: return "Not Modified";

        // 4xx / 详见英文原注释。
        case 400: return "Bad Request";
        case 401: return "Unauthorized";
        case 403: return "Forbidden";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 408: return "Request Timeout";
        case 409: return "Conflict";
        case 410: return "Gone";
        case 413: return "Payload Too Large";
        case 414: return "URI Too Long";
        case 415: return "Unsupported Media Type";
        case 451: return "Unavailable For Legal Reasons";
        case 454: return "Session Not Found";
        case 461: return "Unsupported Transport";

        // 5xx / 详见英文原注释。
        case 500: return "Internal Server Error";
        case 501: return "Not Implemented";
        case 503: return "Service Unavailable";
        case 505: return "Version Not Supported";
        case 513: return "Message Too Large";

        default:
            return "Error";
    }
}


