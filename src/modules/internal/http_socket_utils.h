// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_MODULES_HTTP_SOCKET_UTILS_H_
#define VISIONG_MODULES_HTTP_SOCKET_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include <errno.h>
#include <sys/socket.h>
#include <unistd.h>

namespace visiong {
namespace http {

inline bool send_all(int fd, const void* data, size_t bytes) {
    const uint8_t* cursor = static_cast<const uint8_t*>(data);
    size_t remaining = bytes;
    while (remaining > 0) {
#ifdef MSG_NOSIGNAL
        const ssize_t sent = ::send(fd, cursor, remaining, MSG_NOSIGNAL);
#else
        const ssize_t sent = ::send(fd, cursor, remaining, 0);
#endif
        if (sent < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (sent == 0) {
            return false;
        }
        cursor += static_cast<size_t>(sent);
        remaining -= static_cast<size_t>(sent);
    }
    return true;
}

inline bool read_http_request(int fd, std::string& out, size_t max_bytes = 16 * 1024) {
    out.clear();
    char buf[1024];
    while (out.find("\r\n\r\n") == std::string::npos && out.size() < max_bytes) {
        const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (n == 0) {
            return false;
        }
        out.append(buf, static_cast<size_t>(n));
    }
    return out.find("\r\n\r\n") != std::string::npos;
}

inline bool parse_http_request_path(const std::string& req, std::string& out_path) {
    const size_t line_end = req.find("\r\n");
    if (line_end == std::string::npos) {
        return false;
    }
    const std::string line = req.substr(0, line_end);
    const size_t sp1 = line.find(' ');
    if (sp1 == std::string::npos) {
        return false;
    }
    const size_t sp2 = line.find(' ', sp1 + 1);
    if (sp2 == std::string::npos) {
        return false;
    }
    const std::string method = line.substr(0, sp1);
    if (method != "GET") {
        return false;
    }
    out_path = line.substr(sp1 + 1, sp2 - sp1 - 1);
    return !out_path.empty();
}

}  // namespace http
}  // namespace visiong

#endif  // VISIONG_MODULES_HTTP_SOCKET_UTILS_H_

