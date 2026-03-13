// SPDX-License-Identifier: LGPL-3.0-or-later
#include "core/internal/log_filter.h"
#include "common/internal/string_utils.h"

#include <pthread.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace visiong {
namespace {

bool is_false_like(std::string value) {
    visiong::to_lower_inplace(value);
    return value == "0" || value == "false" || value == "off" || value == "no";
}

std::vector<std::string> default_patterns() {
    return {
        "CAMHW:E",
        "can not find sof time",
        "getAfInfoParams",
        "getIrisInfoParams",
    };
}

std::vector<std::string> split_patterns(const char* raw_patterns) {
    std::vector<std::string> patterns;
    if (raw_patterns == nullptr || *raw_patterns == '\0') {
        return patterns;
    }

    std::string token;
    std::string raw(raw_patterns);
    for (char ch : raw) {
        if (ch == ';') {
            if (!token.empty()) {
                patterns.push_back(token);
                token.clear();
            }
            continue;
        }
        token.push_back(ch);
    }
    if (!token.empty()) {
        patterns.push_back(token);
    }
    return patterns;
}

class CameraNoiseFilter {
public:
    CameraNoiseFilter() {
        if (!should_enable()) {
            return;
        }

        patterns_ = default_patterns();
        const char* custom_patterns = std::getenv("VISIONG_LOG_FILTER_PATTERNS");
        if (custom_patterns != nullptr) {
            const auto parsed = split_patterns(custom_patterns);
            if (!parsed.empty()) {
                patterns_ = parsed;
            }
        }

        if (patterns_.empty()) {
            return;
        }

        if (::pipe(stdout_pipe_fds_) != 0) {
            return;
        }
        if (::pipe(stderr_pipe_fds_) != 0) {
            close_fd(stdout_pipe_fds_[0]);
            close_fd(stdout_pipe_fds_[1]);
            return;
        }

        original_stdout_ = ::dup(STDOUT_FILENO);
        original_stderr_ = ::dup(STDERR_FILENO);
        if (original_stdout_ < 0 || original_stderr_ < 0) {
            shutdown();
            return;
        }

        ::fflush(stdout);
        ::fflush(stderr);
        if (::dup2(stdout_pipe_fds_[1], STDOUT_FILENO) < 0 ||
            ::dup2(stderr_pipe_fds_[1], STDERR_FILENO) < 0) {
            shutdown();
            return;
        }

        close_fd(stdout_pipe_fds_[1]);
        close_fd(stderr_pipe_fds_[1]);

        stdout_thread_args_.self = this;
        stdout_thread_args_.read_fd = stdout_pipe_fds_[0];
        stdout_thread_args_.forward_fd = original_stdout_;
        if (::pthread_create(&stdout_thread_id_, nullptr, &CameraNoiseFilter::thread_entry,
                             &stdout_thread_args_) != 0) {
            shutdown();
            return;
        }
        stdout_thread_started_ = true;

        stderr_thread_args_.self = this;
        stderr_thread_args_.read_fd = stderr_pipe_fds_[0];
        stderr_thread_args_.forward_fd = original_stderr_;
        if (::pthread_create(&stderr_thread_id_, nullptr, &CameraNoiseFilter::thread_entry,
                             &stderr_thread_args_) != 0) {
            shutdown();
            return;
        }
        stderr_thread_started_ = true;
    }

    ~CameraNoiseFilter() {
        shutdown();
    }

private:
    struct ThreadArgs {
        CameraNoiseFilter* self = nullptr;
        int read_fd = -1;
        int forward_fd = -1;
    };

    static void close_fd(int& fd) {
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
    }

    static void write_all(int fd, const char* data, size_t size) {
        size_t offset = 0;
        while (offset < size) {
            const ssize_t written = ::write(fd, data + offset, size - offset);
            if (written < 0) {
                if (errno == EINTR) {
                    continue;
                }
                break;
            }
            offset += static_cast<size_t>(written);
        }
    }

    void shutdown() {
        if (shutdown_done_) {
            return;
        }
        shutdown_done_ = true;

        if (original_stdout_ >= 0 || original_stderr_ >= 0) {
            ::fflush(stdout);
            ::fflush(stderr);
        }
        if (original_stdout_ >= 0) {
            ::dup2(original_stdout_, STDOUT_FILENO);
        }
        if (original_stderr_ >= 0) {
            ::dup2(original_stderr_, STDERR_FILENO);
        }

        close_fd(stdout_pipe_fds_[1]);
        close_fd(stderr_pipe_fds_[1]);

        if (stdout_thread_started_) {
            ::pthread_join(stdout_thread_id_, nullptr);
            stdout_thread_started_ = false;
        }
        if (stderr_thread_started_) {
            ::pthread_join(stderr_thread_id_, nullptr);
            stderr_thread_started_ = false;
        }

        close_fd(stdout_pipe_fds_[0]);
        close_fd(stderr_pipe_fds_[0]);
        close_fd(original_stdout_);
        close_fd(original_stderr_);
    }

    static void* thread_entry(void* opaque) {
        auto* args = static_cast<ThreadArgs*>(opaque);
        if (args != nullptr && args->self != nullptr) {
            args->self->run(args->read_fd, args->forward_fd);
        }
        return nullptr;
    }

    void run(int read_fd, int forward_fd) const {
        std::array<char, 1024> buffer{};
        while (true) {
            const ssize_t bytes = ::read(read_fd, buffer.data(), buffer.size() - 1);
            if (bytes <= 0) {
                if (bytes < 0 && errno == EINTR) {
                    continue;
                }
                break;
            }

            const std::string chunk(buffer.data(), static_cast<size_t>(bytes));
            if (should_drop(chunk)) {
                continue;
            }
            write_all(forward_fd, chunk.data(), chunk.size());
        }
    }

    bool should_drop(const std::string& chunk) const {
        return std::any_of(patterns_.begin(), patterns_.end(), [&](const std::string& pattern) {
            return !pattern.empty() &&
                   chunk.find(std::string_view(pattern.data(), pattern.size())) != std::string::npos;
        });
    }

    static bool should_enable() {
        const char* env = std::getenv("VISIONG_LOG_FILTER");
        if (env == nullptr) {
            return true;
        }
        return !is_false_like(env);
    }

    int stdout_pipe_fds_[2] = {-1, -1};
    int stderr_pipe_fds_[2] = {-1, -1};
    int original_stdout_ = -1;
    int original_stderr_ = -1;
    pthread_t stdout_thread_id_{};
    pthread_t stderr_thread_id_{};
    bool stdout_thread_started_ = false;
    bool stderr_thread_started_ = false;
    bool shutdown_done_ = false;
    ThreadArgs stdout_thread_args_{};
    ThreadArgs stderr_thread_args_{};
    std::vector<std::string> patterns_;
};

}  // namespace

void initialize_camera_log_filter() {
    static std::once_flag once;
    std::call_once(once, []() {
        static CameraNoiseFilter filter;
        (void)filter;
    });
}

}  // namespace visiong

