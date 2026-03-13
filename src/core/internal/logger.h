// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_CORE_LOGGER_H
#define VISIONG_CORE_LOGGER_H

#include <sstream>
#include <string>
#include <string_view>

namespace visiong::log {

enum class Level {
    Debug = 0,
    Info = 1,
    Warn = 2,
    Error = 3
};

Level parse_level(std::string_view level_text);
const char* level_name(Level level);

void set_level(Level level);
Level get_level();

void set_module_filter(std::string_view csv_modules);
bool should_log(Level level, std::string_view module);

void log(Level level, std::string_view module, std::string_view message);
void logf(Level level, std::string_view module, const char* format, ...);

} // namespace visiong::log

#define VISIONG_LOG_STREAM(level, module, message_expr)                                   \
    do {                                                                                   \
        std::ostringstream _visiong_log_oss;                                               \
        _visiong_log_oss << message_expr;                                                  \
        ::visiong::log::log(level, module, _visiong_log_oss.str());                       \
    } while (0)

#define VISIONG_LOG_DEBUG(module, message_expr)                                            \
    VISIONG_LOG_STREAM(::visiong::log::Level::Debug, module, message_expr)
#define VISIONG_LOG_INFO(module, message_expr)                                             \
    VISIONG_LOG_STREAM(::visiong::log::Level::Info, module, message_expr)
#define VISIONG_LOG_WARN(module, message_expr)                                             \
    VISIONG_LOG_STREAM(::visiong::log::Level::Warn, module, message_expr)
#define VISIONG_LOG_ERROR(module, message_expr)                                            \
    VISIONG_LOG_STREAM(::visiong::log::Level::Error, module, message_expr)

#define VISIONG_LOGF_DEBUG(module, format, ...)                                            \
    ::visiong::log::logf(::visiong::log::Level::Debug, module, format, ##__VA_ARGS__)
#define VISIONG_LOGF_INFO(module, format, ...)                                             \
    ::visiong::log::logf(::visiong::log::Level::Info, module, format, ##__VA_ARGS__)
#define VISIONG_LOGF_WARN(module, format, ...)                                             \
    ::visiong::log::logf(::visiong::log::Level::Warn, module, format, ##__VA_ARGS__)
#define VISIONG_LOGF_ERROR(module, format, ...)                                            \
    ::visiong::log::logf(::visiong::log::Level::Error, module, format, ##__VA_ARGS__)

#endif // VISIONG_CORE_LOGGER_H

