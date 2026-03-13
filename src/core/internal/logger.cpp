// SPDX-License-Identifier: LGPL-3.0-or-later
#include "core/internal/logger.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_set>

namespace visiong::log {

namespace {

std::atomic<Level> g_level{Level::Info};
std::atomic<bool> g_force_flush{false};
std::unordered_set<std::string> g_module_allowlist;
bool g_module_filter_enabled = false;
std::mutex g_state_mutex;
std::mutex g_output_mutex;
std::once_flag g_init_once;

std::string trim_copy(std::string text) {
    const auto not_space = [](unsigned char c) { return !std::isspace(c); };
    text.erase(text.begin(), std::find_if(text.begin(), text.end(), not_space));
    text.erase(std::find_if(text.rbegin(), text.rend(), not_space).base(), text.end());
    return text;
}

std::string lower_copy(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return text;
}

std::string normalize_token(std::string text) {
    return lower_copy(trim_copy(std::move(text)));
}

bool env_truthy(const char* value) {
    if (!value || value[0] == '\0') {
        return false;
    }
    const std::string normalized = normalize_token(value);
    return normalized != "0" && normalized != "false" && normalized != "off" && normalized != "no";
}

void set_module_filter_impl(std::string_view csv_modules) {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    g_module_allowlist.clear();
    g_module_filter_enabled = false;

    std::string csv(csv_modules);
    size_t start = 0;
    while (start <= csv.size()) {
        const size_t comma = csv.find(',', start);
        const size_t end = (comma == std::string::npos) ? csv.size() : comma;
        std::string token = normalize_token(csv.substr(start, end - start));
        if (!token.empty()) {
            if (token == "*") {
                g_module_allowlist.clear();
                g_module_filter_enabled = false;
                return;
            }
            g_module_allowlist.insert(std::move(token));
        }
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    g_module_filter_enabled = !g_module_allowlist.empty();
}

void init_from_env_once() {
    const char* level_env = std::getenv("VISIONG_LOG_LEVEL");
    if (level_env && level_env[0] != '\0') {
        g_level.store(parse_level(level_env), std::memory_order_relaxed);
    }

    const char* modules_env = std::getenv("VISIONG_LOG_MODULES");
    if (modules_env && modules_env[0] != '\0') {
        set_module_filter_impl(modules_env);
    }

    g_force_flush.store(env_truthy(std::getenv("VISIONG_LOG_FORCE_FLUSH")), std::memory_order_relaxed);
}

void ensure_initialized() {
    std::call_once(g_init_once, init_from_env_once);
}

} // namespace

Level parse_level(std::string_view level_text) {
    std::string normalized = normalize_token(std::string(level_text));
    if (normalized == "debug") return Level::Debug;
    if (normalized == "warn" || normalized == "warning") return Level::Warn;
    if (normalized == "error" || normalized == "err") return Level::Error;
    return Level::Info;
}

const char* level_name(Level level) {
    switch (level) {
        case Level::Debug: return "DEBUG";
        case Level::Info: return "INFO";
        case Level::Warn: return "WARN";
        case Level::Error: return "ERROR";
        default: return "INFO";
    }
}

void set_level(Level level) {
    ensure_initialized();
    g_level.store(level, std::memory_order_relaxed);
}

Level get_level() {
    ensure_initialized();
    return g_level.load(std::memory_order_relaxed);
}

void set_module_filter(std::string_view csv_modules) {
    ensure_initialized();
    set_module_filter_impl(csv_modules);
}

bool should_log(Level level, std::string_view module) {
    ensure_initialized();
    if (static_cast<int>(level) < static_cast<int>(g_level.load(std::memory_order_relaxed))) {
        return false;
    }

    std::lock_guard<std::mutex> lock(g_state_mutex);
    if (!g_module_filter_enabled) return true;
    std::string normalized = normalize_token(std::string(module));
    return g_module_allowlist.find(normalized) != g_module_allowlist.end();
}

void log(Level level, std::string_view module, std::string_view message) {
    if (!should_log(level, module)) return;

    FILE* stream = (level == Level::Warn || level == Level::Error) ? stderr : stdout;
    std::lock_guard<std::mutex> lock(g_output_mutex);
    std::fprintf(stream, "[%s][%.*s] %.*s\n",
                 level_name(level),
                 static_cast<int>(module.size()), module.data(),
                 static_cast<int>(message.size()), message.data());
    if (level == Level::Warn || level == Level::Error || g_force_flush.load(std::memory_order_relaxed)) {
        std::fflush(stream);
    }
}

void logf(Level level, std::string_view module, const char* format, ...) {
    if (!format) return;
    char buffer[1024];
    va_list args;
    va_start(args, format);
    std::vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    log(level, module, buffer);
}

} // namespace visiong::log

