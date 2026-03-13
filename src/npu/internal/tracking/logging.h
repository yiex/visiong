// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_NPU_INTERNAL_TRACKING_LOGGING_H
#define VISIONG_NPU_INTERNAL_TRACKING_LOGGING_H

#include <cstdio>

// Log level from low to high: / Log 层 from 低 以 高:
// 0: disabled
// 1: error / 1: 错误
// 2: warning
// 3: info / 详见英文原注释。
// 4: debug
inline int g_log_level = 3;

#define NN_LOG_ERROR(...)            \
    do                               \
    {                                \
        if (g_log_level >= 1)        \
        {                            \
            std::printf("[NN_ERROR] "); \
            std::printf(__VA_ARGS__);   \
            std::printf("\n");         \
        }                            \
    } while (0)

#define NN_LOG_WARNING(...)            \
    do                                 \
    {                                  \
        if (g_log_level >= 2)          \
        {                              \
            std::printf("[NN_WARNING] "); \
            std::printf(__VA_ARGS__);     \
            std::printf("\n");           \
        }                              \
    } while (0)

#define NN_LOG_INFO(...)            \
    do                              \
    {                               \
        if (g_log_level >= 3)       \
        {                           \
            std::printf("[NN_INFO] "); \
            std::printf(__VA_ARGS__);  \
            std::printf("\n");        \
        }                           \
    } while (0)

#define NN_LOG_DEBUG(...)            \
    do                               \
    {                                \
        if (g_log_level >= 4)        \
        {                            \
            std::printf("[NN_DEBUG] "); \
            std::printf(__VA_ARGS__);   \
            std::printf("\n");         \
        }                            \
    } while (0)

#endif  // VISIONG_NPU_INTERNAL_TRACKING_LOGGING_H
