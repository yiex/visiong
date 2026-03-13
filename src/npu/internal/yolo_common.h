// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace visiong::npu::yolo {

inline int clamp_to_int(float value, int min_value, int max_value) {
    if (value < static_cast<float>(min_value)) {
        return min_value;
    }
    if (value > static_cast<float>(max_value)) {
        return max_value;
    }
    return static_cast<int>(value);
}

inline float calculate_iou_xywh(float x0, float y0, float w0, float h0, float x1, float y1, float w1, float h1) {
    const float xmin0 = x0;
    const float ymin0 = y0;
    const float xmax0 = x0 + w0;
    const float ymax0 = y0 + h0;
    const float xmin1 = x1;
    const float ymin1 = y1;
    const float xmax1 = x1 + w1;
    const float ymax1 = y1 + h1;

    const float overlap_w = std::max(0.0f, std::min(xmax0, xmax1) - std::max(xmin0, xmin1) + 1.0f);
    const float overlap_h = std::max(0.0f, std::min(ymax0, ymax1) - std::max(ymin0, ymin1) + 1.0f);
    const float intersection = overlap_w * overlap_h;

    const float area0 = (xmax0 - xmin0 + 1.0f) * (ymax0 - ymin0 + 1.0f);
    const float area1 = (xmax1 - xmin1 + 1.0f) * (ymax1 - ymin1 + 1.0f);
    const float union_area = area0 + area1 - intersection;
    return (union_area <= 0.0f) ? 0.0f : (intersection / union_area);
}

inline int32_t clip_to_int32(float value, float min_value, float max_value) {
    const float clipped = value < min_value ? min_value : (value > max_value ? max_value : value);
    return static_cast<int32_t>(clipped);
}

inline int8_t quantize_to_i8(float value, int32_t zero_point, float scale) {
    if (scale == 0.0f) {
        return 0;
    }
    const float dst = value / scale + static_cast<float>(zero_point);
    return static_cast<int8_t>(
        clip_to_int32(dst,
                      static_cast<float>(std::numeric_limits<int8_t>::min()),
                      static_cast<float>(std::numeric_limits<int8_t>::max())));
}

inline float dequantize_from_i8(int8_t value, int32_t zero_point, float scale) {
    return (static_cast<float>(value) - static_cast<float>(zero_point)) * scale;
}

inline void trim_line(char* line, size_t* start, size_t* len) {
    *start = 0;
    size_t n = std::strlen(line);
    while (n > 0 && (line[n - 1] == '\r' || line[n - 1] == '\n' || static_cast<unsigned char>(line[n - 1]) <= ' ')) {
        --n;
    }
    line[n] = '\0';
    while (*start < n && static_cast<unsigned char>(line[*start]) <= ' ') {
        ++(*start);
    }
    *len = (*start < n) ? (n - *start) : 0;
}

inline int load_non_empty_lines(const char* path, std::vector<std::string>* lines, bool* has_empty_line_after_data) {
    FILE* fp = std::fopen(path, "r");
    if (!fp) {
        return -1;
    }

    lines->clear();
    bool has_empty = false;
    char line[256];
    while (std::fgets(line, sizeof(line), fp)) {
        size_t start = 0;
        size_t len = 0;
        trim_line(line, &start, &len);
        if (len == 0) {
            if (!lines->empty()) {
                has_empty = true;
            }
            continue;
        }
        lines->emplace_back(line + start, len);
    }
    std::fclose(fp);

    if (has_empty_line_after_data != nullptr) {
        *has_empty_line_after_data = has_empty;
    }
    return 0;
}

inline char* dup_string(const std::string& src) {
    char* dst = static_cast<char*>(std::malloc(src.size() + 1));
    if (!dst) {
        return nullptr;
    }
    std::memcpy(dst, src.c_str(), src.size() + 1);
    return dst;
}

inline int assign_c_labels(const std::vector<std::string>& src, char*** labels_out) {
    *labels_out = static_cast<char**>(std::calloc(src.size(), sizeof(char*)));
    if (!*labels_out) {
        return -1;
    }

    for (size_t i = 0; i < src.size(); ++i) {
        (*labels_out)[i] = dup_string(src[i]);
        if (!(*labels_out)[i]) {
            for (size_t j = 0; j < i; ++j) {
                std::free((*labels_out)[j]);
            }
            std::free(*labels_out);
            *labels_out = nullptr;
            return -1;
        }
    }
    return 0;
}

inline void free_c_labels(char*** labels, int count) {
    if (!labels || !*labels) {
        return;
    }
    for (int i = 0; i < count; ++i) {
        std::free((*labels)[i]);
    }
    std::free(*labels);
    *labels = nullptr;
}

inline void sort_indices_desc(const std::vector<float>& scores, std::vector<int>* indices) {
    indices->resize(scores.size());
    std::iota(indices->begin(), indices->end(), 0);
    std::stable_sort(indices->begin(), indices->end(),
                     [&scores](int lhs, int rhs) { return scores[lhs] > scores[rhs]; });
}

} // namespace visiong::npu::yolo

