// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_COMMON_STRING_UTILS_H_
#define VISIONG_COMMON_STRING_UTILS_H_

#include <algorithm>
#include <cctype>
#include <string>

namespace visiong {

inline void to_lower_inplace(std::string& value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
}

inline std::string to_lower_copy(std::string value) {
    to_lower_inplace(value);
    return value;
}

inline bool iequals(const std::string& lhs, const std::string& rhs) {
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(lhs[i])) !=
            std::tolower(static_cast<unsigned char>(rhs[i]))) {
            return false;
        }
    }
    return true;
}

}  // namespace visiong

#endif  // VISIONG_COMMON_STRING_UTILS_H_

