// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_MODULES_FONT_SUPPORT_H
#define VISIONG_MODULES_FONT_SUPPORT_H

#include <cstddef>
#include <string>
#include <vector>

namespace visiong::font {

struct FontBlob {
    std::vector<char> owned_data;
    const char* data = nullptr;
    size_t size = 0;
    std::string source;
    bool using_embedded = false;
};

FontBlob load_font_blob(const std::string& font_path);
using Rune = unsigned int;

std::vector<Rune> build_rune_ranges(const FontBlob& font_blob,
                                    const std::string& pre_chars,
                                    bool include_full_cjk,
                                    size_t max_glyphs = 0);

}  // namespace visiong::font

#endif  // VISIONG_MODULES_FONT_SUPPORT_H

