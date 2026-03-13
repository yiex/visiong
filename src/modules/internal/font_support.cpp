// SPDX-License-Identifier: LGPL-3.0-or-later
#include "modules/internal/font_support.h"

#include "modules/internal/embedded_font.h"
#include "core/internal/logger.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>

namespace visiong::font {

namespace {

constexpr uint32_t make_tag(char a, char b, char c, char d) {
    return (static_cast<uint32_t>(static_cast<unsigned char>(a)) << 24) |
           (static_cast<uint32_t>(static_cast<unsigned char>(b)) << 16) |
           (static_cast<uint32_t>(static_cast<unsigned char>(c)) << 8) |
           static_cast<uint32_t>(static_cast<unsigned char>(d));
}

bool decode_utf8_next(const std::string& bytes, size_t* offset, Rune* out_cp) {
    if (offset == nullptr || out_cp == nullptr || *offset >= bytes.size()) {
        return false;
    }

    const auto* data = reinterpret_cast<const uint8_t*>(bytes.data());
    const size_t n = bytes.size();
    size_t i = *offset;

    const uint8_t c0 = data[i];
    Rune cp = 0;
    size_t need = 0;

    if (c0 < 0x80) {
        cp = static_cast<Rune>(c0);
        need = 1;
    } else if ((c0 & 0xE0) == 0xC0) {
        cp = static_cast<Rune>(c0 & 0x1F);
        need = 2;
    } else if ((c0 & 0xF0) == 0xE0) {
        cp = static_cast<Rune>(c0 & 0x0F);
        need = 3;
    } else if ((c0 & 0xF8) == 0xF0) {
        cp = static_cast<Rune>(c0 & 0x07);
        need = 4;
    } else {
        *offset = i + 1;
        return false;
    }

    if (i + need > n) {
        *offset = n;
        return false;
    }

    for (size_t k = 1; k < need; ++k) {
        const uint8_t cx = data[i + k];
        if ((cx & 0xC0) != 0x80) {
            *offset = i + 1;
            return false;
        }
        cp = static_cast<Rune>((cp << 6) | (cx & 0x3F));
    }

    if ((need == 2 && cp < 0x80) ||
        (need == 3 && cp < 0x800) ||
        (need == 4 && cp < 0x10000) ||
        cp > 0x10FFFF ||
        (cp >= 0xD800 && cp <= 0xDFFF)) {
        *offset = i + 1;
        return false;
    }

    *out_cp = cp;
    *offset = i + need;
    return true;
}

bool read_u16_be(const uint8_t* data, size_t size, size_t off, uint16_t* out) {
    if (out == nullptr || off + 2 > size) {
        return false;
    }
    *out = static_cast<uint16_t>((static_cast<uint16_t>(data[off]) << 8) |
                                 static_cast<uint16_t>(data[off + 1]));
    return true;
}

bool read_i16_be(const uint8_t* data, size_t size, size_t off, int16_t* out) {
    uint16_t v = 0;
    if (!read_u16_be(data, size, off, &v) || out == nullptr) {
        return false;
    }
    *out = static_cast<int16_t>(v);
    return true;
}

bool read_u32_be(const uint8_t* data, size_t size, size_t off, uint32_t* out) {
    if (out == nullptr || off + 4 > size) {
        return false;
    }
    *out = (static_cast<uint32_t>(data[off]) << 24) |
           (static_cast<uint32_t>(data[off + 1]) << 16) |
           (static_cast<uint32_t>(data[off + 2]) << 8) |
           static_cast<uint32_t>(data[off + 3]);
    return true;
}

void append_range_merged(std::vector<Rune>& ranges, Rune start, Rune end) {
    if (start > end) {
        return;
    }
    if (ranges.size() >= 2) {
        Rune& last_end = ranges[ranges.size() - 1];
        if (start <= last_end + 1) {
            last_end = std::max(last_end, end);
            return;
        }
    }
    ranges.push_back(start);
    ranges.push_back(end);
}

bool find_sfnt_offset(const uint8_t* data, size_t size, size_t* sfnt_off) {
    if (sfnt_off == nullptr || size < 12) {
        return false;
    }

    uint32_t tag = 0;
    if (!read_u32_be(data, size, 0, &tag)) {
        return false;
    }

    if (tag == make_tag('t', 't', 'c', 'f')) {
        uint32_t num_fonts = 0;
        if (!read_u32_be(data, size, 8, &num_fonts) || num_fonts == 0) {
            return false;
        }
        uint32_t first_off = 0;
        if (!read_u32_be(data, size, 12, &first_off)) {
            return false;
        }
        if (first_off + 12 > size) {
            return false;
        }
        *sfnt_off = static_cast<size_t>(first_off);
        return true;
    }

    *sfnt_off = 0;
    return true;
}

bool find_table(const uint8_t* data,
                size_t size,
                size_t sfnt_off,
                uint32_t table_tag,
                uint32_t* table_off,
                uint32_t* table_len) {
    if (sfnt_off + 12 > size || table_off == nullptr || table_len == nullptr) {
        return false;
    }

    uint16_t num_tables = 0;
    if (!read_u16_be(data, size, sfnt_off + 4, &num_tables)) {
        return false;
    }

    const size_t records_off = sfnt_off + 12;
    for (uint16_t i = 0; i < num_tables; ++i) {
        const size_t rec = records_off + static_cast<size_t>(i) * 16;
        if (rec + 16 > size) {
            return false;
        }
        uint32_t tag = 0;
        uint32_t off = 0;
        uint32_t len = 0;
        if (!read_u32_be(data, size, rec, &tag) ||
            !read_u32_be(data, size, rec + 8, &off) ||
            !read_u32_be(data, size, rec + 12, &len)) {
            return false;
        }
        if (tag == table_tag) {
            if (static_cast<uint64_t>(off) + static_cast<uint64_t>(len) > static_cast<uint64_t>(size)) {
                return false;
            }
            *table_off = off;
            *table_len = len;
            return true;
        }
    }
    return false;
}

bool parse_cmap_format12(const uint8_t* data,
                         size_t size,
                         size_t sub_off,
                         std::vector<Rune>* out_ranges) {
    if (out_ranges == nullptr || sub_off + 16 > size) {
        return false;
    }

    uint16_t fmt = 0;
    uint16_t reserved = 0;
    uint32_t length = 0;
    uint32_t n_groups = 0;
    if (!read_u16_be(data, size, sub_off, &fmt) ||
        !read_u16_be(data, size, sub_off + 2, &reserved) ||
        !read_u32_be(data, size, sub_off + 4, &length) ||
        !read_u32_be(data, size, sub_off + 12, &n_groups)) {
        return false;
    }

    if (fmt != 12 || reserved != 0 || length < 16 || sub_off + length > size) {
        return false;
    }

    size_t g_off = sub_off + 16;
    for (uint32_t i = 0; i < n_groups; ++i) {
        if (g_off + 12 > sub_off + length) {
            return false;
        }
        uint32_t start_cp = 0;
        uint32_t end_cp = 0;
        uint32_t start_gid = 0;
        if (!read_u32_be(data, size, g_off, &start_cp) ||
            !read_u32_be(data, size, g_off + 4, &end_cp) ||
            !read_u32_be(data, size, g_off + 8, &start_gid)) {
            return false;
        }

        if (start_cp <= end_cp) {
            // If the run maps to gid 0 it usually means missing glyphs. / 如果 run 映射 以 gid 0 it usually means missing glyphs.
            if (start_gid != 0) {
                append_range_merged(*out_ranges, static_cast<Rune>(start_cp), static_cast<Rune>(end_cp));
            }
        }
        g_off += 12;
    }

    return !out_ranges->empty();
}

bool parse_cmap_format4(const uint8_t* data,
                        size_t size,
                        size_t sub_off,
                        std::vector<Rune>* out_ranges) {
    if (out_ranges == nullptr || sub_off + 16 > size) {
        return false;
    }

    uint16_t fmt = 0;
    uint16_t length = 0;
    uint16_t seg_count_x2 = 0;
    if (!read_u16_be(data, size, sub_off, &fmt) ||
        !read_u16_be(data, size, sub_off + 2, &length) ||
        !read_u16_be(data, size, sub_off + 6, &seg_count_x2)) {
        return false;
    }

    if (fmt != 4 || length < 24 || (seg_count_x2 % 2) != 0 || sub_off + length > size) {
        return false;
    }

    const uint16_t seg_count = static_cast<uint16_t>(seg_count_x2 / 2);
    const size_t end_code_off = sub_off + 14;
    const size_t start_code_off = end_code_off + static_cast<size_t>(seg_count) * 2 + 2;
    const size_t id_delta_off = start_code_off + static_cast<size_t>(seg_count) * 2;
    const size_t id_range_off = id_delta_off + static_cast<size_t>(seg_count) * 2;

    if (id_range_off + static_cast<size_t>(seg_count) * 2 > sub_off + length) {
        return false;
    }

    for (uint16_t i = 0; i < seg_count; ++i) {
        uint16_t end_code = 0;
        uint16_t start_code = 0;
        int16_t id_delta = 0;
        uint16_t id_range = 0;
        if (!read_u16_be(data, size, end_code_off + static_cast<size_t>(i) * 2, &end_code) ||
            !read_u16_be(data, size, start_code_off + static_cast<size_t>(i) * 2, &start_code) ||
            !read_i16_be(data, size, id_delta_off + static_cast<size_t>(i) * 2, &id_delta) ||
            !read_u16_be(data, size, id_range_off + static_cast<size_t>(i) * 2, &id_range)) {
            return false;
        }

        if (start_code > end_code || end_code == 0xFFFF) {
            continue;
        }

        bool in_run = false;
        Rune run_start = 0;

        for (uint32_t cp = start_code; cp <= end_code; ++cp) {
            uint16_t glyph = 0;
            if (id_range == 0) {
                glyph = static_cast<uint16_t>((cp + static_cast<uint16_t>(id_delta)) & 0xFFFF);
            } else {
                const size_t id_range_entry_off = id_range_off + static_cast<size_t>(i) * 2;
                const size_t glyph_off = id_range_entry_off + id_range + static_cast<size_t>(cp - start_code) * 2;
                uint16_t glyph_index = 0;
                if (glyph_off + 2 <= sub_off + length && read_u16_be(data, size, glyph_off, &glyph_index) && glyph_index != 0) {
                    glyph = static_cast<uint16_t>((glyph_index + static_cast<uint16_t>(id_delta)) & 0xFFFF);
                }
            }

            const bool present = (glyph != 0);
            if (present && !in_run) {
                in_run = true;
                run_start = static_cast<Rune>(cp);
            }
            if (in_run && (!present || cp == end_code)) {
                const Rune run_end = static_cast<Rune>((present && cp == end_code) ? cp : (cp - 1));
                append_range_merged(*out_ranges, run_start, run_end);
                in_run = false;
            }
        }
    }

    return !out_ranges->empty();
}

bool build_font_present_ranges(const FontBlob& font_blob, std::vector<Rune>* out_ranges) {
    if (out_ranges == nullptr || font_blob.data == nullptr || font_blob.size == 0) {
        return false;
    }

    const auto* data = reinterpret_cast<const uint8_t*>(font_blob.data);
    const size_t size = font_blob.size;

    size_t sfnt_off = 0;
    if (!find_sfnt_offset(data, size, &sfnt_off)) {
        return false;
    }

    uint32_t cmap_off = 0;
    uint32_t cmap_len = 0;
    if (!find_table(data, size, sfnt_off, make_tag('c', 'm', 'a', 'p'), &cmap_off, &cmap_len)) {
        return false;
    }

    if (cmap_off + 4 > size) {
        return false;
    }

    uint16_t num_tables = 0;
    if (!read_u16_be(data, size, cmap_off + 2, &num_tables)) {
        return false;
    }

    size_t best12_off = 0;
    int best12_prio = -1;
    size_t best4_off = 0;
    int best4_prio = -1;

    const size_t rec_base = cmap_off + 4;
    for (uint16_t i = 0; i < num_tables; ++i) {
        const size_t rec = rec_base + static_cast<size_t>(i) * 8;
        if (rec + 8 > cmap_off + cmap_len || rec + 8 > size) {
            break;
        }

        uint16_t pid = 0;
        uint16_t eid = 0;
        uint32_t sub_rel = 0;
        if (!read_u16_be(data, size, rec, &pid) ||
            !read_u16_be(data, size, rec + 2, &eid) ||
            !read_u32_be(data, size, rec + 4, &sub_rel)) {
            continue;
        }

        const size_t sub_off = static_cast<size_t>(cmap_off) + static_cast<size_t>(sub_rel);
        uint16_t fmt = 0;
        if (!read_u16_be(data, size, sub_off, &fmt)) {
            continue;
        }

        if (fmt == 12) {
            int prio = 0;
            if (pid == 3 && eid == 10) prio = 3;
            else if (pid == 0) prio = 2;
            else prio = 1;
            if (prio > best12_prio) {
                best12_prio = prio;
                best12_off = sub_off;
            }
        } else if (fmt == 4) {
            int prio = 0;
            if (pid == 3 && (eid == 1 || eid == 0)) prio = 3;
            else if (pid == 0) prio = 2;
            else prio = 1;
            if (prio > best4_prio) {
                best4_prio = prio;
                best4_off = sub_off;
            }
        }
    }

    std::vector<Rune> parsed;
    parsed.reserve(256);

    if (best12_prio >= 0 && parse_cmap_format12(data, size, best12_off, &parsed)) {
        *out_ranges = std::move(parsed);
        return true;
    }

    parsed.clear();
    if (best4_prio >= 0 && parse_cmap_format4(data, size, best4_off, &parsed)) {
        *out_ranges = std::move(parsed);
        return true;
    }

    return false;
}

}  // namespace

FontBlob load_font_blob(const std::string& font_path) {
    FontBlob out;

    const std::string effective_font_path = font_path.empty() ? "assets/font.ttf" : font_path;
    if (std::filesystem::exists(effective_font_path)) {
        std::ifstream font_file(effective_font_path, std::ios::binary | std::ios::ate);
        if (font_file.is_open()) {
            const std::streamsize size = font_file.tellg();
            font_file.seekg(0, std::ios::beg);
            if (size > 0) {
                out.owned_data.resize(static_cast<size_t>(size));
                if (font_file.read(out.owned_data.data(), size)) {
                    out.data = out.owned_data.data();
                    out.size = out.owned_data.size();
                    out.source = effective_font_path;
                }
            }
        }
    }

    if (out.data == nullptr || out.size == 0) {
        out.data = reinterpret_cast<const char*>(DroidSans_ttf);
        out.size = DroidSans_ttf_len;
        out.source = "Embedded DroidSans";
        out.using_embedded = true;
    }

    return out;
}

std::vector<Rune> build_rune_ranges(const FontBlob& font_blob,
                                    const std::string& pre_chars,
                                    bool include_full_cjk,
                                    size_t max_glyphs) {
    std::vector<Rune> ranges;
    ranges.reserve(256);

    append_range_merged(ranges, 0x0020, 0x00FF);

    auto append_with_budget = [&](Rune start, Rune end, size_t* added_glyphs) -> bool {
        if (added_glyphs == nullptr || start > end) {
            return false;
        }

        if (max_glyphs > 0 && *added_glyphs >= max_glyphs) {
            return true;
        }

        const uint64_t run_len = static_cast<uint64_t>(end) - static_cast<uint64_t>(start) + 1ULL;
        if (max_glyphs > 0) {
            const size_t remain = max_glyphs - *added_glyphs;
            if (remain == 0) {
                return true;
            }
            if (run_len > static_cast<uint64_t>(remain)) {
                const Rune clipped_end = static_cast<Rune>(static_cast<uint64_t>(start) + static_cast<uint64_t>(remain) - 1ULL);
                append_range_merged(ranges, start, clipped_end);
                *added_glyphs += remain;
                return true;
            }
        }

        append_range_merged(ranges, start, end);
        *added_glyphs += static_cast<size_t>(run_len);
        return (max_glyphs > 0 && *added_glyphs >= max_glyphs);
    };

    if (!pre_chars.empty()) {
        std::vector<Rune> cps;
        cps.reserve(pre_chars.size());

        bool had_invalid_utf8 = false;
        size_t off = 0;
        while (off < pre_chars.size()) {
            Rune cp = 0;
            if (!decode_utf8_next(pre_chars, &off, &cp)) {
                had_invalid_utf8 = true;
                continue;
            }
            cps.push_back(cp);
        }

        if (had_invalid_utf8) {
            VISIONG_LOG_WARN("Font", "Invalid UTF-8 sequence in predefine_chars.");
        }

        std::sort(cps.begin(), cps.end());
        cps.erase(std::unique(cps.begin(), cps.end()), cps.end());

        ranges.reserve(ranges.size() + cps.size() * 2 + 1);
        for (Rune cp : cps) {
            append_range_merged(ranges, cp, cp);
        }
    } else if (include_full_cjk) {
        std::vector<Rune> font_ranges;
        font_ranges.reserve(256);

        size_t added_glyphs = 0;
        if (build_font_present_ranges(font_blob, &font_ranges)) {
            for (size_t i = 0; i + 1 < font_ranges.size(); i += 2) {
                if (append_with_budget(font_ranges[i], font_ranges[i + 1], &added_glyphs)) {
                    break;
                }
            }
            VISIONG_LOG_INFO("Font",
                             "No predefine_chars: using glyph ranges from font, segments="
                                 << (font_ranges.size() / 2)
                                 << ", baked_glyphs=" << added_glyphs
                                 << ((max_glyphs > 0) ? ", glyph_budget=" + std::to_string(max_glyphs) : std::string("")));
            if (max_glyphs > 0 && added_glyphs >= max_glyphs) {
                VISIONG_LOG_WARN("Font",
                                 "Glyph ranges were capped by glyph_budget. "
                                 "Use predefine_chars for exact Chinese coverage.");
            }
        } else {
            static const Rune full_cjk_ranges[] = {
                0x3000, 0x30FF,
                0xFF00, 0xFFEF,
                0x4E00, 0x9FFF,
            };
            for (size_t i = 0; i + 1 < std::size(full_cjk_ranges); i += 2) {
                if (append_with_budget(full_cjk_ranges[i], full_cjk_ranges[i + 1], &added_glyphs)) {
                    break;
                }
            }
            VISIONG_LOG_WARN("Font",
                             "Failed to parse font cmap, fallback to CJK ranges. baked_glyphs="
                                 << added_glyphs
                                 << ((max_glyphs > 0) ? ", glyph_budget=" + std::to_string(max_glyphs) : std::string("")));
        }
    }

    ranges.push_back(0);
    return ranges;
}

}  // namespace visiong::font

