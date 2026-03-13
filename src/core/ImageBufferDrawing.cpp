// SPDX-License-Identifier: LGPL-3.0-or-later
#include "visiong/core/ImageBuffer.h"
#include "visiong/core/BufferStateMachine.h"
#include "visiong/common/pixel_format.h"
#include "core/internal/logger.h"
#include "modules/internal/font_support.h"
#include "modules/internal/gui_render_utils.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#define NK_INCLUDE_FIXED_TYPES
#define NK_INCLUDE_STANDARD_IO
#define NK_INCLUDE_DEFAULT_ALLOCATOR
#define NK_INCLUDE_FONT_BAKING
#define NK_INCLUDE_DEFAULT_FONT
#include "nuklear.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
struct YUVColor { uint8_t y, u, v; };

static inline YUVColor rgb_to_yuv_bt601(const std::tuple<uint8_t, uint8_t, uint8_t>& rgb) {
    float r = std::get<0>(rgb);
    float g = std::get<1>(rgb);
    float b = std::get<2>(rgb);
    return {
        (uint8_t)(0.299f * r + 0.587f * g + 0.114f * b),
        (uint8_t)(-0.169f * r - 0.331f * g + 0.500f * b + 128),
        (uint8_t)(0.500f * r - 0.419f * g - 0.081f * b + 128)
    };
}

static inline void neon_fill_y_line(uint8_t* ptr, int len, uint8_t y_val) {
    if (len <= 0) return;
    int i = 0;
#if defined(__ARM_NEON)
    if (len >= 16) {
        uint8x16_t v_y = vdupq_n_u8(y_val);
        for (; i <= len - 16; i += 16) {
            vst1q_u8(ptr + i, v_y);
        }
    }
#endif
    for (; i < len; i++) ptr[i] = y_val;
}

static inline void neon_fill_uv_line(uint8_t* ptr, int len_pixels, uint8_t u_val, uint8_t v_val) {
    if (len_pixels <= 0) return;
    int pairs = (len_pixels + 1) / 2;
    int i = 0;
#if defined(__ARM_NEON)
    if (pairs >= 8) {
        uint8x8x2_t v_uv;
        v_uv.val[0] = vdup_n_u8(u_val);
        v_uv.val[1] = vdup_n_u8(v_val);
        for (; i <= pairs - 8; i += 8) {
            vst2_u8(ptr + i * 2, v_uv);
        }
    }
#endif
    for (; i < pairs; i++) {
        ptr[i * 2] = u_val;
        ptr[i * 2 + 1] = v_val;
    }
}

static inline void write_yuv_pixel_safe(uint8_t* y_base, uint8_t* uv_base, int width, int height, int stride, 
                                       int x, int y, const YUVColor& c, int thickness) {
    for (int ty = y; ty < y + thickness; ++ty) {
        if (ty < 0 || ty >= height) continue;
        for (int tx = x; tx < x + thickness; ++tx) {
            if (tx < 0 || tx >= width) continue;
            y_base[ty * stride + tx] = c.y;
            int uv_idx = (ty / 2) * stride + (tx & ~1);
            uv_base[uv_idx] = c.u;
            uv_base[uv_idx + 1] = c.v;
        }
    }
}
// ============================================================================
// ============================================================================

static cv::Mat image_buffer_to_gray_mat_view(const ImageBuffer& img_buf) {
    if (!img_buf.is_valid()) return cv::Mat();
    const ImageBuffer& gray_version = img_buf.get_gray_version();
    
    cv::Mat view(gray_version.height, gray_version.w_stride, CV_8UC1, const_cast<void*>(gray_version.get_data()));
    
    if (gray_version.w_stride != gray_version.width) {
        return view(cv::Rect(0, 0, gray_version.width, gray_version.height));
    }
    return view;
}


constexpr PIXEL_FORMAT_E GRAY8 = visiong::kGray8Format;
static void prepare_for_drawing(ImageBuffer& img) {
    if (!img.is_valid()) {
        throw std::runtime_error("Cannot draw on an invalid ImageBuffer.");
    }
    if (img.format == GRAY8) return;
    if (img.format != RK_FMT_BGR888 && img.format != RK_FMT_RGB888) {
        img = img.to_format(RK_FMT_BGR888);
    }
}

class ScopedCpuDrawWrite {
public:
    explicit ScopedCpuDrawWrite(ImageBuffer& img) : img_(img) {
        visiong::bufstate::prepare_cpu_write(img_, visiong::bufstate::AccessIntent::ReadModifyWrite);
    }

    ~ScopedCpuDrawWrite() {
        visiong::bufstate::mark_cpu_write(img_);
    }

private:
    ImageBuffer& img_;
};

[[maybe_unused]] static unsigned char rgb_to_gray(std::tuple<unsigned char, unsigned char, unsigned char> color_rgb) {
    int r = std::get<0>(color_rgb), g = std::get<1>(color_rgb), b = std::get<2>(color_rgb);
    return static_cast<unsigned char>((r * 77 + g * 150 + b * 29) >> 8);
}

// Helper to create a cv::Mat view for drawing (in-place modification) / 用于创建绘图 cv::Mat 视图的辅助函数（原地修改）。
static cv::Mat get_mat_view_for_drawing(ImageBuffer& img) {
    if (img.format == GRAY8) {
        return cv::Mat(img.height, img.w_stride, CV_8UC1, img.get_data());
    }
    return cv::Mat(img.height, img.w_stride, CV_8UC3, img.get_data());
}

static cv::Scalar rgb_to_cv_scalar_for_image(const ImageBuffer& img,
                                             const std::tuple<unsigned char, unsigned char, unsigned char>& color_rgb) {
    const unsigned char r = std::get<0>(color_rgb);
    const unsigned char g = std::get<1>(color_rgb);
    const unsigned char b = std::get<2>(color_rgb);
    // OpenCV writes channels in memory order [0,1,2]. Keep API semantic as RGB. / OpenCV 按内存顺序 [0,1,2] 写通道；对外 API 语义仍保持为 RGB。
    if (img.format == RK_FMT_RGB888) {
        return cv::Scalar(r, g, b);
    }
    return cv::Scalar(b, g, r);
}

struct DrawStringFontAtlas {
    nk_font_atlas atlas{};
    nk_font* font = nullptr;
    std::vector<uint8_t> alpha_atlas;
    std::vector<nk_rune> rune_ranges;
    int atlas_w = 0;
    int atlas_h = 0;
    bool ready = false;

    ~DrawStringFontAtlas() { clear(); }

    void clear() {
        if (ready) {
            nk_font_atlas_clear(&atlas);
        }
        std::memset(&atlas, 0, sizeof(atlas));
        font = nullptr;
        alpha_atlas.clear();
        rune_ranges.clear();
        atlas_w = 0;
        atlas_h = 0;
        ready = false;
    }

    bool init(const std::string& font_path,
              const std::string& pre_chars,
              size_t glyph_budget,
              std::string* source_out = nullptr) {
        clear();

        const visiong::font::FontBlob font_blob = visiong::font::load_font_blob(font_path);
        if (source_out != nullptr) {
            *source_out = font_blob.source;
        }

        nk_font_atlas_init_default(&atlas);
        nk_font_atlas_begin(&atlas);

        rune_ranges = visiong::font::build_rune_ranges(
            font_blob, pre_chars, pre_chars.empty() && !font_blob.using_embedded, glyph_budget);

        struct nk_font_config cfg = nk_font_config(18.0f);
        cfg.oversample_h = 1;
        cfg.oversample_v = 1;
        cfg.fallback_glyph = 0x20;
        cfg.range = rune_ranges.data();

        font = nk_font_atlas_add_from_memory(
            &atlas, const_cast<char*>(font_blob.data), font_blob.size, 18.0f, &cfg);
        if (font == nullptr) {
            clear();
            return false;
        }

        const void* atlas_pixels = nk_font_atlas_bake(&atlas, &atlas_w, &atlas_h, NK_FONT_ATLAS_ALPHA8);
        if (atlas_pixels == nullptr || atlas_w <= 0 || atlas_h <= 0) {
            clear();
            return false;
        }

        alpha_atlas.assign(static_cast<const uint8_t*>(atlas_pixels),
                           static_cast<const uint8_t*>(atlas_pixels) +
                               static_cast<size_t>(atlas_w) * static_cast<size_t>(atlas_h));

        nk_draw_null_texture null_tex;
        nk_font_atlas_end(&atlas, nk_handle_id(0), &null_tex);

        ready = true;
        return true;
    }
};

struct DrawStringFontState {
    std::mutex mutex;
    bool configured = false;
    std::string font_path;
    std::string pre_chars;
    size_t glyph_budget = 6623;
    DrawStringFontAtlas atlas;
};

DrawStringFontState& draw_string_font_state() {
    static DrawStringFontState state;
    return state;
}

bool contains_non_ascii_utf8(const std::string& text) {
    for (const unsigned char ch : text) {
        if (ch >= 0x80) {
            return true;
        }
    }
    return false;
}

inline uint8_t alpha_blend_u8(uint8_t dst, uint8_t src, uint8_t alpha) {
    const uint16_t inv = static_cast<uint16_t>(255 - alpha);
    return static_cast<uint8_t>((static_cast<uint16_t>(src) * alpha + static_cast<uint16_t>(dst) * inv) >> 8);
}

void blend_glyph_on_image(ImageBuffer& img,
                          const DrawStringFontAtlas& atlas,
                          int src_x,
                          int src_y,
                          int src_w,
                          int src_h,
                          int dst_x,
                          int dst_y,
                          int dst_w,
                          int dst_h,
                          const std::tuple<unsigned char, unsigned char, unsigned char>& color_rgb,
                          unsigned char gray_color,
                          const YUVColor& yuv_color) {
    if (dst_w <= 0 || dst_h <= 0 || src_w <= 0 || src_h <= 0) {
        return;
    }

    const float inv_scale_x = static_cast<float>(src_w) / static_cast<float>(dst_w);
    const float inv_scale_y = static_cast<float>(src_h) / static_cast<float>(dst_h);

    const uint8_t r = std::get<0>(color_rgb);
    const uint8_t g = std::get<1>(color_rgb);
    const uint8_t b = std::get<2>(color_rgb);

    uint8_t* data = static_cast<uint8_t*>(img.get_data());
    if (data == nullptr) {
        return;
    }

    const bool is_bgr = (img.format == RK_FMT_BGR888);
    const bool is_rgb = (img.format == RK_FMT_RGB888);
    const bool is_gray = (img.format == GRAY8);
    const bool is_yuv = (img.format == RK_FMT_YUV420SP || img.format == RK_FMT_YUV420SP_VU);

    uint8_t* y_base = nullptr;
    uint8_t* uv_base = nullptr;
    int uv_plane_size = 0;
    const bool uv_is_vu = (img.format == RK_FMT_YUV420SP_VU);
    if (is_yuv) {
        y_base = data;
        uv_base = y_base + (img.w_stride * img.h_stride);
        uv_plane_size = img.w_stride * (img.h_stride / 2);
    }

    for (int oy = 0; oy < dst_h; ++oy) {
        const int ty = dst_y + oy;
        if (ty < 0 || ty >= img.height) {
            continue;
        }

        const int sy = src_y + std::min(src_h - 1, static_cast<int>(oy * inv_scale_y));
        for (int ox = 0; ox < dst_w; ++ox) {
            const int tx = dst_x + ox;
            if (tx < 0 || tx >= img.width) {
                continue;
            }

            const int sx = src_x + std::min(src_w - 1, static_cast<int>(ox * inv_scale_x));
            const uint8_t alpha = atlas.alpha_atlas[static_cast<size_t>(sy) * atlas.atlas_w + static_cast<size_t>(sx)];
            if (alpha == 0) {
                continue;
            }

            if (is_bgr || is_rgb) {
                uint8_t* p = data + ty * img.w_stride * 3 + tx * 3;
                if (is_bgr) {
                    p[0] = alpha_blend_u8(p[0], b, alpha);
                    p[1] = alpha_blend_u8(p[1], g, alpha);
                    p[2] = alpha_blend_u8(p[2], r, alpha);
                } else {
                    p[0] = alpha_blend_u8(p[0], r, alpha);
                    p[1] = alpha_blend_u8(p[1], g, alpha);
                    p[2] = alpha_blend_u8(p[2], b, alpha);
                }
            } else if (is_gray) {
                uint8_t* p = data + ty * img.w_stride + tx;
                *p = alpha_blend_u8(*p, gray_color, alpha);
            } else if (is_yuv) {
                y_base[ty * img.w_stride + tx] = alpha_blend_u8(y_base[ty * img.w_stride + tx], yuv_color.y, alpha);

                const int uv_idx = (ty / 2) * img.w_stride + (tx & ~1);
                if (uv_idx + 1 < uv_plane_size) {
                    if (uv_is_vu) {
                        uv_base[uv_idx] = alpha_blend_u8(uv_base[uv_idx], yuv_color.v, alpha);
                        uv_base[uv_idx + 1] = alpha_blend_u8(uv_base[uv_idx + 1], yuv_color.u, alpha);
                    } else {
                        uv_base[uv_idx] = alpha_blend_u8(uv_base[uv_idx], yuv_color.u, alpha);
                        uv_base[uv_idx + 1] = alpha_blend_u8(uv_base[uv_idx + 1], yuv_color.v, alpha);
                    }
                }
            }
        }
    }
}

bool try_draw_utf8_text(ImageBuffer& img,
                        int x,
                        int y,
                        const std::string& text,
                        std::tuple<unsigned char, unsigned char, unsigned char> color_rgb,
                        double scale,
                        int thickness) {
    auto& state = draw_string_font_state();
    std::lock_guard<std::mutex> lock(state.mutex);

    if (!state.configured) {
        return false;
    }

    if (!state.atlas.ready) {
        if (!state.atlas.init(state.font_path, state.pre_chars, state.glyph_budget, nullptr)) {
            return false;
        }
    }

    if (state.atlas.font == nullptr || state.atlas.alpha_atlas.empty()) {
        return false;
    }

    const float draw_scale = std::max(0.1f, static_cast<float>(scale));

    struct GlyphPlacement {
        const nk_font_glyph* glyph = nullptr;
        float pen_x = 0.0f;
        int src_x = 0;
        int src_y = 0;
        int src_w = 0;
        int src_h = 0;
    };

    std::vector<GlyphPlacement> placements;
    placements.reserve(text.size());

    float pen_x = 0.0f;
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();

    const char* utf8 = text.c_str();
    const int len = static_cast<int>(text.size());
    int offset = 0;

    while (offset < len) {
        nk_rune unicode = 0;
        const int ulen = nk_utf_decode(utf8 + offset, &unicode, len - offset);
        if (ulen <= 0) {
            break;
        }

        const nk_font_glyph* glyph = nk_font_find_glyph(state.atlas.font, unicode);
        if (glyph == nullptr) {
            glyph = state.atlas.font->fallback;
        }

        if (glyph != nullptr) {
            const int sx0 = std::clamp(static_cast<int>(std::floor(glyph->u0 * state.atlas.atlas_w)), 0, state.atlas.atlas_w - 1);
            const int sy0 = std::clamp(static_cast<int>(std::floor(glyph->v0 * state.atlas.atlas_h)), 0, state.atlas.atlas_h - 1);
            const int sx1 = std::clamp(static_cast<int>(std::ceil(glyph->u1 * state.atlas.atlas_w)), sx0 + 1, state.atlas.atlas_w);
            const int sy1 = std::clamp(static_cast<int>(std::ceil(glyph->v1 * state.atlas.atlas_h)), sy0 + 1, state.atlas.atlas_h);

            GlyphPlacement p;
            p.glyph = glyph;
            p.pen_x = pen_x;
            p.src_x = sx0;
            p.src_y = sy0;
            p.src_w = sx1 - sx0;
            p.src_h = sy1 - sy0;

            if (p.src_w > 0 && p.src_h > 0) {
                placements.push_back(p);
                min_x = std::min(min_x, pen_x + glyph->x0 * draw_scale);
                min_y = std::min(min_y, glyph->y0 * draw_scale);
            }

            pen_x += glyph->xadvance * draw_scale;
        }

        offset += ulen;
    }

    if (placements.empty() || !std::isfinite(min_x) || !std::isfinite(min_y)) {
        return false;
    }

    const unsigned char gray_color = rgb_to_gray(color_rgb);
    const YUVColor yuv_color = rgb_to_yuv_bt601(color_rgb);
    const int thick = std::max(1, thickness);

    for (const auto& p : placements) {
        const int dst_w = std::max(1, static_cast<int>(std::lround((p.glyph->x1 - p.glyph->x0) * draw_scale)));
        const int dst_h = std::max(1, static_cast<int>(std::lround((p.glyph->y1 - p.glyph->y0) * draw_scale)));
        const int base_x = x + static_cast<int>(std::lround((p.pen_x + p.glyph->x0 * draw_scale) - min_x));
        const int base_y = y + static_cast<int>(std::lround((p.glyph->y0 * draw_scale) - min_y));

        for (int ty = 0; ty < thick; ++ty) {
            for (int tx = 0; tx < thick; ++tx) {
                blend_glyph_on_image(img,
                                     state.atlas,
                                     p.src_x,
                                     p.src_y,
                                     p.src_w,
                                     p.src_h,
                                     base_x + tx,
                                     base_y + ty,
                                     dst_w,
                                     dst_h,
                                     color_rgb,
                                     gray_color,
                                     yuv_color);
            }
        }
    }

    return true;
}

void ImageBuffer::set_text_font(const std::string& font_path,
                                const std::string& pre_chars,
                                size_t glyph_budget) {
    auto& state = draw_string_font_state();
    std::lock_guard<std::mutex> lock(state.mutex);

    std::string source;
    if (!state.atlas.init(font_path, pre_chars, glyph_budget, &source)) {
        throw std::runtime_error("draw_string font init failed.");
    }

    state.configured = true;
    state.font_path = font_path;
    state.pre_chars = pre_chars;
    state.glyph_budget = glyph_budget;
    VISIONG_LOG_INFO("ImageBuffer",
                     "draw_string font loaded from: " << source
                                                      << ", glyph_budget=" << glyph_budget);
}

void ImageBuffer::clear_text_font() {
    auto& state = draw_string_font_state();
    std::lock_guard<std::mutex> lock(state.mutex);

    state.configured = false;
    state.font_path.clear();
    state.pre_chars.clear();
    state.glyph_budget = 6623;
    state.atlas.clear();
}

// ============================================================================
// draw_rectangle / draw_rectangle（绘制矩形）。
// ============================================================================
ImageBuffer& ImageBuffer::draw_rectangle(int x, int y, int w, int h, 
    std::tuple<unsigned char, unsigned char, unsigned char> color_rgb, 
    int thickness, bool fill) {
    if (this->format != RK_FMT_YUV420SP) {
        prepare_for_drawing(*this);
        ScopedCpuDrawWrite cpu_write(*this);
        cv::Mat mat = get_mat_view_for_drawing(*this);
        cv::rectangle(mat, cv::Rect(x, y, w, h), rgb_to_cv_scalar_for_image(*this, color_rgb), fill ? -1 : thickness, cv::LINE_AA);
        return *this;
    }

    YUVColor c = rgb_to_yuv_bt601(color_rgb);
    uint8_t* y_base = static_cast<uint8_t*>(this->get_data());
    uint8_t* uv_base = y_base + (w_stride * h_stride);

    int x1 = std::max(0, x);
    int y1 = std::max(0, y);
    int x2 = std::min(width, x + w);
    int y2 = std::min(height, y + h);
    if (x1 >= x2 || y1 >= y2) {
        return *this;
    }

    ScopedCpuDrawWrite cpu_write(*this);

    if (fill) {
        for (int cy = y1; cy < y2; ++cy) {
            neon_fill_y_line(y_base + cy * w_stride + x1, x2 - x1, c.y);
            neon_fill_uv_line(uv_base + (cy / 2) * w_stride + (x1 & ~1), x2 - x1, c.u, c.v);
        }
        return *this;
    }

    for (int t = 0; t < thickness; ++t) {
        if (y1 + t < y2) {
            neon_fill_y_line(y_base + (y1 + t) * w_stride + x1, x2 - x1, c.y);
            neon_fill_uv_line(uv_base + ((y1 + t) / 2) * w_stride + (x1 & ~1), x2 - x1, c.u, c.v);
        }
        if (y2 - 1 - t >= y1) {
            neon_fill_y_line(y_base + (y2 - 1 - t) * w_stride + x1, x2 - x1, c.y);
            neon_fill_uv_line(uv_base + ((y2 - 1 - t) / 2) * w_stride + (x1 & ~1), x2 - x1, c.u, c.v);
        }
    }

    for (int cy = y1; cy < y2; ++cy) {
        for (int t = 0; t < thickness; ++t) {
            int lx = x1 + t;
            int rx = x2 - 1 - t;
            if (lx < x2) {
                y_base[cy * w_stride + lx] = c.y;
                int i = (cy / 2) * w_stride + (lx & ~1);
                uv_base[i] = c.u;
                uv_base[i + 1] = c.v;
            }
            if (rx >= x1) {
                y_base[cy * w_stride + rx] = c.y;
                int i = (cy / 2) * w_stride + (rx & ~1);
                uv_base[i] = c.u;
                uv_base[i + 1] = c.v;
            }
        }
    }
    return *this;
}
ImageBuffer& ImageBuffer::draw_rectangle(const std::tuple<int,int,int,int>& rect_tuple, std::tuple<unsigned char, unsigned char, unsigned char> color_rgb, int thickness, bool fill) {
    return draw_rectangle(std::get<0>(rect_tuple), std::get<1>(rect_tuple), std::get<2>(rect_tuple), std::get<3>(rect_tuple), color_rgb, thickness, fill);
}

// ============================================================================
// ============================================================================
ImageBuffer& ImageBuffer::draw_line(int x0, int y0, int x1, int y1, std::tuple<unsigned char, unsigned char, unsigned char> color_rgb, int thickness) {
    if (this->format != RK_FMT_YUV420SP) {
        prepare_for_drawing(*this);
        ScopedCpuDrawWrite cpu_write(*this);
        cv::Mat mat = get_mat_view_for_drawing(*this);
        cv::line(mat, cv::Point(x0, y0), cv::Point(x1, y1), rgb_to_cv_scalar_for_image(*this, color_rgb), thickness, cv::LINE_AA);
        return *this;
    }

    ScopedCpuDrawWrite cpu_write(*this);
    YUVColor c = rgb_to_yuv_bt601(color_rgb);
    uint8_t* y_p = (uint8_t*)this->get_data();
    uint8_t* uv_p = y_p + (w_stride * h_stride);

    int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;

    while (true) {
        write_yuv_pixel_safe(y_p, uv_p, width, height, w_stride, x0, y0, c, thickness);
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
    return *this;
}

// ============================================================================
// draw_circle / draw_circle（绘制圆）。
// ============================================================================
ImageBuffer& ImageBuffer::draw_circle(int cx, int cy, int radius, std::tuple<unsigned char, unsigned char, unsigned char> color_rgb, int thickness, bool fill) {
    if (this->format != RK_FMT_YUV420SP) {
        prepare_for_drawing(*this);
        ScopedCpuDrawWrite cpu_write(*this);
        cv::Mat mat = get_mat_view_for_drawing(*this);
        cv::circle(mat, cv::Point(cx, cy), radius, rgb_to_cv_scalar_for_image(*this, color_rgb), fill ? -1 : thickness, cv::LINE_AA);
        return *this;
    }

    ScopedCpuDrawWrite cpu_write(*this);
    YUVColor c = rgb_to_yuv_bt601(color_rgb);
    uint8_t* y_p = (uint8_t*)this->get_data();
    uint8_t* uv_p = y_p + (w_stride * h_stride);

    if (fill) {
        for (int y = -radius; y <= radius; y++) {
            int ty = cy + y;
            if (ty < 0 || ty >= height) continue;
            int len = (int)std::sqrt(radius * radius - y * y);
            int x_start = std::max(0, cx - len);
            int x_end = std::min(width, cx + len);
            if (x_end > x_start) {
                neon_fill_y_line(y_p + ty * w_stride + x_start, x_end - x_start, c.y);
                neon_fill_uv_line(uv_p + (ty / 2) * w_stride + (x_start & ~1), x_end - x_start, c.u, c.v);
            }
        }
    } else {
        int x = 0, y = radius, d = 3 - 2 * radius;
        auto plot8 = [&](int x, int y) {
            int pts[8][2] = {{cx+x, cy+y}, {cx-x, cy+y}, {cx+x, cy-y}, {cx-x, cy-y}, 
                             {cx+y, cy+x}, {cx-y, cy+x}, {cx+y, cy-x}, {cx-y, cy-x}};
            for (auto& p : pts) write_yuv_pixel_safe(y_p, uv_p, width, height, w_stride, p[0], p[1], c, thickness);
        };
        while (y >= x) {
            plot8(x, y);
            if (d < 0) d += 4 * x + 6;
            else { d += 4 * (x - y) + 10; y--; }
            x++;
        }
    }
    return *this;
}

// ============================================================================
// ============================================================================
ImageBuffer& ImageBuffer::draw_string(int x, int y, const std::string& text,
                                     std::tuple<unsigned char, unsigned char, unsigned char> color_rgb,
                                     double scale, int thickness) {
    if (text.empty()) {
        return *this;
    }

    // UTF-8 (e.g. Chinese) path: use the shared glyph atlas used by GUI font loading. / UTF-8（如中文）路径：使用与 GUI 字体加载共享的字形图集。
    if (contains_non_ascii_utf8(text)) {
        if (this->format != RK_FMT_YUV420SP && this->format != RK_FMT_YUV420SP_VU &&
            this->format != GRAY8 && this->format != RK_FMT_BGR888 && this->format != RK_FMT_RGB888) {
            prepare_for_drawing(*this);
        }
        ScopedCpuDrawWrite cpu_write(*this);
        if (try_draw_utf8_text(*this, x, y, text, color_rgb, scale, thickness)) {
            return *this;
        }
    }

    int baseline = 0;
    cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, thickness, &baseline);
    // (x, y) is TOP-LEFT of the text box so that y=10 shows text fully below; OpenCV putText uses bottom-left baseline. / (x, y) 表示文本框左上角，因此 y=10 时文本会完整显示在其下方；OpenCV putText 使用的是左下基线。
    int baseline_y = y + text_size.height + baseline;

    if (this->format != RK_FMT_YUV420SP) {
        prepare_for_drawing(*this);
        ScopedCpuDrawWrite cpu_write(*this);
        cv::Mat mat = get_mat_view_for_drawing(*this);
        cv::putText(mat, text, cv::Point(x, baseline_y), cv::FONT_HERSHEY_SIMPLEX, scale,
                    rgb_to_cv_scalar_for_image(*this, color_rgb), thickness, cv::LINE_AA);
        return *this;
    }

    ScopedCpuDrawWrite cpu_write(*this);
    YUVColor c = rgb_to_yuv_bt601(color_rgb);
    uint8_t* y_base = (uint8_t*)this->get_data();
    uint8_t* uv_base = y_base + (w_stride * h_stride);

    cv::Mat mask = cv::Mat::zeros(text_size.height + baseline + 4, text_size.width + 4, CV_8UC1);
    cv::putText(mask, text, cv::Point(2, text_size.height + 2), cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(255), thickness, cv::LINE_AA);

    for (int r = 0; r < mask.rows; r++) {
        int target_y = y + r;
        if (target_y < 0 || target_y >= height) continue;

        uint8_t* mask_row = mask.ptr<uint8_t>(r);
        for (int col = 0; col < mask.cols; ) {
            if (mask_row[col] > 0) {
                int start_col = col;
                while (col < mask.cols && mask_row[col] > 0) col++;

                int target_x = x + start_col - 2;
                int run_len = col - start_col;

                if (target_x < 0) { run_len += target_x; target_x = 0; }
                if (target_x + run_len > width) run_len = width - target_x;

                if (run_len > 0) {
                    neon_fill_y_line(y_base + target_y * w_stride + target_x, run_len, c.y);
                    neon_fill_uv_line(uv_base + (target_y/2) * w_stride + (target_x & ~1), run_len, c.u, c.v);
                }
            } else {
                col++;
            }
        }
    }
    return *this;
}

// ============================================================================
// draw_cross / draw_cross（绘制十字）。
// ============================================================================
ImageBuffer& ImageBuffer::draw_cross(int cx, int cy, std::tuple<unsigned char, unsigned char, unsigned char> color_rgb, int size, int thickness) {
    this->draw_line(cx - size / 2, cy, cx + size / 2, cy, color_rgb, thickness);
    this->draw_line(cx, cy - size / 2, cx, cy + size / 2, color_rgb, thickness);
    return *this;
}

ImageBuffer ImageBuffer::binarize(const std::string& method,
                                  const std::tuple<int, int>& threshold_range,
                                  bool invert,
                                  int adaptive_block_size,
                                  int adaptive_c,
                                  int pre_blur_kernel_size,
                                  int post_morph_kernel_size
                                  ) const {
    if (!is_valid()) {
        throw std::runtime_error("binarize: Cannot operate on an invalid ImageBuffer.");
    }

    const ImageBuffer& gray_img = this->get_gray_version();
    if (!gray_img.is_valid()) {
        throw std::runtime_error("binarize: Failed to get grayscale version of the image.");
    }

    cv::Mat gray_mat = image_buffer_to_gray_mat_view(gray_img);
    if (gray_mat.empty()) {
        throw std::runtime_error("binarize: Failed to create cv::Mat view from grayscale ImageBuffer.");
    }

    // Step A: preprocessing with configurable Gaussian blur. / 步骤 A：使用可配置高斯模糊做预处理。
    if (pre_blur_kernel_size >= 3) {
        if (pre_blur_kernel_size % 2 == 0) {
            throw std::invalid_argument("pre_blur_kernel_size must be an odd number.");
        }
        cv::GaussianBlur(gray_mat, gray_mat, cv::Size(pre_blur_kernel_size, pre_blur_kernel_size), 0);
    }

    cv::Mat binary_mat;

    if (method == "manual") {
        int low_thresh = std::get<0>(threshold_range);
        int high_thresh = std::get<1>(threshold_range);
        cv::inRange(gray_mat, low_thresh, high_thresh, binary_mat);
        if (invert) cv::bitwise_not(binary_mat, binary_mat);
    } else if (method == "otsu") {
        int flags = invert ? (cv::THRESH_BINARY_INV | cv::THRESH_OTSU) : (cv::THRESH_BINARY | cv::THRESH_OTSU);
        cv::threshold(gray_mat, binary_mat, 0, 255, flags);
    } else if (method == "adaptive_mean" || method == "adaptive_gaussian") {
        if (adaptive_block_size <= 1 || adaptive_block_size % 2 == 0) {
            throw std::invalid_argument("adaptive_block_size must be an odd number greater than 1.");
        }
        int adaptive_method = (method == "adaptive_mean") ? cv::ADAPTIVE_THRESH_MEAN_C : cv::ADAPTIVE_THRESH_GAUSSIAN_C;
        int threshold_type = invert ? cv::THRESH_BINARY_INV : cv::THRESH_BINARY;
        cv::adaptiveThreshold(gray_mat, binary_mat, 255, adaptive_method, threshold_type, adaptive_block_size, adaptive_c);
    } else {
        throw std::invalid_argument("Unknown binarization method: '" + method + "'. Supported methods are: 'manual', 'otsu', 'adaptive_mean', 'adaptive_gaussian'.");
    }

    if (post_morph_kernel_size >= 2) {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(post_morph_kernel_size, post_morph_kernel_size));
        cv::morphologyEx(binary_mat, binary_mat, cv::MORPH_OPEN, kernel);
    }

    return ImageBuffer::from_cv_mat(binary_mat, visiong::kGray8Format);
}
#if defined(__ARM_NEON)
[[maybe_unused]] static void blend_rgba_on_bgr_neon(const ImageBuffer& src_rgba, ImageBuffer& dst_bgr, int x_offset, int y_offset) {
    int w = src_rgba.width;
    int h = src_rgba.height;

    const uint8_t* src_base = (const uint8_t*)src_rgba.get_data();
    uint8_t* dst_base = (uint8_t*)dst_bgr.get_data();

    uint8x8_t u8_255 = vdup_n_u8(255);

    for (int y = 0; y < h; ++y) {
        const uint8_t* src_row = src_base + y * src_rgba.w_stride * 4;
        uint8_t* dst_row = dst_base + (y + y_offset) * dst_bgr.w_stride * 3 + x_offset * 3;
        
        for (int x = 0; x <= w - 8; x += 8) {
            uint8x8x4_t rgba = vld4_u8(src_row + x * 4); // R G B A
            uint8x8x3_t bgr = vld3_u8(dst_row + x * 3);  // B G R

            uint8x8_t alpha = rgba.val[3];
            if (vget_lane_u64(vreinterpret_u64_u8(alpha), 0) == 0) continue;

            uint8x8_t inv_alpha = vsub_u8(u8_255, alpha);

            uint16x8_t r_wide = vmovl_u8(rgba.val[0]);
            uint16x8_t g_wide = vmovl_u8(rgba.val[1]);
            uint16x8_t b_wide = vmovl_u8(rgba.val[2]);

            uint16x8_t dst_r_wide = vmovl_u8(bgr.val[2]);
            uint16x8_t dst_g_wide = vmovl_u8(bgr.val[1]);
            uint16x8_t dst_b_wide = vmovl_u8(bgr.val[0]);

            uint16x8_t alpha_wide = vmovl_u8(alpha);
            uint16x8_t inv_alpha_wide = vmovl_u8(inv_alpha);
            
            // Dst.B = (Src.B * A + Dst.B * (255-A)) >> 8 / 目标蓝色通道按该公式进行 Alpha 混合。
            uint16x8_t out_b_wide = vmlaq_u16(vmulq_u16(dst_b_wide, inv_alpha_wide), b_wide, alpha_wide);
            uint16x8_t out_g_wide = vmlaq_u16(vmulq_u16(dst_g_wide, inv_alpha_wide), g_wide, alpha_wide);
            uint16x8_t out_r_wide = vmlaq_u16(vmulq_u16(dst_r_wide, inv_alpha_wide), r_wide, alpha_wide);
            
            bgr.val[0] = vshrn_n_u16(out_b_wide, 8);
            bgr.val[1] = vshrn_n_u16(out_g_wide, 8);
            bgr.val[2] = vshrn_n_u16(out_r_wide, 8);

            vst3_u8(dst_row + x * 3, bgr);
        }
        for (int x = w - (w % 8); x < w; ++x) {
            const uint8_t* p_src = src_row + x * 4;
            uint8_t* p_dst = dst_row + x * 3;
            uint8_t alpha = p_src[3];
            if (alpha == 0) continue;
            if (alpha == 255) {
                p_dst[0] = p_src[2]; // B
                p_dst[1] = p_src[1]; // G
                p_dst[2] = p_src[0]; // R
            } else {
                uint16_t inv_alpha = 255 - alpha;
                p_dst[0] = (uint8_t)((p_src[2] * alpha + p_dst[0] * inv_alpha) >> 8);
                p_dst[1] = (uint8_t)((p_src[1] * alpha + p_dst[1] * inv_alpha) >> 8);
                p_dst[2] = (uint8_t)((p_src[0] * alpha + p_dst[2] * inv_alpha) >> 8);
            }
        }
    }
}
#endif

[[maybe_unused]] static void blend_rgba_on_bgr_c(const ImageBuffer& src_rgba, ImageBuffer& dst_bgr, int x_offset, int y_offset) {
    int w = src_rgba.width;
    int h = src_rgba.height;

    const uint8_t* src_base = (const uint8_t*)src_rgba.get_data();
    uint8_t* dst_base = (uint8_t*)dst_bgr.get_data();

    for (int y = 0; y < h; ++y) {
        const uint8_t* src_row = src_base + y * src_rgba.w_stride * 4;
        uint8_t* dst_row = dst_base + (y + y_offset) * dst_bgr.w_stride * 3 + x_offset * 3;
        
        for (int x = 0; x < w; ++x) {
            const uint8_t* p_src = src_row + x * 4;
            uint8_t* p_dst = dst_row + x * 3;
            uint8_t alpha = p_src[3];

            if (alpha == 0) continue;
            if (alpha == 255) {
                p_dst[0] = p_src[2]; // B
                p_dst[1] = p_src[1]; // G
                p_dst[2] = p_src[0]; // R
            } else {
                uint16_t inv_alpha = 255 - alpha;
                p_dst[0] = (uint8_t)((p_src[2] * alpha + p_dst[0] * inv_alpha) >> 8);
                p_dst[1] = (uint8_t)((p_src[1] * alpha + p_dst[1] * inv_alpha) >> 8);
                p_dst[2] = (uint8_t)((p_src[0] * alpha + p_dst[2] * inv_alpha) >> 8);
            }
        }
    }
}

// High-performance alpha blending (safe C++ path). / 高性能 Alpha 混合（安全的 C++ 路径）。
ImageBuffer& ImageBuffer::blend(const ImageBuffer& img_to_blend, int x_offset, int y_offset) {
    if (!this->is_valid() || !img_to_blend.is_valid()) {
        throw std::runtime_error("blend: Both source and destination images must be valid.");
    }
    if (this->is_zero_copy()) {
        *this = this->copy();
    }
    if (this->format != RK_FMT_BGR888) {
        *this = this->to_format(RK_FMT_BGR888);
    }
    const ImageBuffer* src_rgba = &img_to_blend;
    ImageBuffer converted_src_owner;
    if (img_to_blend.format != RK_FMT_RGBA8888) {
        converted_src_owner = img_to_blend.to_format(RK_FMT_RGBA8888);
        src_rgba = &converted_src_owner;
    }
    
    int w = src_rgba->width;
    int h = src_rgba->height;

    const uint8_t* src_base = (const uint8_t*)src_rgba->get_data();
    uint8_t* dst_base = (uint8_t*)this->get_data();

    for (int y = 0; y < h; ++y) {
        int dst_y = y + y_offset;
        if (dst_y < 0 || dst_y >= this->height) continue;

        const uint8_t* src_row = src_base + y * src_rgba->w_stride * 4;
        uint8_t* dst_row = dst_base + dst_y * this->w_stride * 3;

        for (int x = 0; x < w; ++x) {
            int dst_x = x + x_offset;
            if (dst_x < 0 || dst_x >= this->width) continue;

            const uint8_t* p_src = src_row + x * 4; // R,G,B,A
            uint8_t* p_dst = dst_row + dst_x * 3;     // B,G,R

            uint32_t alpha = p_src[3];
            if (alpha == 0) continue;
            if (alpha == 255) {
                p_dst[0] = p_src[2]; // B
                p_dst[1] = p_src[1]; // G
                p_dst[2] = p_src[0]; // R
            } else {
                uint32_t inv_alpha = 255 - alpha;
                p_dst[0] = (uint8_t)((p_src[2] * alpha + p_dst[0] * inv_alpha) >> 8);
                p_dst[1] = (uint8_t)((p_src[1] * alpha + p_dst[1] * inv_alpha) >> 8);
                p_dst[2] = (uint8_t)((p_src[0] * alpha + p_dst[2] * inv_alpha) >> 8);
            }
        }
    }
    return *this;
}

