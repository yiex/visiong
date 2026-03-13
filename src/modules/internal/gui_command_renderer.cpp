// SPDX-License-Identifier: LGPL-3.0-or-later
#include "modules/internal/gui_command_renderer.h"

#include "modules/internal/gui_render_utils.h"
#include "visiong/core/ImageBuffer.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace visiong::gui::render {
namespace {

template <typename RenderGlyphFn>
void render_nk_command_text_safe(const nk_command_text* command,
                                 const std::vector<uint8_t>& atlas_data,
                                 int atlas_w,
                                 int atlas_h,
                                 RenderGlyphFn&& render_glyph) {
    if (command == nullptr || !command->font || !command->font->userdata.ptr) {
        return;
    }
    if (atlas_data.empty() || atlas_w <= 0 || atlas_h <= 0) {
        return;
    }

    const nk_font* font = static_cast<const nk_font*>(command->font->userdata.ptr);
    if (font == nullptr || command->string == nullptr) {
        return;
    }

    const int len = std::min(std::max(0, static_cast<int>(command->length)), 4096);
    int offset = 0;
    int glyph_count = 0;
    float pen_x = command->x;
    constexpr int kMaxGlyphsPerCommand = 4096;

    while (offset < len && glyph_count < kMaxGlyphsPerCommand) {
        nk_rune unicode = 0;
        const int decoded = nk_utf_decode(command->string + offset, &unicode, len - offset);
        if (decoded <= 0) {
            break;
        }
        offset += decoded;
        ++glyph_count;

        const nk_font_glyph* glyph = nk_font_find_glyph(font, unicode);
        if (glyph == nullptr) {
            glyph = font->fallback;
        }
        if (glyph == nullptr) {
            continue;
        }

        const int src_x0 = std::clamp(static_cast<int>(std::floor(glyph->u0 * atlas_w)), 0, atlas_w - 1);
        const int src_y0 = std::clamp(static_cast<int>(std::floor(glyph->v0 * atlas_h)), 0, atlas_h - 1);
        const int src_x1 = std::clamp(static_cast<int>(std::ceil(glyph->u1 * atlas_w)), src_x0 + 1, atlas_w);
        const int src_y1 = std::clamp(static_cast<int>(std::ceil(glyph->v1 * atlas_h)), src_y0 + 1, atlas_h);
        const int src_w = src_x1 - src_x0;
        const int src_h = src_y1 - src_y0;
        if (src_w <= 0 || src_h <= 0) {
            pen_x += glyph->xadvance;
            continue;
        }

        const int dst_x = static_cast<int>(std::floor(pen_x + glyph->x0));
        const int dst_y = static_cast<int>(std::floor(command->y + glyph->y0));
        render_glyph(dst_x, dst_y, src_w, src_h, src_x0, src_y0);
        pen_x += glyph->xadvance;
    }
}

void blit_image_nearest_rgba8888(const ImageBuffer& src_img,
                                 const nk_command_image& image_cmd,
                                 ImageBuffer& target) {
    const ImageBuffer& bgr_src = src_img.get_bgr_version();
    const uint8_t* src_data = static_cast<const uint8_t*>(bgr_src.get_data());
    uint8_t* dst_data = static_cast<uint8_t*>(target.get_data());
    if (src_data == nullptr || dst_data == nullptr || bgr_src.width <= 0 || bgr_src.height <= 0) {
        return;
    }

    const int src_w = bgr_src.width;
    const int src_h = bgr_src.height;
    const int src_stride = bgr_src.w_stride;
    const int dst_w = image_cmd.w;
    const int dst_h = image_cmd.h;
    const float x_ratio = dst_w > 1 ? static_cast<float>(src_w - 1) / static_cast<float>(dst_w - 1) : 0.0f;
    const float y_ratio = dst_h > 1 ? static_cast<float>(src_h - 1) / static_cast<float>(dst_h - 1) : 0.0f;

    for (int y_dst = 0; y_dst < dst_h; ++y_dst) {
        const int current_y = image_cmd.y + y_dst;
        if (current_y < 0 || current_y >= target.height) {
            continue;
        }
        const int src_y = std::clamp(static_cast<int>(y_dst * y_ratio), 0, src_h - 1);
        const uint8_t* src_row = src_data + src_y * src_stride * 3;
        uint8_t* dst_row = dst_data + current_y * target.w_stride * 4;

        for (int x_dst = 0; x_dst < dst_w; ++x_dst) {
            const int current_x = image_cmd.x + x_dst;
            if (current_x < 0 || current_x >= target.width) {
                continue;
            }
            const int src_x = std::clamp(static_cast<int>(x_dst * x_ratio), 0, src_w - 1);
            const uint8_t* src_pixel = src_row + src_x * 3;
            uint8_t* dst_pixel = dst_row + current_x * 4;
            dst_pixel[0] = src_pixel[2];
            dst_pixel[1] = src_pixel[1];
            dst_pixel[2] = src_pixel[0];
            dst_pixel[3] = 255;
        }
    }
}

void render_commands_to_rgba8888(nk_context* ctx,
                                 ImageBuffer& target,
                                 const std::vector<uint8_t>& atlas_data,
                                 int atlas_w,
                                 int atlas_h) {
    uint8_t* target_cpu_addr = static_cast<uint8_t*>(target.get_data());
    if (target_cpu_addr == nullptr) {
        return;
    }

    const int target_w_stride = target.w_stride;
    const struct nk_command* cmd = nullptr;
    nk_foreach(cmd, ctx) {
        switch (cmd->type) {
            case NK_COMMAND_RECT_FILLED: {
                const auto* rect = reinterpret_cast<const nk_command_rect_filled*>(cmd);
                cpu_fill_rect_rgba8888(target_cpu_addr, target.width, target.height, target_w_stride,
                                       rect->x, rect->y, rect->w, rect->h, rect->color);
                break;
            }
            case NK_COMMAND_IMAGE: {
                const auto* image = reinterpret_cast<const nk_command_image*>(cmd);
                if (image->img.handle.ptr == nullptr) {
                    break;
                }
                const auto* src_img = static_cast<const ImageBuffer*>(image->img.handle.ptr);
                if (!src_img->is_valid()) {
                    break;
                }
                blit_image_nearest_rgba8888(*src_img, *image, target);
                break;
            }
            case NK_COMMAND_TEXT: {
                const auto* text = reinterpret_cast<const nk_command_text*>(cmd);
                render_nk_command_text_safe(
                    text,
                    atlas_data,
                    atlas_w,
                    atlas_h,
                    [&](int dst_x, int dst_y, int src_w, int src_h, int src_x, int src_y) {
                        cpu_render_text_rgba8888(target_cpu_addr,
                                                 target.width,
                                                 target.height,
                                                 target_w_stride,
                                                 atlas_data.data(),
                                                 atlas_w,
                                                 dst_x,
                                                 dst_y,
                                                 src_w,
                                                 src_h,
                                                 src_x,
                                                 src_y,
                                                 text->foreground);
                    });
                break;
            }
            default:
                break;
        }
    }
}

void render_commands_to_bgr888(nk_context* ctx,
                               ImageBuffer& target,
                               const std::vector<uint8_t>& atlas_data,
                               int atlas_w,
                               int atlas_h) {
    uint8_t* target_cpu_addr = static_cast<uint8_t*>(target.get_data());
    if (target_cpu_addr == nullptr) {
        return;
    }

    const int target_w_stride = target.w_stride;
    const struct nk_command* cmd = nullptr;
    nk_foreach(cmd, ctx) {
        switch (cmd->type) {
            case NK_COMMAND_RECT_FILLED: {
                const auto* rect = reinterpret_cast<const nk_command_rect_filled*>(cmd);
                cpu_fill_rect_bgr888(target_cpu_addr, target.width, target.height, target_w_stride,
                                     rect->x, rect->y, rect->w, rect->h, rect->color);
                break;
            }
            case NK_COMMAND_IMAGE: {
                const auto* image = reinterpret_cast<const nk_command_image*>(cmd);
                if (image->img.handle.ptr == nullptr) {
                    break;
                }
                const auto* src_img = static_cast<const ImageBuffer*>(image->img.handle.ptr);
                if (!src_img->is_valid()) {
                    break;
                }
                const ImageBuffer& bgr_src = src_img->get_bgr_version();
                cpu_resize_bgr888_bilinear(static_cast<const uint8_t*>(bgr_src.get_data()),
                                           bgr_src.width,
                                           bgr_src.height,
                                           bgr_src.w_stride,
                                           target_cpu_addr,
                                           target.width,
                                           target.height,
                                           target_w_stride,
                                           image->x,
                                           image->y,
                                           image->w,
                                           image->h);
                break;
            }
            case NK_COMMAND_TEXT: {
                const auto* text = reinterpret_cast<const nk_command_text*>(cmd);
                render_nk_command_text_safe(
                    text,
                    atlas_data,
                    atlas_w,
                    atlas_h,
                    [&](int dst_x, int dst_y, int src_w, int src_h, int src_x, int src_y) {
                        cpu_render_text_bgr888(target_cpu_addr,
                                               target.width,
                                               target.height,
                                               target_w_stride,
                                               atlas_data.data(),
                                               atlas_w,
                                               dst_x,
                                               dst_y,
                                               src_w,
                                               src_h,
                                               src_x,
                                               src_y,
                                               text->foreground);
                    });
                break;
            }
            default:
                break;
        }
    }
}

}  // namespace

struct nk_image image_from_buffer(const ImageBuffer* img) {
    struct nk_image image{};
    image.handle.ptr = const_cast<ImageBuffer*>(img);
    image.w = img ? img->width : 0;
    image.h = img ? img->height : 0;
    image.region[0] = 0;
    image.region[1] = 0;
    image.region[2] = image.w;
    image.region[3] = image.h;
    return image;
}

void render_commands_to_image(nk_context* ctx,
                              ImageBuffer& target,
                              const std::vector<uint8_t>& atlas_data,
                              int atlas_w,
                              int atlas_h) {
    if (ctx == nullptr || !target.is_valid()) {
        return;
    }

    if (target.format == RK_FMT_RGBA8888) {
        render_commands_to_rgba8888(ctx, target, atlas_data, atlas_w, atlas_h);
        return;
    }

    ImageBuffer owned_render_target;
    ImageBuffer* render_target_ptr = nullptr;
    bool needs_writeback = false;

    if (target.format == RK_FMT_BGR888 && !target.is_zero_copy()) {
        render_target_ptr = &target;
    } else {
        owned_render_target = target.copy();
        if (owned_render_target.format != RK_FMT_BGR888) {
            owned_render_target = owned_render_target.to_format(RK_FMT_BGR888);
        }
        render_target_ptr = &owned_render_target;
        needs_writeback = true;
    }

    ImageBuffer& render_target = *render_target_ptr;
    render_commands_to_bgr888(ctx, render_target, atlas_data, atlas_w, atlas_h);
    if (needs_writeback) {
        target = std::move(render_target);
    }
}

}  // namespace visiong::gui::render