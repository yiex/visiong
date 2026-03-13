// SPDX-License-Identifier: LGPL-3.0-or-later

#include "visiong/modules/GUIManager.h"
#include "visiong/core/ImageBuffer.h"
#include "core/internal/logger.h"
#include "modules/internal/font_support.h"

#include <stdexcept>

#define VISIONG_NUKLEAR_IMPLEMENTATION
#include "modules/internal/gui_nuklear_style.h"
#include "modules/internal/gui_command_renderer.h"
#include "modules/internal/gui_manager_impl.h"

namespace gui_style = visiong::gui;
namespace gui_render = visiong::gui::render;

GUIManager::Impl::Impl(int w, int h, const std::string& font_path, const std::string& pre_chars)
    : m_ctx(new nk_context), m_atlas(new nk_font_atlas), m_cmds(new nk_buffer), m_width(w), m_height(h) {
    nk_font_atlas_init_default(m_atlas);
    nk_font_atlas_begin(m_atlas);

    const visiong::font::FontBlob font_blob = visiong::font::load_font_blob(font_path);
    VISIONG_LOG_INFO("GUI", "Loading font from: " << font_blob.source);

    if (!pre_chars.empty()) {
        VISIONG_LOG_INFO("GUI", "Pre-defining " << pre_chars.size() << " bytes of UTF-8 chars.");
    } else if (!font_blob.using_embedded) {
        VISIONG_LOG_INFO("GUI", "No pre-defined chars, loading full CJK character set.");
    }

    const gui_style::TouchUiProfile profile = gui_style::make_touch_ui_profile(w, h);
    m_ui_scale = profile.ui_scale;
    const float font_size = profile.font_size;
    m_min_row_height = profile.min_row_height;
    m_touch_slop = profile.touch_slop;
    m_long_press_seconds = profile.long_press_seconds;
    m_min_fling_velocity = profile.min_fling_velocity;
    m_max_fling_velocity = profile.max_fling_velocity;
    m_momentum_deceleration = profile.momentum_deceleration;
    m_scroll_pixels_per_unit = profile.scroll_pixels_per_unit;

    struct nk_font_config cfg = nk_font_config(font_size);
    cfg.oversample_h = 1;
    cfg.oversample_v = 1;
    cfg.fallback_glyph = 0x20;

    m_rune_ranges = visiong::font::build_rune_ranges(
        font_blob,
        pre_chars,
        pre_chars.empty() && !font_blob.using_embedded,
        8192);
    cfg.range = m_rune_ranges.data();

    nk_font* font = nk_font_atlas_add_from_memory(
        m_atlas,
        const_cast<char*>(font_blob.data),
        font_blob.size,
        font_size,
        &cfg);
    if (!font) {
        throw std::runtime_error("[GUI] Fatal: nk_font_atlas_add_from_memory failed.");
    }

    const void* atlas_pixels =
        nk_font_atlas_bake(m_atlas, &m_atlas_width, &m_atlas_height, NK_FONT_ATLAS_ALPHA8);
    if (!atlas_pixels) {
        throw std::runtime_error(
            "[GUI] Fatal: Failed to bake font atlas (OOM). Use 'predefine_chars' to reduce memory.");
    }

    m_font_atlas_data.assign(
        static_cast<const uint8_t*>(atlas_pixels),
        static_cast<const uint8_t*>(atlas_pixels) + static_cast<size_t>(m_atlas_width) * m_atlas_height);

    nk_draw_null_texture null_tex;
    nk_font_atlas_end(m_atlas, nk_handle_id(0), &null_tex);

    if (!nk_init_default(m_ctx, &font->handle)) {
        throw std::runtime_error("[GUI] Fatal: nk_init_default failed.");
    }

    nk_buffer_init_default(m_cmds);
    m_ctx->style.window.fixed_background = nk_style_item_color(nk_rgb(50, 50, 50));
    gui_style::apply_touch_friendly_style(m_ctx, profile);
}

GUIManager::Impl::~Impl() {
    nk_font_atlas_clear(m_atlas);
    nk_free(m_ctx);
    nk_buffer_free(m_cmds);
    delete m_atlas;
    delete m_cmds;
    delete m_ctx;
}

bool GUIManager::Impl::beginWindow(const std::string& title,
                                   float x,
                                   float y,
                                   float w,
                                   float h,
                                   const std::string& flags_str) {
    const bool visible = nk_begin(
        m_ctx,
        title.c_str(),
        nk_rect(x, y, w, h),
        gui_style::parse_window_flags(flags_str));
    if (visible) {
        nk_layout_set_min_row_height(m_ctx, m_min_row_height);
    }
    return visible;
}

void GUIManager::Impl::endWindow() {
    nk_end(m_ctx);
}

void GUIManager::Impl::endFrame(ImageBuffer& target) {
    render(target);
    nk_clear(m_ctx);
}

void GUIManager::Impl::render(ImageBuffer& target) {
    gui_render::render_commands_to_image(m_ctx, target, m_font_atlas_data, m_atlas_width, m_atlas_height);
}

GUIManager::GUIManager(int w, int h)
    : m_impl(std::make_unique<Impl>(w, h, "", "")) {}

GUIManager::GUIManager(int w, int h, const std::string& font_path, const std::string& pre_chars)
    : m_impl(std::make_unique<Impl>(w, h, font_path, pre_chars)) {
    try {
        ImageBuffer::set_text_font(font_path, pre_chars);
    } catch (const std::exception& e) {
        VISIONG_LOG_WARN("GUI", "Failed to sync draw_string font from GUI font settings: " << e.what());
    }
}
