// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_MODULES_INTERNAL_GUI_COMMAND_RENDERER_H
#define VISIONG_MODULES_INTERNAL_GUI_COMMAND_RENDERER_H

#include "modules/internal/gui_nuklear_config.h"

#include <cstdint>
#include <vector>

class ImageBuffer;

namespace visiong::gui::render {

struct nk_image image_from_buffer(const ImageBuffer* img);
void render_commands_to_image(nk_context* ctx,
                              ImageBuffer& target,
                              const std::vector<uint8_t>& atlas_data,
                              int atlas_w,
                              int atlas_h);

}  // namespace visiong::gui::render

#endif  // VISIONG_MODULES_INTERNAL_GUI_COMMAND_RENDERER_H