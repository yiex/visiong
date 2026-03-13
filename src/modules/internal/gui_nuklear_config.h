// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_MODULES_INTERNAL_GUI_NUKLEAR_CONFIG_H
#define VISIONG_MODULES_INTERNAL_GUI_NUKLEAR_CONFIG_H

#define NK_BUTTON_TRIGGER_ON_RELEASE
#define NK_INCLUDE_FIXED_TYPES
#define NK_INCLUDE_STANDARD_IO
#define NK_INCLUDE_DEFAULT_ALLOCATOR
#define NK_INCLUDE_FONT_BAKING
#define NK_INCLUDE_DEFAULT_FONT

#ifdef VISIONG_NUKLEAR_IMPLEMENTATION
#define NK_IMPLEMENTATION
#endif

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "nuklear.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#ifdef VISIONG_NUKLEAR_IMPLEMENTATION
#undef NK_IMPLEMENTATION
#endif

#endif  // VISIONG_MODULES_INTERNAL_GUI_NUKLEAR_CONFIG_H
