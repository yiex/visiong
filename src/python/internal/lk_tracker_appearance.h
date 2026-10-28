// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#ifndef VISIONG_LK_APPEARANCE_CORE_ONLY
#include <pybind11/pybind11.h>
#endif

namespace visiong {
namespace python {
namespace lk_appearance {

using Pixel = std::array<int, 2>;
using Point = std::array<double, 2>;

enum class Reason {
    None,
    PatchBorder,
    LowTexture,
    Photometric,
    Ambiguous,
    AppearanceBudget,
};

struct Options {
    double max_patch_error = 12.0;
    double ambiguity_ratio = 0.2;
    int ambiguity_radius = 16;
};

struct AppearanceResult {
    Pixel point{};
    Reason reason = Reason::None;
    bool has_evidence = false;
    Pixel offset{};
    double relative_error = 0.0;
};

struct MatchResult {
    bool has_error = false;
    double error = 0.0;
    Reason photometric_reason = Reason::None;
    Reason appearance_reason = Reason::None;
    Reason reason = Reason::None;
};

struct Metrics {
    std::size_t photometric_checks = 0;
    std::size_t appearance_checks = 0;
    std::size_t ambiguity_candidates = 0;
    std::size_t ambiguity_fine_checks = 0;
    double photometric_ms = 0.0;
    double ambiguity_ms = 0.0;
    double native_ms = 0.0;
};

struct BatchResult {
    std::vector<MatchResult> matches;
    std::vector<Reason> candidates;
    std::vector<AppearanceResult> cache;
    Metrics metrics;
};

// Compact immutable GRAY8 planes remain owned by the caller during this call.
// 调用期间由调用者持有紧凑、不可变的 GRAY8 平面。
BatchResult evaluate(const std::uint8_t* previous, std::size_t previous_size,
                     const std::uint8_t* current, std::size_t current_size, int width, int height,
                     const std::vector<Pixel>& starts, const std::vector<Point>& endpoints,
                     const std::vector<Point>& candidates, const Options& options = Options{});

const char* reason_name(Reason reason);
bool neon_enabled();

} // namespace lk_appearance

#ifndef VISIONG_LK_APPEARANCE_CORE_ONLY
void bind_lk_tracker_appearance(pybind11::module_& module);
#endif

} // namespace python
} // namespace visiong
