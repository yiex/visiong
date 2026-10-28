// SPDX-License-Identifier: LGPL-3.0-or-later
#include "lk_tracker_appearance.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

#ifndef VISIONG_LK_APPEARANCE_CORE_ONLY
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#endif

namespace visiong {
namespace python {
namespace lk_appearance {
namespace {

using Clock = std::chrono::steady_clock;
using Patch = std::array<std::uint8_t, 49>;

double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

struct Statistics {
    int sum = 0;
    int squares = 0;
};

// Integer statistics are exact; only the final centered energy uses double.
// 整数统计量保持精确，仅最终去均值能量使用双精度。
Statistics statistics(const std::uint8_t* first, const std::uint8_t* second, std::size_t count) {
    Statistics result;
    std::size_t index = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    int32x4_t sums = vdupq_n_s32(0);
    int32x4_t squares = vdupq_n_s32(0);
    for (; index + 8 <= count; index += 8) {
        const uint8x8_t left = vld1_u8(first + index);
        const int16x8_t value = second
                                    ? vreinterpretq_s16_u16(vsubl_u8(left, vld1_u8(second + index)))
                                    : vreinterpretq_s16_u16(vmovl_u8(left));
        const int16x4_t low = vget_low_s16(value);
        const int16x4_t high = vget_high_s16(value);
        sums = vaddq_s32(sums, vaddl_s16(low, high));
        squares = vaddq_s32(squares, vmull_s16(low, low));
        squares = vaddq_s32(squares, vmull_s16(high, high));
    }
    std::int32_t lanes[4];
    vst1q_s32(lanes, sums);
    result.sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    vst1q_s32(lanes, squares);
    result.squares = lanes[0] + lanes[1] + lanes[2] + lanes[3];
#endif
    for (; index < count; ++index) {
        const int value =
            static_cast<int>(first[index]) - (second ? static_cast<int>(second[index]) : 0);
        result.sum += value;
        result.squares += value * value;
    }
    return result;
}

double energy(const std::uint8_t* first, const std::uint8_t* second, std::size_t count) {
    const Statistics value = statistics(first, second, count);
    const double square_sum = static_cast<double>(value.sum) * value.sum;
    return std::max(0.0, value.squares - square_sum / count);
}

template <std::size_t Size> double floating_energy(const std::array<double, Size>& values) {
    double sum = 0.0;
    double squares = 0.0;
    for (double value : values) {
        sum += value;
        squares += value * value;
    }
    return std::max(0.0, squares - sum * sum / Size);
}

Patch patch(const std::uint8_t* data, int width, int x, int y) {
    Patch values{};
    std::size_t index = 0;
    for (int dy = -3; dy <= 3; ++dy) {
        const std::uint8_t* row = data + static_cast<std::size_t>(y + dy) * width + x - 3;
        for (int dx = 0; dx < 7; ++dx)
            values[index++] = row[dx];
    }
    return values;
}

std::array<std::uint8_t, 9> descriptor(const std::uint8_t* data, int width, int x, int y) {
    std::array<std::uint8_t, 9> values{};
    std::size_t index = 0;
    for (int dy = -3; dy <= 3; dy += 3) {
        for (int dx = -3; dx <= 3; dx += 3) {
            values[index++] = data[static_cast<std::size_t>(y + dy) * width + x + dx];
        }
    }
    return values;
}

bool inside(int x, int y, int width, int height, int right = 3) {
    return x >= 3 && x < width - right && y >= 3 && y < height - right;
}

Pixel rounded(const Point& point) {
    return {{static_cast<int>(std::floor(point[0] + 0.5)),
             static_cast<int>(std::floor(point[1] + 0.5))}};
}

void check_point(const Point& point) {
    for (double value : point) {
        if (!std::isfinite(value) || value < -65536.0 || value > 65536.0) {
            throw std::invalid_argument("Point coordinates must be finite and bounded");
        }
    }
}

std::vector<Pixel> offsets(int radius) {
    std::set<Pixel> unique;
    for (int distance = 3; distance <= radius; ++distance) {
        for (int sx = -1; sx <= 1; ++sx) {
            for (int sy = -1; sy <= 1; ++sy) {
                if (sx || sy)
                    unique.insert({{sx * distance, sy * distance}});
            }
        }
    }
    for (int dx = -radius; dx <= radius; dx += 4) {
        for (int dy = -radius; dy <= radius; dy += 4) {
            if (std::max(std::abs(dx), std::abs(dy)) >= 3)
                unique.insert({{dx, dy}});
        }
    }
    std::vector<Pixel> result(unique.begin(), unique.end());
    std::sort(result.begin(), result.end(), [](const Pixel& left, const Pixel& right) {
        const int ld = left[0] * left[0] + left[1] * left[1];
        const int rd = right[0] * right[0] + right[1] * right[1];
        return ld == rd ? left < right : ld < rd;
    });
    return result;
}

class Evaluator {
  public:
    Evaluator(const std::uint8_t* previous, const std::uint8_t* current, int width, int height,
              const Options& options, BatchResult& output)
        : previous_(previous), current_(current), width_(width), height_(height), options_(options),
          offsets_(offsets(options.ambiguity_radius)), output_(output) {}

    Reason appearance(const Point& point) {
        const Pixel location = rounded(point);
        const auto existing = cache_.find(location);
        if (existing != cache_.end())
            return output_.cache[existing->second].reason;
        const auto start = Clock::now();
        AppearanceResult result;
        result.point = location;
        const int x = location[0], y = location[1];
        if (!inside(x, y, width_, height_)) {
            result.reason = Reason::PatchBorder;
        } else {
            const Patch reference = patch(current_, width_, x, y);
            const double reference_energy = energy(reference.data(), nullptr, reference.size());
            if (reference_energy < 49 * 16) {
                result.reason = Reason::LowTexture;
            } else {
                const auto sparse = descriptor(current_, width_, x, y);
                const double sparse_energy = energy(sparse.data(), nullptr, sparse.size());
                const double ratio2 = options_.ambiguity_ratio * options_.ambiguity_ratio;
                int fine_checks = 0;
                for (const Pixel& offset : offsets_) {
                    const int xx = x + offset[0], yy = y + offset[1];
                    if (!inside(xx, yy, width_, height_))
                        continue;
                    ++output_.metrics.ambiguity_candidates;
                    const auto other = descriptor(current_, width_, xx, yy);
                    if (energy(sparse.data(), other.data(), sparse.size()) >
                        ratio2 * std::max(sparse_energy, 1.0))
                        continue;
                    ++output_.metrics.ambiguity_fine_checks;
                    ++fine_checks;
                    const Patch other_patch = patch(current_, width_, xx, yy);
                    const double difference =
                        energy(reference.data(), other_patch.data(), reference.size());
                    if (difference <= ratio2 * reference_energy) {
                        result.reason = Reason::Ambiguous;
                        result.has_evidence = true;
                        result.offset = offset;
                        result.relative_error = std::sqrt(difference / reference_energy);
                        break;
                    }
                    if (fine_checks == 8) {
                        result.reason = Reason::AppearanceBudget;
                        break;
                    }
                }
            }
        }
        ++output_.metrics.appearance_checks;
        output_.metrics.ambiguity_ms += milliseconds(start);
        cache_[location] = output_.cache.size();
        output_.cache.push_back(result);
        return result.reason;
    }

    MatchResult match(const Pixel& source, const Point& endpoint) {
        const auto start = Clock::now();
        MatchResult result;
        const int x = static_cast<int>(std::floor(endpoint[0]));
        const int y = static_cast<int>(std::floor(endpoint[1]));
        if (!inside(x, y, width_, height_, 4)) {
            result.photometric_reason = result.reason = Reason::PatchBorder;
            return result;
        }
        const double fx = endpoint[0] - x, fy = endpoint[1] - y;
        const Patch reference = patch(previous_, width_, source[0], source[1]);
        std::array<double, 49> target{}, difference{};
        std::size_t index = 0;
        for (int dy = -3; dy <= 3; ++dy) {
            for (int dx = -3; dx <= 3; ++dx) {
                const std::size_t at = static_cast<std::size_t>(y + dy) * width_ + x + dx;
                const double upper = current_[at] * (1 - fx) + current_[at + 1] * fx;
                const double lower =
                    current_[at + width_] * (1 - fx) + current_[at + width_ + 1] * fx;
                target[index] = upper * (1 - fy) + lower * fy;
                difference[index] = reference[index] - target[index];
                ++index;
            }
        }
        result.has_error = true;
        result.error = std::sqrt(floating_energy(difference) / 49);
        if (floating_energy(target) < 49 * 16) {
            result.photometric_reason = Reason::LowTexture;
        } else if (result.error > options_.max_patch_error) {
            result.photometric_reason = Reason::Photometric;
        }
        ++output_.metrics.photometric_checks;
        output_.metrics.photometric_ms += milliseconds(start);
        if (result.photometric_reason == Reason::None)
            result.appearance_reason = appearance(endpoint);
        result.reason = result.photometric_reason == Reason::None ? result.appearance_reason
                                                                  : result.photometric_reason;
        return result;
    }

  private:
    const std::uint8_t* previous_;
    const std::uint8_t* current_;
    int width_, height_;
    const Options& options_;
    std::vector<Pixel> offsets_;
    BatchResult& output_;
    std::map<Pixel, std::size_t> cache_;
};

} // namespace

const char* reason_name(Reason reason) {
    switch (reason) {
    case Reason::None:
        return nullptr;
    case Reason::PatchBorder:
        return "patch_border";
    case Reason::LowTexture:
        return "low_texture";
    case Reason::Photometric:
        return "photometric";
    case Reason::Ambiguous:
        return "ambiguous";
    case Reason::AppearanceBudget:
        return "appearance_budget";
    }
    throw std::logic_error("Unknown appearance reason");
}

bool neon_enabled() {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    return true;
#else
    return false;
#endif
}

BatchResult evaluate(const std::uint8_t* previous, std::size_t previous_size,
                     const std::uint8_t* current, std::size_t current_size, int width, int height,
                     const std::vector<Pixel>& starts, const std::vector<Point>& endpoints,
                     const std::vector<Point>& candidates, const Options& options) {
    const auto begin = Clock::now();
    if (width < 8 || height < 8 || width > 65535 || height > 65535) {
        throw std::invalid_argument("GRAY8 dimensions must be in [8,65535]");
    }
    const std::size_t size = static_cast<std::size_t>(width) * height;
    if (!current || current_size != size ||
        (!starts.empty() && (!previous || previous_size != size)) ||
        (previous_size != 0 && previous_size != size)) {
        throw std::invalid_argument("Expected compact GRAY8 bytes with matching dimensions");
    }
    if (starts.size() != endpoints.size() || starts.size() > 500 || candidates.size() > 500) {
        throw std::invalid_argument(
            "Expected equal match lists and at most 500 matches or candidates");
    }
    if (!std::isfinite(options.max_patch_error) || options.max_patch_error < 0 ||
        !std::isfinite(options.ambiguity_ratio) || options.ambiguity_ratio < 0 ||
        options.ambiguity_ratio > 1 || options.ambiguity_radius < 3 ||
        options.ambiguity_radius > 32) {
        throw std::invalid_argument("Invalid appearance thresholds or radius");
    }
    for (const Pixel& point : starts) {
        if (!inside(point[0], point[1], width, height)) {
            throw std::invalid_argument("Source patches must lie within the previous frame");
        }
    }
    for (const Point& point : endpoints)
        check_point(point);
    for (const Point& point : candidates)
        check_point(point);
    BatchResult result;
    result.matches.reserve(starts.size());
    result.candidates.reserve(candidates.size());
    result.cache.reserve(starts.size() + candidates.size());
    Evaluator evaluator(previous, current, width, height, options, result);
    for (std::size_t i = 0; i < starts.size(); ++i) {
        result.matches.push_back(evaluator.match(starts[i], endpoints[i]));
    }
    for (const Point& point : candidates)
        result.candidates.push_back(evaluator.appearance(point));
    result.metrics.native_ms = milliseconds(begin);
    return result;
}

} // namespace lk_appearance

#ifndef VISIONG_LK_APPEARANCE_CORE_ONLY
namespace {

namespace py = pybind11;

py::object reason_object(lk_appearance::Reason reason) {
    const char* name = lk_appearance::reason_name(reason);
    return name ? py::object(py::str(name)) : py::object(py::none());
}

} // namespace

void bind_lk_tracker_appearance(py::module_& module) {
    module.def(
        "_lk_tracker_appearance",
        [](py::bytes previous, py::bytes current, int width, int height,
           const std::vector<lk_appearance::Pixel>& starts,
           const std::vector<lk_appearance::Point>& endpoints,
           const std::vector<lk_appearance::Point>& candidates, double max_patch_error,
           double ambiguity_ratio, int ambiguity_radius) {
            char* previous_data = nullptr;
            char* current_data = nullptr;
            Py_ssize_t previous_size = 0, current_size = 0;
            if (PyBytes_AsStringAndSize(previous.ptr(), &previous_data, &previous_size) != 0 ||
                PyBytes_AsStringAndSize(current.ptr(), &current_data, &current_size) != 0) {
                throw py::error_already_set();
            }
            lk_appearance::Options options;
            options.max_patch_error = max_patch_error;
            options.ambiguity_ratio = ambiguity_ratio;
            options.ambiguity_radius = ambiguity_radius;
            lk_appearance::BatchResult result;
            {
                // Keep immutable bytes alive while the numerical work releases the
                // GIL. 数值计算释放 GIL，但保留不可变 bytes 的生命周期。
                py::gil_scoped_release release;
                result = lk_appearance::evaluate(
                    reinterpret_cast<const std::uint8_t*>(previous_data), previous_size,
                    reinterpret_cast<const std::uint8_t*>(current_data), current_size, width,
                    height, starts, endpoints, candidates, options);
            }
            py::list matches, candidate_results, cache, examples;
            for (const auto& value : result.matches) {
                py::dict row;
                row["error"] =
                    value.has_error ? py::object(py::float_(value.error)) : py::object(py::none());
                row["reason"] = reason_object(value.reason);
                row["photometric_reason"] = reason_object(value.photometric_reason);
                row["appearance_reason"] = reason_object(value.appearance_reason);
                matches.append(row);
            }
            for (auto value : result.candidates)
                candidate_results.append(reason_object(value));
            for (const auto& value : result.cache) {
                py::dict row;
                row["point"] = py::cast(value.point);
                row["reason"] = reason_object(value.reason);
                cache.append(row);
                if (value.has_evidence && py::len(examples) < 8) {
                    py::dict evidence;
                    evidence["point"] = py::cast(value.point);
                    evidence["offset"] = py::cast(value.offset);
                    evidence["relative_error"] = value.relative_error;
                    examples.append(evidence);
                }
            }
            const auto& value = result.metrics;
            py::dict checks;
            checks["photometric_checks"] = value.photometric_checks;
            checks["appearance_checks"] = value.appearance_checks;
            checks["ambiguity_candidates"] = value.ambiguity_candidates;
            checks["ambiguity_fine_checks"] = value.ambiguity_fine_checks;
            checks["photometric_ms"] = value.photometric_ms;
            checks["ambiguity_ms"] = value.ambiguity_ms;
            checks["native_ms"] = value.native_ms;
            checks["ambiguity_examples"] = examples;
            checks["backend"] = lk_appearance::neon_enabled() ? "native_neon" : "native_scalar";
            py::dict output;
            output["matches"] = matches;
            output["candidates"] = candidate_results;
            output["cache"] = cache;
            output["checks"] = checks;
            return output;
        },
        py::arg("previous"), py::arg("current"), py::arg("width"), py::arg("height"),
        py::arg("starts"), py::arg("endpoints"), py::arg("candidates"),
        py::arg("max_patch_error") = 12.0, py::arg("ambiguity_ratio") = 0.2,
        py::arg("ambiguity_radius") = 16,
        "Batch zero-mean patch and bounded ambiguity checks. / "
        "批量去均值块与有界歧义检查。");
}
#endif

} // namespace python
} // namespace visiong
