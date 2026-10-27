// SPDX-License-Identifier: LGPL-3.0-or-later
// Compare with an independent float/full-sort reference. / 与独立的浮点全排序参考实现比较。
#include "npu/internal/yolo26_decode.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <stdexcept>
using namespace visiong::npu::yolo26;

static std::vector<Candidate> reference(const Scores& s, float threshold, size_t limit) {
    std::vector<Candidate> out;
    for (int i = 0; i < s.count; ++i) {
        float best = -1e30f;
        int8_t best_q = -128;
        for (int c = 0; c < s.classes; ++c) {
            const int8_t q = s.data[size_t(i) * s.anchor_stride + size_t(c) * s.class_stride];
            float value = (int(q) - s.zero_point) * s.scale;
            if (value > best) {
                best = value;
                best_q = q;
            }
        }
        if (best > threshold) {
            out.push_back({i, best_q});
        }
    }
    std::sort(out.begin(), out.end(), better);
    if (out.size() > limit) {
        out.resize(limit);
    }
    return out;
}
static void check(const Scores& s, float threshold, size_t limit) {
    auto expected = reference(s, threshold, limit);
    std::vector<Candidate> actual;
    std::vector<int8_t> maxima;
    for (bool neon : {false, true}) {
        select(s, threshold, limit, actual, maxima, neon);
        if (actual.size() != expected.size()) {
            throw std::runtime_error("count mismatch");
        }
        for (size_t k = 0; k < actual.size(); ++k) {
            if (actual[k].index != expected[k].index || actual[k].score != expected[k].score) {
                throw std::runtime_error("selection mismatch");
            }
            int cls = class_id(s, actual[k]);
            if (cls < 0 || cls >= s.classes ||
                s.data[size_t(actual[k].index) * s.anchor_stride + size_t(cls) * s.class_stride] != actual[k].score) {
                throw std::runtime_error("class mismatch");
            }
        }
    }
}
template <class F> static double time_ms(F f, int loops) {
    auto t = std::chrono::steady_clock::now();
    for (int i = 0; i < loops; ++i) {
        f();
    }
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count() / loops;
}
int main(int argc, char** argv) {
    std::mt19937 gen(26);
    int checks = 0;
    for (int n : {1, 17, 129, 2100, 8400}) {
        for (int c : {1, 4, 17, 80, 81}) {
            for (bool planar : {false, true}) {
                const int padded = planar ? n + 3 : c + 3;
                std::vector<int8_t> data(size_t(padded) * (planar ? c : n), -128);
                Scores s{
                    data.data(), n, c, planar ? size_t(1) : size_t(padded), planar ? size_t(padded) : size_t(1), -128,
                    1.f / 255.f};
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < c; ++j) {
                        data[size_t(i) * s.anchor_stride + size_t(j) * s.class_stride] = int8_t(int(gen() % 256) - 128);
                    }
                }
                for (float threshold : {0.f, 0.25f, 0.5f, 1.f}) {
                    for (size_t limit : {size_t(1), size_t(128), size_t(256)}) {
                        check(s, threshold, limit);
                        ++checks;
                    }
                }
            }
        }
    }
    for (float scale : {1.f / 255.f, 0.00321f, 0.1f}) {
        for (int q = -128; q <= 127; ++q) {
            int8_t data[] = {int8_t(q)};
            Scores s{data, 1, 1, 1, 1, -128, scale};
            float threshold = (q + 128) * scale;
            for (float t : {threshold, std::nextafter(threshold, -INFINITY), std::nextafter(threshold, INFINITY)}) {
                check(s, t, 128);
                ++checks;
            }
        }
    }
    // Invalid quantization must not leave old candidates behind. / 无效量化参数不得残留上次候选结果。
    for (float scale : {0.f, -1.f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        const int8_t data[] = {127};
        Scores scores{data, 1, 1, 1, 1, -128, scale};
        std::vector<Candidate> candidates{{0, 127}};
        std::vector<int8_t> maxima;
        select(scores, .25f, 128, candidates, maxima);
        if (!candidates.empty()) {
            throw std::runtime_error("invalid quantization accepted");
        }
        ++checks;
    }
    printf("PASS %d reference and validation cases (scalar and NEON)\n", checks);
    int loops = argc > 1 ? std::atoi(argv[1]) : 30;
    volatile size_t sink = 0;
    for (bool planar : {false, true}) {
        const int n = 8400, c = 80;
        std::vector<int8_t> data(size_t(n) * c, -128);
        Scores s{data.data(), n, c, planar ? size_t(1) : size_t(c), planar ? size_t(n) : size_t(1), -128, 1.f / 255.f};
        // Sparse one2one predictions and low-score background. / 模拟稀疏的 one2one 目标和低分背景。
        for (auto& v : data) {
            v = int8_t(-128 + int(gen() % 50));
        }
        for (int i = 0; i < 200; ++i) {
            data[size_t(i * 41) * s.anchor_stride + size_t(i % c) * s.class_stride] = int8_t(gen() % 128);
        }
        std::vector<Candidate> candidates;
        candidates.reserve(n);
        std::vector<int8_t> maxima;
        maxima.reserve(n);
        auto ref = time_ms([&] { sink += reference(s, .25f, 128).size(); }, loops);
        auto scalar = time_ms(
            [&] {
                select(s, .25f, 128, candidates, maxima, false);
                sink += candidates.size();
            },
            loops);
        auto neon = time_ms(
            [&] {
                select(s, .25f, 128, candidates, maxima, true);
                sink += candidates.size();
            },
            loops);
        printf("BENCH %s reference_ms=%.6f optimized_scalar_ms=%.6f neon_ms=%.6f loops=%d\n", planar ? "NCHW" : "NHWC",
               ref, scalar, neon, loops);
    }
    return sink == 0;
}
