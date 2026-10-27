// SPDX-License-Identifier: LGPL-3.0-or-later
// Benchmark real outputs against an independent float/full-sort reference.
// 使用真实 RKNN 输出，对照独立的浮点全排序参考实现测试。
#include "npu/internal/models/yolo26.h"
#include "npu/internal/rknn_model_utils.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

template <class F> double timed(F f, int loops) {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < loops; ++i) {
        if (f() != 0) {
            throw std::runtime_error("benchmark call failed");
        }
    }
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / loops;
}
int main(int argc, char** argv) {
    if (argc < 3) {
        printf("usage: bench model.rknn input_rgb.bin [loops]\n");
        return 2;
    }
    rknn_app_context_t app{};
    if (init_yolo26_model(argv[1], &app) != 0) {
        return 1;
    }
    auto* ctx = create_yolo26_post_process_ctx(&app, nullptr, .25f);
    if (!ctx) {
        return 3;
    }
    FILE* file = fopen(argv[2], "rb");
    if (!file) {
        return 4;
    }
    if (fread(app.input_mems[0]->virt_addr, 1, app.input_attrs[0].size, file) != app.input_attrs[0].size) {
        return 5;
    }
    fclose(file);
    dma_sync_cpu_to_device(app.input_mems[0]->fd);
    auto hash = [](const void* ptr, size_t bytes) {
        const auto* p = static_cast<const unsigned char*>(ptr);
        uint32_t h = 2166136261u;
        for (size_t i = 0; i < bytes; ++i) {
            h ^= p[i];
            h *= 16777619u;
        }
        return h;
    };
    const auto input_hash = hash(app.input_mems[0]->virt_addr, app.input_attrs[0].size);
    std::vector<uint32_t> output_hashes;
    for (int i = 0; i < 20; ++i) {
        if (visiong::npu::rknn::run_and_sync_outputs(&app) != 0) {
            return 6;
        }
        output_hashes.push_back(hash(app.output_mems[0]->virt_addr, app.output_attrs[0].size));
    }
    std::sort(output_hashes.begin(), output_hashes.end());
    auto unique = std::unique(output_hashes.begin(), output_hashes.end()) - output_hashes.begin();
    printf("RAW_STABILITY input_unchanged=%s unique_output_hashes=%td runs=20\n",
           input_hash == hash(app.input_mems[0]->virt_addr, app.input_attrs[0].size) ? "true" : "false", unique);
    for (int i = 0; i < 5; ++i) {
        if (visiong::npu::rknn::run_and_sync_outputs(&app) != 0) {
            return 6;
        }
    }
    const int loops = argc > 3 ? std::atoi(argv[3]) : 100;
    double run = timed([&] { return visiong::npu::rknn::run_and_sync_outputs(&app); }, loops);
    object_detect_result_list result{};
    // The converter emits rank-3 UNDEFINED [1,C,N] outputs. / 转换脚本导出三维 UNDEFINED [1,C,N] 输出。
    const auto& sa = app.output_attrs[0];
    const auto& ba = app.output_attrs[1];
    if (sa.n_dims != 3 || sa.dims[2] != ba.dims[2] || ba.dims[1] != 4) {
        return 7;
    }
    int n = sa.dims[2], classes = sa.dims[1];
    const auto* scores = static_cast<int8_t*>(app.output_mems[0]->virt_addr);
    const auto* boxes = static_cast<int8_t*>(app.output_mems[1]->virt_addr);
    struct Ref {
        float score;
        int index, cls;
    };
    std::vector<Ref> refs;
    auto reference = [&] {
        refs.clear();
        for (int i = 0; i < n; ++i) {
            float max_score = -1;
            int cls = -1;
            for (int c = 0; c < classes; ++c) {
                const float score = (int(scores[c * n + i]) - sa.zp) * sa.scale;
                if (score > max_score) {
                    max_score = score;
                    cls = c;
                }
            }
            if (max_score > .25f) {
                refs.push_back({max_score, i, cls});
            }
        }
        std::sort(refs.begin(), refs.end(), [](const Ref& a, const Ref& b) {
            return a.score != b.score ? a.score > b.score : a.index < b.index;
        });
        if (refs.size() > OBJ_NUMB_MAX_SIZE) {
            refs.resize(OBJ_NUMB_MAX_SIZE);
        }
        return 0;
    };
    auto post = [&] { return post_process_yolo26(&app, ctx, 1.f, 0, 0, &result); };
    double reference_ms = timed(reference, loops);
    double post_ms = timed(post, loops);
    if (result.count != int(refs.size())) {
        throw std::runtime_error("real output count mismatch");
    }
    for (int k = 0; k < result.count; ++k) {
        const auto& r = refs[k];
        const auto& d = result.results[k];
        int xyxy[4];
        for (int c = 0; c < 4; ++c) {
            xyxy[c] = int((int(boxes[c * n + r.index]) - ba.zp) * ba.scale);
        }
        if (d.cls_id != r.cls || d.prop != r.score || d.box.left != xyxy[0] || d.box.top != xyxy[1] ||
            d.box.right != xyxy[2] || d.box.bottom != xyxy[3]) {
            throw std::runtime_error("real output mismatch");
        }
    }
    printf("NATIVE_RESULT run_sync_ms=%.6f reference_select_ms=%.6f optimized_full_post_ms=%.6f detections=%d loops=%d "
           "reference_match=true\n",
           run, reference_ms, post_ms, result.count, loops);
    destroy_yolo26_post_process_ctx(ctx);
    release_yolo26_model(&app);
}
