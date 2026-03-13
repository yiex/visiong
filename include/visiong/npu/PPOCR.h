// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_NPU_PPOCR_H
#define VISIONG_NPU_PPOCR_H

#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

struct rknn_app_context_t;

class ImageBuffer;

struct OCRResult {
    std::vector<std::tuple<int, int>> quad;
    std::tuple<int, int, int, int> rect;
    float det_score = 0.0f;
    std::string text;
    float text_score = 0.0f;
};

class PPOCR {
public:
    PPOCR(const std::string& det_model_path,
          const std::string& rec_model_path,
          const std::string& dict_path = "",
          float det_threshold = 0.3f,
          float box_threshold = 0.5f,
          bool use_dilate = true,
          const std::string& rec_fast_model_path = "",
          float rec_fast_max_ratio = 2.4f,
          bool rec_fast_enable_fallback = true,
          float rec_fast_fallback_score_thresh = 0.2f,
          const std::string& model_input_format = "rgb",
          float det_unclip_ratio = 1.6f);
    ~PPOCR();

    PPOCR(const PPOCR&) = delete;
    PPOCR& operator=(const PPOCR&) = delete;
    PPOCR(PPOCR&&) = delete;
    PPOCR& operator=(PPOCR&&) = delete;

    std::vector<OCRResult> infer(const ImageBuffer& image);
    bool is_initialized() const;

    int det_model_width() const;
    int det_model_height() const;
    int rec_model_width() const;
    int rec_model_height() const;

private:
    int initialize_runtime();
    void release_runtime();
    int load_dictionary(const std::string& dict_path);

    std::unique_ptr<rknn_app_context_t> m_det_ctx;
    std::unique_ptr<rknn_app_context_t> m_rec_ctx;
    std::unique_ptr<rknn_app_context_t> m_rec_fast_ctx;

    std::string m_det_model_path;
    std::string m_rec_model_path;
    std::string m_rec_fast_model_path;
    std::string m_dict_path;
    std::string m_model_input_format = "rgb";
    float m_rec_fast_max_ratio = 2.4f;
    bool m_rec_fast_enable_fallback = true;
    float m_rec_fast_fallback_score_thresh = 0.2f;

    float m_det_threshold = 0.3f;
    float m_box_threshold = 0.5f;
    float m_det_unclip_ratio = 1.6f;
    bool m_use_dilate = true;
    bool m_initialized = false;

    std::vector<std::string> m_dict;
    mutable std::mutex m_runtime_mutex;
};

#endif  // VISIONG_NPU_PPOCR_H

