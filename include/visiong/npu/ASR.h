// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_NPU_ASR_H
#define VISIONG_NPU_ASR_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "visiong/npu/LowLevelNPU.h"

struct ASRResult {
    std::string text;
    std::string pinyin;
    std::vector<std::string> pinyin_tokens;
    int acoustic_frames = 0;
    int used_tokens = 0;
    int64_t acoustic_run_us = 0;
    int64_t p2c_run_us = 0;
    int64_t rerank_run_us = 0;
    int64_t total_run_us = 0;
};

class ASR {
public:
    ASR(const std::string& acoustic_model_path,
        const std::string& acoustic_vocab_path,
        const std::string& p2c_model_path,
        const std::string& p2c_input_vocab_path,
        const std::string& p2c_output_vocab_path,
        int feature_frames = 600,
        int feature_bins = 80,
        int max_tokens = 40,
        int segment_topk = 6,
        float candidate_temperature = 1.0f,
        const std::string& char_lm_path = "",
        float char_lm_scale = 0.0f,
        int char_beam_size = 6,
        int char_topk = 3,
        uint32_t acoustic_init_flags = 0,
        uint32_t p2c_init_flags = 0);
    ~ASR();

    ASR(const ASR&) = delete;
    ASR& operator=(const ASR&) = delete;
    ASR(ASR&&) = delete;
    ASR& operator=(ASR&&) = delete;

    ASRResult infer_features(const float* features, size_t frame_count, size_t feature_bins);
    ASRResult infer_features(const std::vector<float>& features, size_t frame_count, size_t feature_bins) {
        return infer_features(features.data(), frame_count, feature_bins);
    }

    bool is_initialized() const;
    int feature_frames() const;
    int feature_bins() const;
    int max_tokens() const;
    int segment_topk() const;
    float candidate_temperature() const;
    bool has_char_lm() const;
    float char_lm_scale() const;
    int char_beam_size() const;
    int char_topk() const;
    std::vector<std::string> acoustic_vocab() const;
    std::vector<std::string> p2c_input_vocab() const;
    std::vector<std::string> p2c_output_vocab() const;
    int64_t last_acoustic_run_us() const;
    int64_t last_p2c_run_us() const;
    int64_t last_rerank_run_us() const;
    int64_t last_total_run_us() const;

private:
    struct CharNgramLm;

    void validate_contract();
    ASRResult infer_impl(const float* features, size_t frame_count, size_t feature_bins);

    std::vector<std::string> m_acoustic_vocab;
    std::vector<std::string> m_p2c_input_vocab;
    std::vector<std::string> m_p2c_output_vocab;
    std::unordered_map<std::string, int> m_p2c_input_token_to_id;
    int m_feature_frames = 0;
    int m_feature_bins = 0;
    int m_max_tokens = 0;
    int m_segment_topk = 0;
    float m_candidate_temperature = 1.0f;
    float m_char_lm_scale = 0.0f;
    int m_char_beam_size = 0;
    int m_char_topk = 0;
    int m_acoustic_output_frames = 0;
    int m_acoustic_blank_id = 0;
    int m_p2c_pad_id = 0;
    int m_p2c_unk_id = 1;
    bool m_acoustic_prefers_quantized_input = false;
    bool m_p2c_prefers_quantized_input = false;
    LowLevelTensorInfo m_acoustic_input_info;
    LowLevelTensorInfo m_acoustic_output_info;
    LowLevelTensorInfo m_p2c_input_info;
    LowLevelTensorInfo m_p2c_output_info;
    mutable std::mutex m_mutex;
    LowLevelNPU m_acoustic_npu;
    LowLevelNPU m_p2c_npu;
    std::vector<float> m_feature_buffer;
    std::vector<uint8_t> m_acoustic_input_bytes;
    std::vector<uint8_t> m_p2c_input_bytes;
    std::vector<float> m_p2c_input_float;
    std::vector<float> m_segment_average_logits;
    std::shared_ptr<CharNgramLm> m_char_lm;
    int64_t m_last_rerank_run_us = 0;
    int64_t m_last_total_run_us = 0;
};

#endif  // VISIONG_NPU_ASR_H
