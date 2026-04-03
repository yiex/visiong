// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_NPU_KWS_H
#define VISIONG_NPU_KWS_H

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "visiong/audio/KwsFrontend.h"
#include "visiong/npu/LowLevelNPU.h"

struct KWSResult {
    int class_id = -1;
    std::string label;
    float score = 0.0f;
    std::vector<float> scores;
};

class KWS {
public:
    KWS(const std::string& model_path,
        const std::string& labels_path = "",
        int sample_rate = 16000,
        int clip_samples = 16000,
        int window_size_ms = 30,
        int window_stride_ms = 20,
        int fft_size = 512,
        int num_mel_bins = 40,
        float lower_edge_hertz = 20.0f,
        float upper_edge_hertz = 4000.0f,
        float epsilon = 1e-6f,
        bool normalize = true,
        uint32_t init_flags = 0);
    ~KWS();

    KWS(const KWS&) = delete;
    KWS& operator=(const KWS&) = delete;
    KWS(KWS&&) = delete;
    KWS& operator=(KWS&&) = delete;

    KWSResult infer_pcm16(const int16_t* samples, size_t count);
    KWSResult infer_float(const float* samples, size_t count);
    KWSResult infer_pcm16(const std::vector<int16_t>& samples) {
        return infer_pcm16(samples.data(), samples.size());
    }
    KWSResult infer_float(const std::vector<float>& samples) {
        return infer_float(samples.data(), samples.size());
    }
    KWSResult infer_wav(const std::string& wav_path);

    bool is_initialized() const;
    int sample_rate() const;
    int clip_samples() const;
    int num_frames() const;
    int num_mel_bins() const;
    int num_classes() const;
    std::vector<std::string> labels() const;
    int64_t last_run_us() const;

private:
    KWSResult infer_impl(const void* samples, size_t count, bool input_is_pcm16);
    void validate_contract();

    visiong::audio::KwsLogMelFrontend m_frontend;
    LowLevelNPU m_npu;
    std::vector<std::string> m_labels;
    int m_num_classes = 0;
    mutable std::mutex m_mutex;
};

#endif  // VISIONG_NPU_KWS_H
