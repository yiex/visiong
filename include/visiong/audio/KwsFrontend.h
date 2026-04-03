// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace visiong::audio {

struct KwsLogMelConfig {
    int sample_rate = 16000;
    int clip_samples = 16000;
    int window_size_ms = 30;
    int window_stride_ms = 20;
    int fft_size = 512;
    int num_mel_bins = 40;
    float lower_edge_hertz = 20.0f;
    float upper_edge_hertz = 4000.0f;
    float epsilon = 1e-6f;
    bool normalize = true;
};

class KwsLogMelFrontend {
public:
    explicit KwsLogMelFrontend(const KwsLogMelConfig& config = KwsLogMelConfig());

    const KwsLogMelConfig& config() const { return config_; }

    int sample_rate() const { return config_.sample_rate; }
    int clip_samples() const { return config_.clip_samples; }
    int frame_length() const { return frame_length_; }
    int frame_step() const { return frame_step_; }
    int fft_size() const { return config_.fft_size; }
    int num_frames() const { return num_frames_; }
    int num_mel_bins() const { return config_.num_mel_bins; }

    std::vector<int64_t> output_shape_hw() const;
    std::vector<int64_t> output_shape_nchw() const;

    std::vector<float> compute_from_float(const float* samples, size_t count) const;
    std::vector<float> compute_from_pcm16(const int16_t* samples, size_t count) const;

private:
    struct MelFilter {
        int start_bin = 0;
        std::vector<float> weights;
    };

    KwsLogMelConfig config_;
    int frame_length_ = 0;
    int frame_step_ = 0;
    int num_frames_ = 0;
    int spectrogram_bins_ = 0;
    std::vector<float> window_;
    std::vector<MelFilter> mel_filters_;
    std::vector<int> bit_reverse_indices_;
    std::vector<float> twiddle_real_;
    std::vector<float> twiddle_imag_;

    void validate_config() const;
    void initialize_window();
    void initialize_mel_filters();
    void initialize_fft_tables();
    void run_fft_inplace(std::vector<float>& real, std::vector<float>& imag) const;
    std::vector<float> compute_impl(const void* samples, size_t count, bool input_is_pcm16) const;
};

}  // namespace visiong::audio
