// SPDX-License-Identifier: LGPL-3.0-or-later
#include "visiong/audio/KwsFrontend.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace visiong::audio {

namespace {

constexpr float kPi = 3.14159265358979323846f;

inline bool is_power_of_two(int value) {
    return value > 0 && (value & (value - 1)) == 0;
}

float hz_to_mel(float freq_hz) {
    return 2595.0f * std::log10(1.0f + freq_hz / 700.0f);
}

float mel_to_hz(float mel) {
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

int reverse_bits(int value, int bit_count) {
    int reversed = 0;
    for (int i = 0; i < bit_count; ++i) {
        reversed = (reversed << 1) | ((value >> i) & 1);
    }
    return reversed;
}

void multiply_window_scalar(const float* src, const float* window, float* dst, int count) {
    for (int i = 0; i < count; ++i) {
        dst[i] = src[i] * window[i];
    }
}

void multiply_window(const float* src, const float* window, float* dst, int count) {
#if defined(__ARM_NEON)
    int i = 0;
    for (; i + 4 <= count; i += 4) {
        const float32x4_t s = vld1q_f32(src + i);
        const float32x4_t w = vld1q_f32(window + i);
        vst1q_f32(dst + i, vmulq_f32(s, w));
    }
    multiply_window_scalar(src + i, window + i, dst + i, count - i);
#else
    multiply_window_scalar(src, window, dst, count);
#endif
}

}  // namespace

KwsLogMelFrontend::KwsLogMelFrontend(const KwsLogMelConfig& config) : config_(config) {
    validate_config();
    frame_length_ = config_.sample_rate * config_.window_size_ms / 1000;
    frame_step_ = config_.sample_rate * config_.window_stride_ms / 1000;
    num_frames_ = 1 + std::max(0, (config_.clip_samples - frame_length_) / frame_step_);
    spectrogram_bins_ = config_.fft_size / 2 + 1;

    initialize_window();
    initialize_mel_filters();
    initialize_fft_tables();
}

std::vector<int64_t> KwsLogMelFrontend::output_shape_hw() const {
    return {static_cast<int64_t>(num_frames_), static_cast<int64_t>(config_.num_mel_bins)};
}

std::vector<int64_t> KwsLogMelFrontend::output_shape_nchw() const {
    return {1, 1, static_cast<int64_t>(num_frames_), static_cast<int64_t>(config_.num_mel_bins)};
}

std::vector<float> KwsLogMelFrontend::compute_from_float(const float* samples, size_t count) const {
    if (count > 0 && samples == nullptr) {
        throw std::invalid_argument("KwsLogMelFrontend::compute_from_float got null samples with non-zero count.");
    }
    return compute_impl(samples, count, false);
}

std::vector<float> KwsLogMelFrontend::compute_from_pcm16(const int16_t* samples, size_t count) const {
    if (count > 0 && samples == nullptr) {
        throw std::invalid_argument("KwsLogMelFrontend::compute_from_pcm16 got null samples with non-zero count.");
    }
    return compute_impl(samples, count, true);
}

void KwsLogMelFrontend::validate_config() const {
    if (config_.sample_rate <= 0) {
        throw std::invalid_argument("KwsLogMelFrontend: sample_rate must be positive.");
    }
    if (config_.clip_samples <= 0) {
        throw std::invalid_argument("KwsLogMelFrontend: clip_samples must be positive.");
    }
    if (config_.window_size_ms <= 0 || config_.window_stride_ms <= 0) {
        throw std::invalid_argument("KwsLogMelFrontend: window size and stride must be positive.");
    }
    if (config_.num_mel_bins <= 0) {
        throw std::invalid_argument("KwsLogMelFrontend: num_mel_bins must be positive.");
    }
    if (!is_power_of_two(config_.fft_size)) {
        throw std::invalid_argument("KwsLogMelFrontend: fft_size must be a power of two.");
    }
    if (config_.lower_edge_hertz < 0.0f || config_.upper_edge_hertz <= config_.lower_edge_hertz) {
        throw std::invalid_argument("KwsLogMelFrontend: invalid mel frequency range.");
    }
}

void KwsLogMelFrontend::initialize_window() {
    window_.resize(static_cast<size_t>(frame_length_));
    if (frame_length_ == 1) {
        window_[0] = 1.0f;
        return;
    }
    for (int i = 0; i < frame_length_; ++i) {
        window_[static_cast<size_t>(i)] = 0.5f - 0.5f * std::cos((2.0f * kPi * static_cast<float>(i)) /
                                                                 static_cast<float>(frame_length_ - 1));
    }
}

void KwsLogMelFrontend::initialize_mel_filters() {
    const float mel_min = hz_to_mel(config_.lower_edge_hertz);
    const float mel_max = hz_to_mel(config_.upper_edge_hertz);
    const int edge_count = config_.num_mel_bins + 2;

    std::vector<float> mel_edges(static_cast<size_t>(edge_count));
    for (int i = 0; i < edge_count; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(edge_count - 1);
        mel_edges[static_cast<size_t>(i)] = mel_min + t * (mel_max - mel_min);
    }

    std::vector<int> fft_bins(static_cast<size_t>(edge_count));
    for (int i = 0; i < edge_count; ++i) {
        const float hz = mel_to_hz(mel_edges[static_cast<size_t>(i)]);
        int bin = static_cast<int>(std::floor((config_.fft_size + 1) * hz / static_cast<float>(config_.sample_rate)));
        bin = std::max(0, std::min(bin, spectrogram_bins_ - 1));
        fft_bins[static_cast<size_t>(i)] = bin;
    }

    mel_filters_.assign(static_cast<size_t>(config_.num_mel_bins), MelFilter{});
    for (int mel_index = 1; mel_index <= config_.num_mel_bins; ++mel_index) {
        int left = fft_bins[static_cast<size_t>(mel_index - 1)];
        int center = fft_bins[static_cast<size_t>(mel_index)];
        int right = fft_bins[static_cast<size_t>(mel_index + 1)];
        center = std::max(center, left + 1);
        right = std::max(right, center + 1);

        MelFilter filter;
        filter.start_bin = left;
        filter.weights.reserve(static_cast<size_t>(right - left));

        for (int bin = left; bin < std::min(center, spectrogram_bins_); ++bin) {
            filter.weights.push_back(static_cast<float>(bin - left) / static_cast<float>(center - left));
        }
        for (int bin = center; bin < std::min(right, spectrogram_bins_); ++bin) {
            filter.weights.push_back(static_cast<float>(right - bin) / static_cast<float>(right - center));
        }
        mel_filters_[static_cast<size_t>(mel_index - 1)] = std::move(filter);
    }
}

void KwsLogMelFrontend::initialize_fft_tables() {
    const int fft_bits = static_cast<int>(std::log2(static_cast<float>(config_.fft_size)));
    bit_reverse_indices_.resize(static_cast<size_t>(config_.fft_size));
    for (int i = 0; i < config_.fft_size; ++i) {
        bit_reverse_indices_[static_cast<size_t>(i)] = reverse_bits(i, fft_bits);
    }

    twiddle_real_.resize(static_cast<size_t>(config_.fft_size / 2));
    twiddle_imag_.resize(static_cast<size_t>(config_.fft_size / 2));
    for (int k = 0; k < config_.fft_size / 2; ++k) {
        const float angle = -2.0f * kPi * static_cast<float>(k) / static_cast<float>(config_.fft_size);
        twiddle_real_[static_cast<size_t>(k)] = std::cos(angle);
        twiddle_imag_[static_cast<size_t>(k)] = std::sin(angle);
    }
}

void KwsLogMelFrontend::run_fft_inplace(std::vector<float>& real, std::vector<float>& imag) const {
    for (int i = 0; i < config_.fft_size; ++i) {
        const int reversed = bit_reverse_indices_[static_cast<size_t>(i)];
        if (i < reversed) {
            std::swap(real[static_cast<size_t>(i)], real[static_cast<size_t>(reversed)]);
            std::swap(imag[static_cast<size_t>(i)], imag[static_cast<size_t>(reversed)]);
        }
    }

    for (int len = 2; len <= config_.fft_size; len <<= 1) {
        const int half = len >> 1;
        const int twiddle_step = config_.fft_size / len;
        for (int offset = 0; offset < config_.fft_size; offset += len) {
            int twiddle_index = 0;
            for (int j = 0; j < half; ++j) {
                const int even = offset + j;
                const int odd = even + half;
                const float wr = twiddle_real_[static_cast<size_t>(twiddle_index)];
                const float wi = twiddle_imag_[static_cast<size_t>(twiddle_index)];

                const float odd_r = real[static_cast<size_t>(odd)];
                const float odd_i = imag[static_cast<size_t>(odd)];
                const float vr = odd_r * wr - odd_i * wi;
                const float vi = odd_r * wi + odd_i * wr;

                const float ur = real[static_cast<size_t>(even)];
                const float ui = imag[static_cast<size_t>(even)];

                real[static_cast<size_t>(even)] = ur + vr;
                imag[static_cast<size_t>(even)] = ui + vi;
                real[static_cast<size_t>(odd)] = ur - vr;
                imag[static_cast<size_t>(odd)] = ui - vi;

                twiddle_index += twiddle_step;
            }
        }
    }
}

std::vector<float> KwsLogMelFrontend::compute_impl(const void* samples, size_t count, bool input_is_pcm16) const {
    std::vector<float> audio(static_cast<size_t>(config_.clip_samples), 0.0f);
    const size_t copy_count = std::min(count, static_cast<size_t>(config_.clip_samples));

    if (samples != nullptr && copy_count > 0) {
        if (!input_is_pcm16) {
            const float* src = static_cast<const float*>(samples);
            std::copy(src, src + copy_count, audio.begin());
        } else {
            const int16_t* pcm = static_cast<const int16_t*>(samples);
            for (size_t i = 0; i < copy_count; ++i) {
                audio[i] = static_cast<float>(pcm[i]) * (1.0f / 32768.0f);
            }
        }
    }

    std::vector<float> features(static_cast<size_t>(num_frames_ * config_.num_mel_bins), 0.0f);
    std::vector<float> frame_real(static_cast<size_t>(config_.fft_size), 0.0f);
    std::vector<float> frame_imag(static_cast<size_t>(config_.fft_size), 0.0f);
    std::vector<float> power(static_cast<size_t>(spectrogram_bins_), 0.0f);

    double sum = 0.0;
    double sum_sq = 0.0;

    for (int frame_index = 0; frame_index < num_frames_; ++frame_index) {
        const int start = frame_index * frame_step_;
        std::fill(frame_real.begin(), frame_real.end(), 0.0f);
        std::fill(frame_imag.begin(), frame_imag.end(), 0.0f);
        multiply_window(audio.data() + start, window_.data(), frame_real.data(), frame_length_);

        run_fft_inplace(frame_real, frame_imag);

        for (int bin = 0; bin < spectrogram_bins_; ++bin) {
            const float re = frame_real[static_cast<size_t>(bin)];
            const float im = frame_imag[static_cast<size_t>(bin)];
            power[static_cast<size_t>(bin)] = (re * re + im * im) / static_cast<float>(config_.fft_size);
        }

        for (int mel_index = 0; mel_index < config_.num_mel_bins; ++mel_index) {
            const MelFilter& filter = mel_filters_[static_cast<size_t>(mel_index)];
            float energy = 0.0f;
            for (size_t i = 0; i < filter.weights.size(); ++i) {
                const int bin = filter.start_bin + static_cast<int>(i);
                if (bin >= 0 && bin < spectrogram_bins_) {
                    energy += power[static_cast<size_t>(bin)] * filter.weights[i];
                }
            }
            // Match the training/reference pipeline: preserve low-energy differences
            // by adding epsilon before log instead of clamping everything below it.
            const float log_mel = std::log(energy + config_.epsilon);
            const size_t out_index = static_cast<size_t>(frame_index * config_.num_mel_bins + mel_index);
            features[out_index] = log_mel;
            sum += static_cast<double>(log_mel);
            sum_sq += static_cast<double>(log_mel) * static_cast<double>(log_mel);
        }
    }

    if (config_.normalize && !features.empty()) {
        const double count_f = static_cast<double>(features.size());
        const double mean = sum / count_f;
        const double variance = std::max(0.0, (sum_sq / count_f) - mean * mean);
        float stddev = static_cast<float>(std::sqrt(variance));
        if (stddev < 1e-5f || !std::isfinite(stddev)) {
            stddev = 1.0f;
        }
        for (float& value : features) {
            value = (value - static_cast<float>(mean)) / stddev;
        }
    }

    return features;
}

}  // namespace visiong::audio
