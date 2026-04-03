// SPDX-License-Identifier: LGPL-3.0-or-later
#include "visiong/npu/KWS.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

visiong::audio::KwsLogMelConfig build_kws_config(int sample_rate,
                                                 int clip_samples,
                                                 int window_size_ms,
                                                 int window_stride_ms,
                                                 int fft_size,
                                                 int num_mel_bins,
                                                 float lower_edge_hertz,
                                                 float upper_edge_hertz,
                                                 float epsilon,
                                                 bool normalize) {
    visiong::audio::KwsLogMelConfig config;
    config.sample_rate = sample_rate;
    config.clip_samples = clip_samples;
    config.window_size_ms = window_size_ms;
    config.window_stride_ms = window_stride_ms;
    config.fft_size = fft_size;
    config.num_mel_bins = num_mel_bins;
    config.lower_edge_hertz = lower_edge_hertz;
    config.upper_edge_hertz = upper_edge_hertz;
    config.epsilon = epsilon;
    config.normalize = normalize;
    return config;
}

std::string trim_line(std::string line) {
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
        line.pop_back();
    }
    return line;
}

std::vector<std::string> load_label_file(const std::string& labels_path) {
    std::ifstream in(labels_path);
    if (!in.is_open()) {
        throw std::runtime_error("KWS: failed to open labels file: " + labels_path);
    }

    std::vector<std::string> labels;
    std::string line;
    while (std::getline(in, line)) {
        line = trim_line(std::move(line));
        if (!line.empty()) {
            labels.push_back(line);
        }
    }
    return labels;
}

std::vector<std::string> build_default_labels(int num_classes) {
    std::vector<std::string> labels;
    labels.reserve(static_cast<size_t>(std::max(0, num_classes)));
    for (int i = 0; i < num_classes; ++i) {
        labels.push_back(std::to_string(i));
    }
    return labels;
}

uint16_t read_le_u16(std::istream& in) {
    unsigned char bytes[2] = {0, 0};
    in.read(reinterpret_cast<char*>(bytes), sizeof(bytes));
    if (!in) {
        throw std::runtime_error("KWS: truncated WAV file while reading uint16.");
    }
    return static_cast<uint16_t>(bytes[0]) |
           static_cast<uint16_t>(static_cast<uint16_t>(bytes[1]) << 8U);
}

uint32_t read_le_u32(std::istream& in) {
    unsigned char bytes[4] = {0, 0, 0, 0};
    in.read(reinterpret_cast<char*>(bytes), sizeof(bytes));
    if (!in) {
        throw std::runtime_error("KWS: truncated WAV file while reading uint32.");
    }
    return static_cast<uint32_t>(bytes[0]) |
           (static_cast<uint32_t>(bytes[1]) << 8U) |
           (static_cast<uint32_t>(bytes[2]) << 16U) |
           (static_cast<uint32_t>(bytes[3]) << 24U);
}

struct WavPcm16Clip {
    int sample_rate = 0;
    std::vector<int16_t> mono_samples;
};

WavPcm16Clip load_wav_pcm16_mono(const std::string& wav_path) {
    std::ifstream in(wav_path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("KWS: failed to open WAV file: " + wav_path);
    }

    char riff_id[4] = {};
    in.read(riff_id, sizeof(riff_id));
    if (!in || std::string(riff_id, sizeof(riff_id)) != "RIFF") {
        throw std::runtime_error("KWS: WAV file must start with RIFF: " + wav_path);
    }
    (void)read_le_u32(in);
    char wave_id[4] = {};
    in.read(wave_id, sizeof(wave_id));
    if (!in || std::string(wave_id, sizeof(wave_id)) != "WAVE") {
        throw std::runtime_error("KWS: unsupported WAV container (expected WAVE): " + wav_path);
    }

    bool found_fmt = false;
    bool found_data = false;
    uint16_t audio_format = 0;
    uint16_t num_channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits_per_sample = 0;
    uint16_t block_align = 0;
    std::vector<int16_t> interleaved_samples;

    while (in && (!found_fmt || !found_data)) {
        char chunk_id[4] = {};
        in.read(chunk_id, sizeof(chunk_id));
        if (!in) {
            break;
        }
        const uint32_t chunk_size = read_le_u32(in);
        const std::string chunk_name(chunk_id, sizeof(chunk_id));

        if (chunk_name == "fmt ") {
            audio_format = read_le_u16(in);
            num_channels = read_le_u16(in);
            sample_rate = read_le_u32(in);
            (void)read_le_u32(in);
            block_align = read_le_u16(in);
            bits_per_sample = read_le_u16(in);
            const uint32_t remaining = (chunk_size >= 16U) ? (chunk_size - 16U) : 0U;
            if (remaining > 0) {
                in.seekg(static_cast<std::streamoff>(remaining), std::ios::cur);
            }
            if (!in) {
                throw std::runtime_error("KWS: malformed WAV fmt chunk: " + wav_path);
            }
            found_fmt = true;
        } else if (chunk_name == "data") {
            if (!found_fmt) {
                throw std::runtime_error("KWS: WAV data chunk appears before fmt chunk: " + wav_path);
            }
            if (block_align == 0) {
                throw std::runtime_error("KWS: invalid WAV block_align=0: " + wav_path);
            }
            if (chunk_size % static_cast<uint32_t>(block_align) != 0U) {
                throw std::runtime_error("KWS: WAV data chunk is not aligned to block size: " + wav_path);
            }
            std::vector<char> raw(chunk_size);
            in.read(raw.data(), static_cast<std::streamsize>(raw.size()));
            if (!in) {
                throw std::runtime_error("KWS: truncated WAV data chunk: " + wav_path);
            }
            const size_t sample_count = raw.size() / sizeof(int16_t);
            interleaved_samples.resize(sample_count);
            std::memcpy(interleaved_samples.data(), raw.data(), raw.size());
            found_data = true;
        } else {
            in.seekg(static_cast<std::streamoff>(chunk_size), std::ios::cur);
            if (!in) {
                throw std::runtime_error("KWS: malformed WAV chunk layout: " + wav_path);
            }
        }

        if ((chunk_size & 1U) != 0U) {
            in.seekg(1, std::ios::cur);
            if (!in) {
                throw std::runtime_error("KWS: malformed WAV padding byte: " + wav_path);
            }
        }
    }

    if (!found_fmt || !found_data) {
        throw std::runtime_error("KWS: missing fmt or data chunk in WAV file: " + wav_path);
    }
    if (audio_format != 1U) {
        throw std::runtime_error("KWS: only PCM16 WAV files are supported: " + wav_path);
    }
    if (bits_per_sample != 16U) {
        throw std::runtime_error("KWS: only 16-bit PCM WAV files are supported: " + wav_path);
    }
    if (num_channels == 0U) {
        throw std::runtime_error("KWS: WAV file reports zero channels: " + wav_path);
    }

    WavPcm16Clip clip;
    clip.sample_rate = static_cast<int>(sample_rate);
    const size_t frame_count = interleaved_samples.size() / static_cast<size_t>(num_channels);
    clip.mono_samples.resize(frame_count);

    if (num_channels == 1U) {
        clip.mono_samples = std::move(interleaved_samples);
        return clip;
    }

    for (size_t frame = 0; frame < frame_count; ++frame) {
        int32_t sum = 0;
        const size_t base = frame * static_cast<size_t>(num_channels);
        for (uint16_t ch = 0; ch < num_channels; ++ch) {
            sum += static_cast<int32_t>(interleaved_samples[base + ch]);
        }
        const int32_t avg = sum / static_cast<int32_t>(num_channels);
        clip.mono_samples[frame] = static_cast<int16_t>(
            std::max<int32_t>(std::numeric_limits<int16_t>::min(),
                              std::min<int32_t>(std::numeric_limits<int16_t>::max(), avg)));
    }
    return clip;
}

void softmax_inplace(std::vector<float>* values) {
    if (values == nullptr || values->empty()) {
        throw std::runtime_error("KWS: model returned no output scores.");
    }

    const float max_value = *std::max_element(values->begin(), values->end());
    if (!std::isfinite(max_value)) {
        throw std::runtime_error("KWS: model output contains non-finite values.");
    }

    double sum = 0.0;
    for (float& value : *values) {
        value = static_cast<float>(std::exp(static_cast<double>(value - max_value)));
        sum += static_cast<double>(value);
    }
    if (!(sum > 0.0) || !std::isfinite(sum)) {
        throw std::runtime_error("KWS: softmax normalization failed.");
    }

    const float inv_sum = static_cast<float>(1.0 / sum);
    for (float& value : *values) {
        value *= inv_sum;
    }
}

size_t argmax_index(const std::vector<float>& values) {
    return static_cast<size_t>(std::distance(values.begin(),
                                             std::max_element(values.begin(), values.end())));
}

}  // namespace

KWS::KWS(const std::string& model_path,
         const std::string& labels_path,
         int sample_rate,
         int clip_samples,
         int window_size_ms,
         int window_stride_ms,
         int fft_size,
         int num_mel_bins,
         float lower_edge_hertz,
         float upper_edge_hertz,
         float epsilon,
         bool normalize,
         uint32_t init_flags)
    : m_frontend(build_kws_config(sample_rate,
                                  clip_samples,
                                  window_size_ms,
                                  window_stride_ms,
                                  fft_size,
                                  num_mel_bins,
                                  lower_edge_hertz,
                                  upper_edge_hertz,
                                  epsilon,
                                  normalize)),
      m_npu(model_path, init_flags) {
    validate_contract();

    if (labels_path.empty()) {
        m_labels = build_default_labels(m_num_classes);
    } else {
        m_labels = load_label_file(labels_path);
        if (static_cast<int>(m_labels.size()) != m_num_classes) {
            throw std::runtime_error("KWS: labels file count (" + std::to_string(m_labels.size()) +
                                     ") does not match model classes (" +
                                     std::to_string(m_num_classes) + ").");
        }
    }
}

KWS::~KWS() = default;

void KWS::validate_contract() {
    if (!m_npu.is_initialized()) {
        throw std::runtime_error("KWS: LowLevelNPU failed to initialize.");
    }
    if (m_npu.num_inputs() != 1 || m_npu.num_outputs() != 1) {
        throw std::runtime_error("KWS: expected a model with exactly 1 input and 1 output.");
    }

    const LowLevelTensorInfo input_info = m_npu.input_tensor(0);
    const LowLevelTensorInfo output_info = m_npu.output_tensor(0);
    const size_t expected_features =
        static_cast<size_t>(m_frontend.num_frames()) * static_cast<size_t>(m_frontend.num_mel_bins());

    if (input_info.num_elements != expected_features) {
        throw std::runtime_error("KWS: model input element count (" +
                                 std::to_string(input_info.num_elements) +
                                 ") does not match frontend feature count (" +
                                 std::to_string(expected_features) + ").");
    }
    if (output_info.num_elements == 0U) {
        throw std::runtime_error("KWS: model output element count must be positive.");
    }

    m_num_classes = static_cast<int>(output_info.num_elements);
}

KWSResult KWS::infer_pcm16(const int16_t* samples, size_t count) {
    if (count > 0 && samples == nullptr) {
        throw std::invalid_argument("KWS: infer_pcm16 got null samples with non-zero count.");
    }
    return infer_impl(samples, count, true);
}

KWSResult KWS::infer_float(const float* samples, size_t count) {
    if (count > 0 && samples == nullptr) {
        throw std::invalid_argument("KWS: infer_float got null samples with non-zero count.");
    }
    return infer_impl(samples, count, false);
}

KWSResult KWS::infer_wav(const std::string& wav_path) {
    if (wav_path.empty()) {
        throw std::invalid_argument("KWS: wav_path must not be empty.");
    }

    const WavPcm16Clip clip = load_wav_pcm16_mono(wav_path);
    if (clip.sample_rate != sample_rate()) {
        throw std::runtime_error("KWS: WAV sample rate " + std::to_string(clip.sample_rate) +
                                 " does not match frontend sample rate " +
                                 std::to_string(sample_rate()) + ".");
    }
    return infer_pcm16(clip.mono_samples);
}

KWSResult KWS::infer_impl(const void* samples, size_t count, bool input_is_pcm16) {
    std::lock_guard<std::mutex> lock(m_mutex);

    std::vector<float> features = input_is_pcm16
                                      ? m_frontend.compute_from_pcm16(static_cast<const int16_t*>(samples), count)
                                      : m_frontend.compute_from_float(static_cast<const float*>(samples), count);
    if (features.empty()) {
        throw std::runtime_error("KWS: frontend returned no features.");
    }

    m_npu.set_input_from_float(0, features.data(), features.size(), true, true, true);
    m_npu.run(true, false, 0);

    std::vector<float> scores = m_npu.output_float(0, true, false);
    if (static_cast<int>(scores.size()) != m_num_classes) {
        throw std::runtime_error("KWS: model output count (" + std::to_string(scores.size()) +
                                 ") does not match expected classes (" +
                                 std::to_string(m_num_classes) + ").");
    }

    softmax_inplace(&scores);
    const size_t best_index = argmax_index(scores);

    KWSResult result;
    result.class_id = static_cast<int>(best_index);
    result.label = (best_index < m_labels.size()) ? m_labels[best_index] : std::to_string(best_index);
    result.score = scores[best_index];
    result.scores = std::move(scores);
    return result;
}

bool KWS::is_initialized() const {
    return m_npu.is_initialized();
}

int KWS::sample_rate() const {
    return m_frontend.sample_rate();
}

int KWS::clip_samples() const {
    return m_frontend.clip_samples();
}

int KWS::num_frames() const {
    return m_frontend.num_frames();
}

int KWS::num_mel_bins() const {
    return m_frontend.num_mel_bins();
}

int KWS::num_classes() const {
    return m_num_classes;
}

std::vector<std::string> KWS::labels() const {
    return m_labels;
}

int64_t KWS::last_run_us() const {
    return m_npu.last_run_us();
}
