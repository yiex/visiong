// SPDX-License-Identifier: LGPL-3.0-or-later
#include "visiong/npu/ASR.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace {

constexpr char kPadToken[] = "<pad>";
constexpr char kUnkToken[] = "<unk>";
constexpr char kBlankToken[] = "<blank>";
constexpr char kBlkToken[] = "<blk>";

std::string trim_line(std::string line) {
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
        line.pop_back();
    }
    return line;
}

std::vector<std::string> load_vocab_file(const std::string& path, const std::string& name) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("ASR: failed to open " + name + " file: " + path);
    }
    std::vector<std::string> values;
    std::string line;
    while (std::getline(in, line)) {
        line = trim_line(std::move(line));
        if (!line.empty()) {
            values.push_back(line);
        }
    }
    if (values.empty()) {
        throw std::runtime_error("ASR: " + name + " file is empty: " + path);
    }
    return values;
}

std::unordered_map<std::string, int> build_token_to_id(const std::vector<std::string>& vocab) {
    std::unordered_map<std::string, int> token_to_id;
    token_to_id.reserve(vocab.size());
    for (size_t i = 0; i < vocab.size(); ++i) {
        token_to_id.emplace(vocab[i], static_cast<int>(i));
    }
    return token_to_id;
}

std::string join_tokens(const std::vector<std::string>& tokens) {
    if (tokens.empty()) {
        return "";
    }
    std::string out = tokens.front();
    for (size_t i = 1; i < tokens.size(); ++i) {
        out.push_back(' ');
        out += tokens[i];
    }
    return out;
}

std::vector<std::string> split_tab_fields(const std::string& line) {
    std::vector<std::string> fields;
    size_t start = 0;
    while (start <= line.size()) {
        const size_t pos = line.find('\t', start);
        if (pos == std::string::npos) {
            fields.emplace_back(line.substr(start));
            break;
        }
        fields.emplace_back(line.substr(start, pos - start));
        start = pos + 1;
    }
    return fields;
}

std::string join_key(const std::vector<std::string>& tokens, size_t start, size_t count) {
    if (count == 0U) {
        return "";
    }
    size_t total = 0;
    for (size_t i = 0; i < count; ++i) {
        total += tokens[start + i].size();
    }
    total += (count - 1U);
    std::string key;
    key.reserve(total);
    key += tokens[start];
    for (size_t i = 1; i < count; ++i) {
        key.push_back('\t');
        key += tokens[start + i];
    }
    return key;
}

bool is_byte_quantized_tensor(const LowLevelTensorInfo& info) {
    return (info.type == "INT8" || info.type == "UINT8") && info.quant_type == "AFFINE";
}

size_t expected_num_elements(int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument("ASR: rows and cols must be positive.");
    }
    return static_cast<size_t>(rows) * static_cast<size_t>(cols);
}

template <typename T>
T clamp_round_cast(double raw) {
    const double min_value = static_cast<double>(std::numeric_limits<T>::lowest());
    const double max_value = static_cast<double>(std::numeric_limits<T>::max());
    if (raw < min_value) {
        raw = min_value;
    } else if (raw > max_value) {
        raw = max_value;
    }
    return static_cast<T>(std::llround(raw));
}

uint8_t quantize_prob_to_byte(float value, const LowLevelTensorInfo& info) {
    const double scale = (info.scale == 0.0f) ? 1.0 : static_cast<double>(info.scale);
    const double raw = static_cast<double>(value) / scale + static_cast<double>(info.zero_point);
    if (info.type == "INT8") {
        return static_cast<uint8_t>(clamp_round_cast<int8_t>(raw));
    }
    return clamp_round_cast<uint8_t>(raw);
}

float stable_softmax_prob(float value, float max_value, float inv_sum, float temperature) {
    const double adjusted = static_cast<double>(value - max_value) / static_cast<double>(temperature);
    return static_cast<float>(std::exp(adjusted) * static_cast<double>(inv_sum));
}

void insert_topk_candidate(std::vector<std::pair<float, int>>* topk, float value, int index, int limit) {
    auto& values = *topk;
    for (const auto& item : values) {
        if (item.second == index) {
            return;
        }
    }
    auto it = values.begin();
    while (it != values.end() && it->first >= value) {
        ++it;
    }
    values.insert(it, std::make_pair(value, index));
    if (static_cast<int>(values.size()) > limit) {
        values.resize(static_cast<size_t>(limit));
    }
}

std::vector<std::pair<float, int>> select_topk_logits(const float* row, int count, int topk, int ensure_index) {
    std::vector<std::pair<float, int>> best;
    best.reserve(static_cast<size_t>(topk + 1));
    for (int i = 0; i < count; ++i) {
        insert_topk_candidate(&best, row[i], i, topk);
    }
    if (ensure_index >= 0 && ensure_index < count) {
        insert_topk_candidate(&best, row[ensure_index], ensure_index, topk);
    }
    return best;
}

int find_token_id_or_default(const std::vector<std::string>& vocab,
                             const std::string& token,
                             int fallback_value) {
    const auto it = std::find(vocab.begin(), vocab.end(), token);
    if (it == vocab.end()) {
        return fallback_value;
    }
    return static_cast<int>(std::distance(vocab.begin(), it));
}

void accumulate_logits_inplace(float* dst, const float* src, size_t count) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        float32x4_t d0 = vld1q_f32(dst + i);
        float32x4_t d1 = vld1q_f32(dst + i + 4);
        float32x4_t d2 = vld1q_f32(dst + i + 8);
        float32x4_t d3 = vld1q_f32(dst + i + 12);
        const float32x4_t s0 = vld1q_f32(src + i);
        const float32x4_t s1 = vld1q_f32(src + i + 4);
        const float32x4_t s2 = vld1q_f32(src + i + 8);
        const float32x4_t s3 = vld1q_f32(src + i + 12);
        vst1q_f32(dst + i, vaddq_f32(d0, s0));
        vst1q_f32(dst + i + 4, vaddq_f32(d1, s1));
        vst1q_f32(dst + i + 8, vaddq_f32(d2, s2));
        vst1q_f32(dst + i + 12, vaddq_f32(d3, s3));
    }
    for (; i + 4 <= count; i += 4) {
        const float32x4_t d = vld1q_f32(dst + i);
        const float32x4_t s = vld1q_f32(src + i);
        vst1q_f32(dst + i, vaddq_f32(d, s));
    }
    for (; i < count; ++i) {
        dst[i] += src[i];
    }
#else
    for (size_t i = 0; i < count; ++i) {
        dst[i] += src[i];
    }
#endif
}

}  // namespace

struct ASR::CharNgramLm {
    int order = 0;
    float backoff_alpha = 0.4f;
    float log_backoff = std::log(0.4f);
    float unknown_log_prob = -20.0f;
    std::unordered_map<std::string, float> unigram_log_probs;
    std::unordered_map<int, std::unordered_map<std::string, float>> higher_order_log_probs;

    static std::shared_ptr<CharNgramLm> load_from_file(const std::string& path) {
        std::ifstream in(path);
        if (!in.is_open()) {
            throw std::runtime_error("ASR: failed to open char LM file: " + path);
        }

        std::string line;
        if (!std::getline(in, line)) {
            throw std::runtime_error("ASR: char LM file is empty: " + path);
        }
        line = trim_line(std::move(line));
        if (line != "VISIONG_CHAR_NGRAM_LM_V1") {
            throw std::runtime_error("ASR: unsupported char LM format in file: " + path);
        }

        auto lm = std::make_shared<CharNgramLm>();
        while (std::getline(in, line)) {
            line = trim_line(std::move(line));
            if (line.empty() || line[0] == '#') {
                continue;
            }
            const std::vector<std::string> fields = split_tab_fields(line);
            if (fields.empty()) {
                continue;
            }

            if (fields[0] == "O") {
                if (fields.size() != 2U) {
                    throw std::runtime_error("ASR: invalid char LM order record.");
                }
                lm->order = std::stoi(fields[1]);
                continue;
            }
            if (fields[0] == "A") {
                if (fields.size() != 2U) {
                    throw std::runtime_error("ASR: invalid char LM alpha record.");
                }
                lm->backoff_alpha = std::stof(fields[1]);
                continue;
            }
            if (fields[0] == "K") {
                if (fields.size() != 2U) {
                    throw std::runtime_error("ASR: invalid char LM unknown record.");
                }
                lm->unknown_log_prob = std::stof(fields[1]);
                continue;
            }
            if (fields[0] == "U") {
                if (fields.size() != 3U) {
                    throw std::runtime_error("ASR: invalid char LM unigram record.");
                }
                lm->unigram_log_probs.emplace(fields[1], std::stof(fields[2]));
                continue;
            }
            if (fields[0] == "N") {
                if (fields.size() < 4U) {
                    throw std::runtime_error("ASR: invalid char LM ngram record.");
                }
                const int ngram_order = std::stoi(fields[1]);
                if (ngram_order < 2) {
                    throw std::runtime_error("ASR: char LM ngram order must be >= 2.");
                }
                if (fields.size() != static_cast<size_t>(ngram_order) + 3U) {
                    throw std::runtime_error("ASR: char LM ngram token count does not match record order.");
                }
                std::vector<std::string> tokens;
                tokens.reserve(static_cast<size_t>(ngram_order));
                for (int i = 0; i < ngram_order; ++i) {
                    tokens.emplace_back(fields[static_cast<size_t>(i) + 2U]);
                }
                lm->higher_order_log_probs[ngram_order].emplace(
                    join_key(tokens, 0, tokens.size()),
                    std::stof(fields.back()));
                continue;
            }

            throw std::runtime_error("ASR: unknown char LM record type: " + fields[0]);
        }

        if (lm->order < 1) {
            throw std::runtime_error("ASR: char LM order must be >= 1.");
        }
        if (!(lm->backoff_alpha > 0.0f)) {
            throw std::runtime_error("ASR: char LM backoff_alpha must be positive.");
        }
        lm->log_backoff = std::log(std::max(lm->backoff_alpha, 1e-6f));
        if (lm->unigram_log_probs.empty()) {
            throw std::runtime_error("ASR: char LM unigram table must not be empty.");
        }
        return lm;
    }

    float score_tokens(const std::vector<std::string>& tokens) const {
        if (tokens.empty()) {
            return 0.0f;
        }

        std::vector<std::string> seq;
        seq.reserve(static_cast<size_t>(order - 1) + tokens.size() + 1U);
        for (int i = 0; i < order - 1; ++i) {
            seq.emplace_back("<s>");
        }
        seq.insert(seq.end(), tokens.begin(), tokens.end());
        seq.emplace_back("</s>");

        float total = 0.0f;
        for (size_t index = static_cast<size_t>(std::max(order - 1, 0)); index < seq.size(); ++index) {
            bool matched = false;
            for (int ngram_order = order; ngram_order >= 2; --ngram_order) {
                if (index + 1U < static_cast<size_t>(ngram_order)) {
                    continue;
                }
                const size_t start = index + 1U - static_cast<size_t>(ngram_order);
                const std::string key = join_key(seq, start, static_cast<size_t>(ngram_order));
                const auto order_it = higher_order_log_probs.find(ngram_order);
                if (order_it == higher_order_log_probs.end()) {
                    total += log_backoff;
                    continue;
                }
                const auto prob_it = order_it->second.find(key);
                if (prob_it != order_it->second.end()) {
                    total += prob_it->second;
                    matched = true;
                    break;
                }
                total += log_backoff;
            }
            if (matched) {
                continue;
            }
            const auto unigram_it = unigram_log_probs.find(seq[index]);
            total += (unigram_it != unigram_log_probs.end()) ? unigram_it->second : unknown_log_prob;
        }
        return total;
    }
};

ASR::ASR(const std::string& acoustic_model_path,
         const std::string& acoustic_vocab_path,
         const std::string& p2c_model_path,
         const std::string& p2c_input_vocab_path,
         const std::string& p2c_output_vocab_path,
         int feature_frames,
         int feature_bins,
         int max_tokens,
         int segment_topk,
         float candidate_temperature,
         const std::string& char_lm_path,
         float char_lm_scale,
         int char_beam_size,
         int char_topk,
         uint32_t acoustic_init_flags,
         uint32_t p2c_init_flags)
    : m_acoustic_vocab(load_vocab_file(acoustic_vocab_path, "acoustic vocab")),
      m_p2c_input_vocab(load_vocab_file(p2c_input_vocab_path, "P2C input vocab")),
      m_p2c_output_vocab(load_vocab_file(p2c_output_vocab_path, "P2C output vocab")),
      m_p2c_input_token_to_id(build_token_to_id(m_p2c_input_vocab)),
      m_feature_frames(feature_frames),
      m_feature_bins(feature_bins),
      m_max_tokens(max_tokens),
      m_segment_topk(segment_topk),
      m_candidate_temperature(candidate_temperature),
      m_char_lm_scale(char_lm_scale),
      m_char_beam_size(char_beam_size),
      m_char_topk(char_topk),
      m_acoustic_npu(acoustic_model_path, acoustic_init_flags),
      m_p2c_npu(p2c_model_path, p2c_init_flags) {
    if (!char_lm_path.empty()) {
        m_char_lm = CharNgramLm::load_from_file(char_lm_path);
    }
    validate_contract();
    m_feature_buffer.assign(expected_num_elements(m_feature_frames, m_feature_bins), 0.0f);
    if (m_acoustic_prefers_quantized_input) {
        m_acoustic_input_bytes.assign(m_acoustic_input_info.num_elements,
                                      quantize_prob_to_byte(0.0f, m_acoustic_input_info));
    }
    m_segment_average_logits.assign(m_acoustic_vocab.size(), 0.0f);
    if (m_p2c_prefers_quantized_input) {
        m_p2c_input_bytes.assign(m_p2c_input_info.num_elements, quantize_prob_to_byte(0.0f, m_p2c_input_info));
    } else {
        m_p2c_input_float.assign(m_p2c_input_info.num_elements, 0.0f);
    }
}

ASR::~ASR() = default;

void ASR::validate_contract() {
    if (!m_acoustic_npu.is_initialized()) {
        throw std::runtime_error("ASR: acoustic LowLevelNPU failed to initialize.");
    }
    if (!m_p2c_npu.is_initialized()) {
        throw std::runtime_error("ASR: P2C LowLevelNPU failed to initialize.");
    }
    if (m_acoustic_npu.num_inputs() != 1 || m_acoustic_npu.num_outputs() != 1) {
        throw std::runtime_error("ASR: acoustic model must have exactly 1 input and 1 output.");
    }
    if (m_p2c_npu.num_inputs() != 1 || m_p2c_npu.num_outputs() != 1) {
        throw std::runtime_error("ASR: P2C model must have exactly 1 input and 1 output.");
    }

    if (m_feature_frames <= 0 || m_feature_bins <= 0) {
        throw std::invalid_argument("ASR: feature_frames and feature_bins must be positive.");
    }
    if (m_max_tokens <= 0) {
        throw std::invalid_argument("ASR: max_tokens must be positive.");
    }
    if (m_segment_topk <= 0) {
        throw std::invalid_argument("ASR: segment_topk must be positive.");
    }
    if (!(m_candidate_temperature > 0.0f)) {
        throw std::invalid_argument("ASR: candidate_temperature must be positive.");
    }
    if (m_char_lm_scale < 0.0f) {
        throw std::invalid_argument("ASR: char_lm_scale must be non-negative.");
    }
    if (m_char_lm && m_char_beam_size <= 0) {
        throw std::invalid_argument("ASR: char_beam_size must be positive when char LM is enabled.");
    }
    if (m_char_lm && m_char_topk <= 0) {
        throw std::invalid_argument("ASR: char_topk must be positive when char LM is enabled.");
    }
    if (!m_char_lm && m_char_lm_scale > 0.0f) {
        throw std::invalid_argument("ASR: char_lm_scale > 0 requires a valid char LM resource.");
    }

    m_acoustic_input_info = m_acoustic_npu.input_tensor(0);
    m_acoustic_output_info = m_acoustic_npu.output_tensor(0);
    m_p2c_input_info = m_p2c_npu.input_tensor(0);
    m_p2c_output_info = m_p2c_npu.output_tensor(0);

    const size_t acoustic_input_expected = expected_num_elements(m_feature_frames, m_feature_bins);
    if (m_acoustic_input_info.num_elements != acoustic_input_expected) {
        throw std::runtime_error("ASR: acoustic input element count (" +
                                 std::to_string(m_acoustic_input_info.num_elements) +
                                 ") does not match feature contract (" +
                                 std::to_string(acoustic_input_expected) + ").");
    }
    if (m_acoustic_vocab.empty()) {
        throw std::runtime_error("ASR: acoustic vocab must not be empty.");
    }
    if (m_acoustic_output_info.num_elements == 0U ||
        (m_acoustic_output_info.num_elements % static_cast<uint32_t>(m_acoustic_vocab.size())) != 0U) {
        throw std::runtime_error("ASR: acoustic output size is not compatible with vocab size.");
    }
    m_acoustic_output_frames =
        static_cast<int>(m_acoustic_output_info.num_elements / static_cast<uint32_t>(m_acoustic_vocab.size()));
    if (m_acoustic_output_frames <= 0) {
        throw std::runtime_error("ASR: acoustic output must contain at least one frame.");
    }
    m_acoustic_blank_id = find_token_id_or_default(m_acoustic_vocab, kBlankToken, 0);
    if (m_acoustic_blank_id == 0) {
        m_acoustic_blank_id = find_token_id_or_default(m_acoustic_vocab, kBlkToken, m_acoustic_blank_id);
    }

    const size_t p2c_input_expected = expected_num_elements(m_max_tokens, static_cast<int>(m_p2c_input_vocab.size()));
    if (m_p2c_input_info.num_elements != p2c_input_expected) {
        throw std::runtime_error("ASR: P2C input element count (" +
                                 std::to_string(m_p2c_input_info.num_elements) +
                                 ") does not match configured max_tokens/input vocab (" +
                                 std::to_string(p2c_input_expected) + ").");
    }
    const size_t p2c_output_expected = expected_num_elements(m_max_tokens, static_cast<int>(m_p2c_output_vocab.size()));
    if (m_p2c_output_info.num_elements != p2c_output_expected) {
        throw std::runtime_error("ASR: P2C output element count (" +
                                 std::to_string(m_p2c_output_info.num_elements) +
                                 ") does not match configured max_tokens/output vocab (" +
                                 std::to_string(p2c_output_expected) + ").");
    }

    const auto pad_it = m_p2c_input_token_to_id.find(kPadToken);
    if (pad_it == m_p2c_input_token_to_id.end()) {
        throw std::runtime_error("ASR: P2C input vocab must contain <pad>.");
    }
    m_p2c_pad_id = pad_it->second;
    const auto unk_it = m_p2c_input_token_to_id.find(kUnkToken);
    if (unk_it != m_p2c_input_token_to_id.end()) {
        m_p2c_unk_id = unk_it->second;
    }

    m_acoustic_prefers_quantized_input = is_byte_quantized_tensor(m_acoustic_input_info);
    m_p2c_prefers_quantized_input = is_byte_quantized_tensor(m_p2c_input_info);
}

ASRResult ASR::infer_features(const float* features, size_t frame_count, size_t feature_bins) {
    if (frame_count > 0 && features == nullptr) {
        throw std::invalid_argument("ASR: infer_features got null features with non-zero frame_count.");
    }
    return infer_impl(features, frame_count, feature_bins);
}

ASRResult ASR::infer_impl(const float* features, size_t frame_count, size_t feature_bins) {
    std::lock_guard<std::mutex> lock(m_mutex);

    if (feature_bins != static_cast<size_t>(m_feature_bins)) {
        throw std::invalid_argument("ASR: feature_bins does not match configured value " +
                                    std::to_string(m_feature_bins) + ".");
    }

    const auto total_start = std::chrono::steady_clock::now();
    std::fill(m_feature_buffer.begin(), m_feature_buffer.end(), 0.0f);
    const size_t rows = std::min(frame_count, static_cast<size_t>(m_feature_frames));
    for (size_t row = 0; row < rows; ++row) {
        const float* src = features + row * feature_bins;
        float* dst = m_feature_buffer.data() + row * static_cast<size_t>(m_feature_bins);
        std::copy(src, src + feature_bins, dst);
    }

    if (m_acoustic_prefers_quantized_input) {
        for (size_t i = 0; i < m_feature_buffer.size(); ++i) {
            m_acoustic_input_bytes[i] = quantize_prob_to_byte(m_feature_buffer[i], m_acoustic_input_info);
        }
        m_acoustic_npu.set_input_buffer(0, m_acoustic_input_bytes.data(), m_acoustic_input_bytes.size(), false, true);
    } else {
        m_acoustic_npu.set_input_from_float(0, m_feature_buffer.data(), m_feature_buffer.size(), true, true, true);
    }
    m_acoustic_npu.run(true, false, 0);
    std::vector<float> acoustic_logits = m_acoustic_npu.output_float(0, true, false);
    const size_t acoustic_vocab_size = m_acoustic_vocab.size();
    const int valid_frames = std::min<int>(static_cast<int>(rows), m_acoustic_output_frames);

    if (acoustic_logits.size() != static_cast<size_t>(m_acoustic_output_frames) * acoustic_vocab_size) {
        throw std::runtime_error("ASR: acoustic logits size does not match expected output shape.");
    }

    if (m_p2c_prefers_quantized_input) {
        const uint8_t zero_value = quantize_prob_to_byte(0.0f, m_p2c_input_info);
        std::fill(m_p2c_input_bytes.begin(), m_p2c_input_bytes.end(), zero_value);
    } else {
        std::fill(m_p2c_input_float.begin(), m_p2c_input_float.end(), 0.0f);
    }

    std::vector<std::string> tokens;
    tokens.reserve(static_cast<size_t>(std::min(m_max_tokens, valid_frames)));

    int used_tokens = 0;
    int active_token_id = -1;
    int segment_start = 0;
    auto flush_segment = [&](int token_id, int start_frame, int end_frame) {
        if (token_id < 0 || token_id >= static_cast<int>(acoustic_vocab_size) || used_tokens >= m_max_tokens) {
            return;
        }

        tokens.push_back(m_acoustic_vocab[static_cast<size_t>(token_id)]);
        const float* logits_base = acoustic_logits.data();
        std::fill(m_segment_average_logits.begin(), m_segment_average_logits.end(), 0.0f);
        const int segment_length = std::max(1, end_frame - start_frame);
        for (int frame = start_frame; frame < end_frame; ++frame) {
            const float* row = logits_base + static_cast<size_t>(frame) * acoustic_vocab_size;
            accumulate_logits_inplace(m_segment_average_logits.data(), row, acoustic_vocab_size);
        }
        const float inv_length = 1.0f / static_cast<float>(segment_length);
        for (float& value : m_segment_average_logits) {
            value *= inv_length;
        }

        const std::vector<std::pair<float, int>> topk =
            select_topk_logits(m_segment_average_logits.data(),
                               static_cast<int>(acoustic_vocab_size),
                               m_segment_topk,
                               token_id);
        if (topk.empty()) {
            return;
        }

        float max_logit = topk.front().first;
        for (const auto& item : topk) {
            max_logit = std::max(max_logit, item.first);
        }
        double sum = 0.0;
        for (const auto& item : topk) {
            sum += std::exp(static_cast<double>(item.first - max_logit) /
                            static_cast<double>(m_candidate_temperature));
        }
        if (!(sum > 0.0) || !std::isfinite(sum)) {
            throw std::runtime_error("ASR: segment softmax normalization failed.");
        }
        const float inv_sum = static_cast<float>(1.0 / sum);
        const size_t row_offset = static_cast<size_t>(used_tokens) * m_p2c_input_vocab.size();

        for (const auto& item : topk) {
            const std::string& token = m_acoustic_vocab[static_cast<size_t>(item.second)];
            const auto token_it = m_p2c_input_token_to_id.find(token);
            const int p2c_id = (token_it != m_p2c_input_token_to_id.end()) ? token_it->second : m_p2c_unk_id;
            const float prob = stable_softmax_prob(item.first, max_logit, inv_sum, m_candidate_temperature);
            if (m_p2c_prefers_quantized_input) {
                m_p2c_input_bytes[row_offset + static_cast<size_t>(p2c_id)] =
                    quantize_prob_to_byte(prob, m_p2c_input_info);
            } else {
                m_p2c_input_float[row_offset + static_cast<size_t>(p2c_id)] = prob;
            }
        }
        ++used_tokens;
    };

    for (int frame = 0; frame < valid_frames; ++frame) {
        const float* row = acoustic_logits.data() + static_cast<size_t>(frame) * acoustic_vocab_size;
        int token_id = 0;
        float best_value = row[0];
        for (size_t col = 1; col < acoustic_vocab_size; ++col) {
            if (row[col] > best_value) {
                best_value = row[col];
                token_id = static_cast<int>(col);
            }
        }
        if (token_id == m_acoustic_blank_id) {
            if (active_token_id >= 0) {
                flush_segment(active_token_id, segment_start, frame);
                active_token_id = -1;
            }
            continue;
        }
        if (active_token_id < 0) {
            active_token_id = token_id;
            segment_start = frame;
            continue;
        }
        if (token_id != active_token_id) {
            flush_segment(active_token_id, segment_start, frame);
            active_token_id = token_id;
            segment_start = frame;
        }
    }
    if (active_token_id >= 0) {
        flush_segment(active_token_id, segment_start, valid_frames);
    }

    if (m_p2c_prefers_quantized_input) {
        const uint8_t pad_byte = quantize_prob_to_byte(1.0f, m_p2c_input_info);
        for (int row = used_tokens; row < m_max_tokens; ++row) {
            const size_t offset = static_cast<size_t>(row) * m_p2c_input_vocab.size() + static_cast<size_t>(m_p2c_pad_id);
            m_p2c_input_bytes[offset] = pad_byte;
        }
        m_p2c_npu.set_input_buffer(0, m_p2c_input_bytes.data(), m_p2c_input_bytes.size(), false, true);
    } else {
        for (int row = used_tokens; row < m_max_tokens; ++row) {
            const size_t offset = static_cast<size_t>(row) * m_p2c_input_vocab.size() + static_cast<size_t>(m_p2c_pad_id);
            m_p2c_input_float[offset] = 1.0f;
        }
        m_p2c_npu.set_input_from_float(0, m_p2c_input_float.data(), m_p2c_input_float.size(), true, false, true);
    }
    m_p2c_npu.run(true, false, 0);
    std::vector<float> p2c_logits = m_p2c_npu.output_float(0, true, false);
    const size_t output_vocab_size = m_p2c_output_vocab.size();
    if (p2c_logits.size() != static_cast<size_t>(m_max_tokens) * output_vocab_size) {
        throw std::runtime_error("ASR: P2C logits size does not match expected output shape.");
    }

    std::string text;
    text.reserve(static_cast<size_t>(used_tokens));
    m_last_rerank_run_us = 0;
    if (m_char_lm && m_char_lm_scale > 0.0f && used_tokens > 0) {
        const auto rerank_start = std::chrono::steady_clock::now();

        struct CharBeam {
            std::vector<std::string> tokens;
            float acoustic_score = 0.0f;
        };

        std::vector<CharBeam> beams(1);
        std::vector<CharBeam> next_beams;
        next_beams.reserve(static_cast<size_t>(std::max(m_char_beam_size * m_char_topk, 1)));

        for (int row = 0; row < used_tokens; ++row) {
            const float* row_logits = p2c_logits.data() + static_cast<size_t>(row) * output_vocab_size;
            float max_logit = row_logits[0];
            for (size_t col = 1; col < output_vocab_size; ++col) {
                max_logit = std::max(max_logit, row_logits[col]);
            }
            double denom = 0.0;
            for (size_t col = 0; col < output_vocab_size; ++col) {
                denom += std::exp(static_cast<double>(row_logits[col] - max_logit));
            }
            if (!(denom > 0.0) || !std::isfinite(denom)) {
                throw std::runtime_error("ASR: P2C log-softmax normalization failed.");
            }
            const float log_denom = max_logit + static_cast<float>(std::log(denom));

            std::vector<std::pair<float, int>> topk =
                select_topk_logits(row_logits, static_cast<int>(output_vocab_size), m_char_topk + 2, -1);
            std::vector<std::pair<float, int>> valid_candidates;
            valid_candidates.reserve(static_cast<size_t>(m_char_topk));
            for (const auto& item : topk) {
                const std::string& token = m_p2c_output_vocab[static_cast<size_t>(item.second)];
                if (token == kPadToken || token == kUnkToken) {
                    continue;
                }
                valid_candidates.push_back(item);
                if (static_cast<int>(valid_candidates.size()) >= m_char_topk) {
                    break;
                }
            }
            if (valid_candidates.empty()) {
                int best_valid_index = -1;
                float best_valid_logit = -std::numeric_limits<float>::infinity();
                for (size_t col = 0; col < output_vocab_size; ++col) {
                    const std::string& token = m_p2c_output_vocab[col];
                    if (token == kPadToken || token == kUnkToken) {
                        continue;
                    }
                    if (row_logits[col] > best_valid_logit) {
                        best_valid_logit = row_logits[col];
                        best_valid_index = static_cast<int>(col);
                    }
                }
                if (best_valid_index >= 0) {
                    valid_candidates.emplace_back(best_valid_logit, best_valid_index);
                }
            }
            if (valid_candidates.empty()) {
                continue;
            }

            next_beams.clear();
            for (const CharBeam& beam : beams) {
                for (const auto& item : valid_candidates) {
                    CharBeam candidate = beam;
                    candidate.tokens.push_back(m_p2c_output_vocab[static_cast<size_t>(item.second)]);
                    candidate.acoustic_score += (item.first - log_denom);

                    bool merged = false;
                    for (CharBeam& existing : next_beams) {
                        if (existing.tokens == candidate.tokens) {
                            if (candidate.acoustic_score > existing.acoustic_score) {
                                existing.acoustic_score = candidate.acoustic_score;
                            }
                            merged = true;
                            break;
                        }
                    }
                    if (!merged) {
                        next_beams.push_back(std::move(candidate));
                    }
                }
            }
            std::sort(next_beams.begin(), next_beams.end(), [](const CharBeam& lhs, const CharBeam& rhs) {
                return lhs.acoustic_score > rhs.acoustic_score;
            });
            if (static_cast<int>(next_beams.size()) > m_char_beam_size) {
                next_beams.resize(static_cast<size_t>(m_char_beam_size));
            }
            beams.swap(next_beams);
            if (beams.empty()) {
                break;
            }
        }

        float best_score = -std::numeric_limits<float>::infinity();
        for (const CharBeam& beam : beams) {
            const float char_count = static_cast<float>(std::max<size_t>(beam.tokens.size(), 1U));
            const float lm_score = m_char_lm->score_tokens(beam.tokens);
            const float combined_score =
                (beam.acoustic_score / char_count) + m_char_lm_scale * (lm_score / char_count);
            if (combined_score > best_score) {
                best_score = combined_score;
                text.clear();
                for (const std::string& token : beam.tokens) {
                    text += token;
                }
            }
        }

        const auto rerank_end = std::chrono::steady_clock::now();
        m_last_rerank_run_us =
            std::chrono::duration_cast<std::chrono::microseconds>(rerank_end - rerank_start).count();
    } else {
        for (int row = 0; row < used_tokens; ++row) {
            const float* row_logits = p2c_logits.data() + static_cast<size_t>(row) * output_vocab_size;
            size_t best_index = 0;
            float best_value = row_logits[0];
            for (size_t col = 1; col < output_vocab_size; ++col) {
                if (row_logits[col] > best_value) {
                    best_value = row_logits[col];
                    best_index = col;
                }
            }
            const std::string& token = m_p2c_output_vocab[best_index];
            if (token != kPadToken && token != kUnkToken) {
                text += token;
            }
        }
    }

    const auto total_end = std::chrono::steady_clock::now();
    m_last_total_run_us =
        std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start).count();

    ASRResult result;
    result.text = std::move(text);
    result.pinyin = join_tokens(tokens);
    result.pinyin_tokens = std::move(tokens);
    result.acoustic_frames = valid_frames;
    result.used_tokens = used_tokens;
    result.acoustic_run_us = m_acoustic_npu.last_run_us();
    result.p2c_run_us = m_p2c_npu.last_run_us();
    result.rerank_run_us = m_last_rerank_run_us;
    result.total_run_us = m_last_total_run_us;
    return result;
}

bool ASR::is_initialized() const {
    return m_acoustic_npu.is_initialized() && m_p2c_npu.is_initialized();
}

int ASR::feature_frames() const {
    return m_feature_frames;
}

int ASR::feature_bins() const {
    return m_feature_bins;
}

int ASR::max_tokens() const {
    return m_max_tokens;
}

int ASR::segment_topk() const {
    return m_segment_topk;
}

float ASR::candidate_temperature() const {
    return m_candidate_temperature;
}

bool ASR::has_char_lm() const {
    return static_cast<bool>(m_char_lm);
}

float ASR::char_lm_scale() const {
    return m_char_lm_scale;
}

int ASR::char_beam_size() const {
    return m_char_beam_size;
}

int ASR::char_topk() const {
    return m_char_topk;
}

std::vector<std::string> ASR::acoustic_vocab() const {
    return m_acoustic_vocab;
}

std::vector<std::string> ASR::p2c_input_vocab() const {
    return m_p2c_input_vocab;
}

std::vector<std::string> ASR::p2c_output_vocab() const {
    return m_p2c_output_vocab;
}

int64_t ASR::last_acoustic_run_us() const {
    return m_acoustic_npu.last_run_us();
}

int64_t ASR::last_p2c_run_us() const {
    return m_p2c_npu.last_run_us();
}

int64_t ASR::last_rerank_run_us() const {
    return m_last_rerank_run_us;
}

int64_t ASR::last_total_run_us() const {
    return m_last_total_run_us;
}
