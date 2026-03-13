// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_NPU_NANOTRACK_H
#define VISIONG_NPU_NANOTRACK_H

#include <memory>
#include <mutex>
#include <string>
#include <tuple>

class ImageBuffer;

struct NanoTrackResult {
    std::tuple<int, int, int, int> box;
    float score = 0.0f;
};

class NanoTrack {
public:
    NanoTrack(const std::string& template_model_path,
              const std::string& search_model_path,
              const std::string& head_model_path);
    ~NanoTrack();

    NanoTrack(const NanoTrack&) = delete;
    NanoTrack& operator=(const NanoTrack&) = delete;
    NanoTrack(NanoTrack&&) = delete;
    NanoTrack& operator=(NanoTrack&&) = delete;

    void init(const ImageBuffer& image, const std::tuple<int, int, int, int>& bbox);
    NanoTrackResult track(const ImageBuffer& image);

    bool is_initialized() const;
    std::tuple<int, int, int, int> get_bbox() const;
    std::tuple<int, int, int, int> bbox() const { return get_bbox(); }
    float get_score() const;
    float score() const { return get_score(); }
    void reset();

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
    mutable std::mutex m_mutex;
};

#endif  // VISIONG_NPU_NANOTRACK_H

