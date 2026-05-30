// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_MODULES_MPPRECORDER_H
#define VISIONG_MODULES_MPPRECORDER_H

#include <memory>
#include <string>

class ImageBuffer;

// Advanced hardware recorder for Annex-B elementary streams or MP4 files.
class MppRecorder {
  public:
    enum class Codec { H264 = 0, H265 = 1 };

    enum class Container {
        ANNEXB = 0, // .h264/.h265
        MP4 = 1     // .mp4
    };

    MppRecorder(const std::string& filepath, Codec codec = Codec::H264, Container container = Container::MP4,
                 int quality = 75, const std::string& rc_mode = "cbr", int fps = 30,
                 bool mp4_faststart = true, int mpp_channel = -1,
                 const std::string& timestamp_mode = "fixed");
    ~MppRecorder();

    MppRecorder(const MppRecorder&) = delete;
    MppRecorder& operator=(const MppRecorder&) = delete;

    // Encode and append one frame.
    void write(const ImageBuffer& img);

    // Finalize and close the output file. MP4 outputs require close() to flush metadata.
    void close();

    bool is_open() const;
    std::string path() const;
    int get_mpp_channel() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

class VideoRecorder {
  public:
    VideoRecorder(const std::string& filepath, int quality = 75, int fps = 30,
                  const std::string& codec = "auto");
    ~VideoRecorder();

    VideoRecorder(const VideoRecorder&) = delete;
    VideoRecorder& operator=(const VideoRecorder&) = delete;

    void write(const ImageBuffer& img);
    void close();

    // True while the high-level recorder can still accept frames.
    bool is_open() const;

    // True after the underlying output has been opened by the first write().
    bool is_started() const;
    std::string path() const;

  private:
    std::unique_ptr<MppRecorder> m_recorder;
    std::string m_path;
};

// Helper used by ImageBuffer::save_video() and advanced save_mpp_* MP4 paths.
void save_mpp_mp4_frame(const std::string& filepath, MppRecorder::Codec codec, const ImageBuffer& img,
                         int quality, const std::string& rc_mode, int fps, bool mp4_faststart, bool append);

// Optionally close one implicitly cached recorder.
void close_mpp_recorder(const std::string& filepath);

// Optionally close all implicitly cached recorders.
void close_all_mpp_recorders();

// User-friendly aliases for cached MP4 writers created by ImageBuffer::save_video.
void close_video(const std::string& filepath);
void close_all_videos();

#endif  // VISIONG_MODULES_MPPRECORDER_H
