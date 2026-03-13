// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_MODULES_VENCRECORDER_H
#define VISIONG_MODULES_VENCRECORDER_H

#include <memory>
#include <string>

class ImageBuffer;

// Hardware VENC-backed recorder that writes Annex-B elementary streams or MP4 files. / 基于硬件 VENC 的录制器，可写出 Annex-B 码流或 MP4 文件。
//
// Design goals: / 设计目标：
// - mirror the Python-facing Display* usage style: construct once, call write(img) repeatedly, then close()
// - keep MP4 finalization explicit because the moov/index is written during close() / - 保持 MP4 收尾动作为显式步骤，因为 moov/index 会在 close() 时写入。
class VencRecorder {
  public:
    enum class Codec { H264 = 0, H265 = 1 };

    enum class Container {
        ANNEXB = 0, // .h264/.h265
        MP4 = 1     // .mp4
    };

    VencRecorder(const std::string& filepath, Codec codec = Codec::H264, Container container = Container::MP4,
                 int quality = 75, const std::string& rc_mode = "cbr", int fps = 30,
                 bool mp4_faststart = true);
    ~VencRecorder();

    VencRecorder(const VencRecorder&) = delete;
    VencRecorder& operator=(const VencRecorder&) = delete;

    // Encode and append one frame. / 编码并追加一帧。
    void write(const ImageBuffer& img);

    // Finalize and close the output file. MP4 outputs require close() to flush metadata. / 完成收尾并关闭输出文件。MP4 输出必须调用 close() 才会刷新元数据。
    void close();

    bool is_open() const;
    std::string path() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

// Helper used by ImageBuffer::save_venc_h26x(container="mp4") to reuse recorders by filepath. / 供 ImageBuffer::save_venc_h26x(container="mp4") 使用的辅助函数，用于按文件路径复用录制器。
void save_venc_mp4_frame(const std::string& filepath, VencRecorder::Codec codec, const ImageBuffer& img,
                         int quality, const std::string& rc_mode, int fps, bool mp4_faststart, bool append);

// Optionally close one implicitly cached recorder. / 按需关闭一个隐式缓存的录制器。
void close_venc_recorder(const std::string& filepath);

// Optionally close all implicitly cached recorders. / 按需关闭所有隐式缓存的录制器。
void close_all_venc_recorders();

#endif  // VISIONG_MODULES_VENCRECORDER_H
