// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_MODULES_INTERNAL_MPP_ENCODER_BACKEND_H
#define VISIONG_MODULES_INTERNAL_MPP_ENCODER_BACKEND_H

#include "rk_comm_video.h"
#include "visiong/modules/MppEncoderManager.h"

class ImageBuffer;

namespace visiong::mpp {

bool is_mpp_supported_config(const MppConfig& config);

class MppEncoderBackend {
  public:
    MppEncoderBackend() = default;
    ~MppEncoderBackend();

    MppEncoderBackend(const MppEncoderBackend&) = delete;
    MppEncoderBackend& operator=(const MppEncoderBackend&) = delete;

    bool configure(const MppConfig& config);
    bool matches(const MppConfig& config) const;
    bool encodeImage(const ImageBuffer& img, MppEncodedPacket& out_packet);
    bool encodeFrameInfo(const VIDEO_FRAME_INFO_S& frame, MppEncodedPacket& out_packet);
    bool requestIDR();
    void reset();

    bool isConfigured() const { return configured_; }
    const MppConfig& config() const { return config_; }
    int horStride() const { return hor_stride_; }
    int verStride() const { return ver_stride_; }

  private:
    bool initMpp(const MppConfig& config);
    bool encodeMppBuffer(void* src_ptr,
                         int src_width,
                         int src_height,
                         int src_hor_stride,
                         PIXEL_FORMAT_E src_format,
                         MppEncodedPacket& out_packet);
    bool encodeCurrentFrame(void* input_buf,
                            int frame_hor_stride,
                            int frame_ver_stride,
                            PIXEL_FORMAT_E src_format,
                            bool imported_input,
                            MppEncodedPacket& out_packet);
    void copyYuv420SpToStride(unsigned char* dst,
                              const unsigned char* src,
                              int width,
                              int height,
                              int src_stride,
                              int dst_stride,
                              int dst_height) const;
    void logFailureOnce(const char* stage, int ret);

    MppConfig config_;
    int hor_stride_ = 0;
    int ver_stride_ = 0;
    bool configured_ = false;
    bool logged_failure_ = false;

    void* ctx_ = nullptr;
    void* mpi_ = nullptr;
    void* cfg_ = nullptr;
    void* buf_group_ = nullptr;
    void* input_buf_ = nullptr;
    void* pkt_buf_ = nullptr;
    void* mpp_lib_ = nullptr;
    void* set_jpege_chan_id_ = nullptr;
    int valloc_fd_ = -1;
};

}  // namespace visiong::mpp

#endif  // VISIONG_MODULES_INTERNAL_MPP_ENCODER_BACKEND_H
