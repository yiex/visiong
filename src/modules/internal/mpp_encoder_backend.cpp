// SPDX-License-Identifier: LGPL-3.0-or-later

#include "modules/internal/mpp_encoder_backend.h"

#include "visiong/core/BufferStateMachine.h"
#include "visiong/core/ImageBuffer.h"
#include "core/internal/logger.h"
#include "core/internal/rga_utils.h"
#include "modules/internal/mpp_utils.h"

#define MODULE_TAG "visiong_mpp_backend"
#include "rockchip/mpp_buffer.h"
#include "rockchip/mpp_frame.h"
#include "rockchip/mpp_meta.h"
#include "rockchip/mpp_packet.h"
#include "rockchip/rk_mpi.h"
#include "rockchip/rk_venc_cfg.h"
#include "rockchip/rk_venc_rc.h"

#include "rk_mpi_mb.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <thread>
#include <vector>

bool extract_codec_data_from_annexb(const std::vector<unsigned char>& data,
                                    MppCodec codec,
                                    std::vector<unsigned char>& out_codec,
                                    bool& out_keyframe);

namespace visiong::mpp {
namespace {

int align_up(int value, int alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

using PFN_mpp_create_ext = MPP_RET (*)(MppCtx* ctx, MppApi** mpi, int flags);
using PFN_mpp_frame_set_jpege_chan_id = void (*)(MppFrame frame, RK_S32 chan_id);

struct MppPackInfo {
    RK_U32 flag;
    RK_U32 temporal_id;
    RK_U32 packet_offset;
    RK_U32 packet_len;
};

struct MppPacketExt {
    RK_U64 u64priv_data;
    RK_U64 u64packet_addr;
    RK_U32 len;
    RK_U32 buf_size;
    RK_U64 u64pts;
    RK_U64 u64dts;
    RK_U32 flag;
    RK_U32 temporal_id;
    RK_U32 offset;
    RK_U32 data_num;
    MppPackInfo packet[8];
};

struct MppApiExt {
    RK_U32 size;
    RK_U32 version;
    MPP_RET (*decode)(MppCtx ctx, MppPacket packet, MppFrame* frame);
    MPP_RET (*decode_put_packet)(MppCtx ctx, MppPacket packet);
    MPP_RET (*decode_get_frame)(MppCtx ctx, MppFrame* frame);
    MPP_RET (*encode)(MppCtx ctx, MppFrame frame, MppPacket* packet);
    MPP_RET (*encode_put_frame)(MppCtx ctx, MppFrame frame);
    MPP_RET (*encode_get_packet)(MppCtx ctx, MppPacket* packet);
    MPP_RET (*isp)(MppCtx ctx, MppFrame dst, MppFrame src);
    MPP_RET (*isp_put_frame)(MppCtx ctx, MppFrame frame);
    MPP_RET (*isp_get_frame)(MppCtx ctx, MppFrame* frame);
    MPP_RET (*poll)(MppCtx ctx, MppPortType type, MppPollType timeout);
    MPP_RET (*dequeue)(MppCtx ctx, MppPortType type, MppTask* task);
    MPP_RET (*enqueue)(MppCtx ctx, MppPortType type, MppTask task);
    MPP_RET (*reset)(MppCtx ctx);
    MPP_RET (*control)(MppCtx ctx, MpiCmd cmd, MppParam param);
    MPP_RET (*encode_release_packet)(MppCtx ctx, MppPacket* packet);
    RK_U32 reserv[16];
};

struct VallocMb {
    int pool_id;
    int mpi_buf_id;
    int dma_buf_fd;
    int struct_size;
    int size;
    int flags;
    int paddr;
};

constexpr unsigned long kVallocIoctlMbGetFd = 0xc01c6101;

bool is_yuv420_format(PIXEL_FORMAT_E format) {
    return format == RK_FMT_YUV420SP || format == RK_FMT_YUV420SP_VU;
}

bool is_packed_rgb_format(PIXEL_FORMAT_E format) {
    return format == RK_FMT_RGB888 || format == RK_FMT_BGR888 || format == RK_FMT_RGB565 || format == RK_FMT_BGR565;
}

bool is_supported_input_format(MppCodec codec, PIXEL_FORMAT_E format) {
    switch (codec) {
        case MppCodec::H264:
        case MppCodec::H265:
            return is_yuv420_format(format);
        case MppCodec::JPEG:
            return is_yuv420_format(format) || is_packed_rgb_format(format);
    }
    return false;
}

MppCodingType to_mpp_codec(MppCodec codec) {
    switch (codec) {
        case MppCodec::H264:
            return MPP_VIDEO_CodingAVC;
        case MppCodec::H265:
            return MPP_VIDEO_CodingHEVC;
        case MppCodec::JPEG:
            return MPP_VIDEO_CodingMJPEG;
    }
    return MPP_VIDEO_CodingUnused;
}

MppFrameFormat to_mpp_format(PIXEL_FORMAT_E format) {
    switch (format) {
        case RK_FMT_YUV420SP_VU:
            return MPP_FMT_YUV420SP_VU;
        case RK_FMT_RGB888:
            return MPP_FMT_RGB888;
        case RK_FMT_BGR888:
            return MPP_FMT_BGR888;
        case RK_FMT_RGB565:
            return MPP_FMT_RGB565;
        case RK_FMT_BGR565:
            return MPP_FMT_BGR565;
        default:
            return MPP_FMT_YUV420SP;
    }
}

int mpp_hor_stride_for_format(int width, PIXEL_FORMAT_E format) {
    if (is_yuv420_format(format)) {
        return align_up(width, 16);
    }
    if (is_packed_rgb_format(format)) {
        return align_up(width, 8) * (get_bpp_for_format(format) / 8);
    }
    return 0;
}

int mpp_ver_stride_for_format(int height, PIXEL_FORMAT_E format) {
    return is_yuv420_format(format) ? align_up(height, 2) : height;
}

size_t mpp_buffer_size_for_format(int hor_stride, int ver_stride, PIXEL_FORMAT_E format) {
    const size_t aligned_hor = static_cast<size_t>(align_up(hor_stride, 64));
    const size_t aligned_ver = static_cast<size_t>(align_up(ver_stride, 64));
    if (is_yuv420_format(format)) {
        return aligned_hor * aligned_ver * 3 / 2;
    }
    if (is_packed_rgb_format(format)) {
        return aligned_hor * aligned_ver;
    }
    return 0;
}

int stride_bytes_for_format(PIXEL_FORMAT_E format, int stride_pixels) {
    if (is_yuv420_format(format)) {
        return stride_pixels;
    }
    if (is_packed_rgb_format(format)) {
        return stride_pixels * (get_bpp_for_format(format) / 8);
    }
    return 0;
}

size_t row_bytes_for_format(PIXEL_FORMAT_E format, int width_pixels) {
    if (is_yuv420_format(format)) {
        return static_cast<size_t>(width_pixels);
    }
    if (is_packed_rgb_format(format)) {
        return static_cast<size_t>(width_pixels) * static_cast<size_t>(get_bpp_for_format(format) / 8);
    }
    return 0;
}

int estimate_bitrate_bps(int width, int height, int fps, int quality) {
    struct Tier {
        int pixels;
        int low_kbps;
        int medium_kbps;
        int high_kbps;
        int ultra_kbps;
    };
    const Tier tiers[] = {
        {640 * 360, 300, 600, 900, 1200},
        {1280 * 720, 800, 1500, 2500, 3500},
        {1920 * 1080, 1500, 3000, 5000, 8000},
        {2560 * 1440, 2500, 5000, 8000, 12000},
        {3840 * 2160, 6000, 10000, 16000, 25000},
    };

    const int pixels = width * height;
    const Tier* tier = &tiers[0];
    for (const auto& item : tiers) {
        tier = &item;
        if (pixels <= item.pixels) {
            break;
        }
    }

    const int q = clamp_quality(quality);
    int kbps = tier->medium_kbps;
    if (q <= 35) {
        kbps = tier->low_kbps;
    } else if (q <= 60) {
        kbps = tier->medium_kbps;
    } else if (q <= 80) {
        kbps = tier->high_kbps;
    } else {
        kbps = tier->ultra_kbps;
    }

    const int use_fps = clamp_record_fps(fps);
    kbps = static_cast<int>(static_cast<double>(kbps) * use_fps / 30.0);
    return std::max(100, std::min(kbps, 200000)) * 1000;
}

}  // namespace

bool is_mpp_supported_config(const MppConfig& config) {
    const auto codec = static_cast<MppCodec>(config.codec);
    return config.width > 0 && config.height > 0 && is_supported_input_format(codec, config.format);
}

MppEncoderBackend::~MppEncoderBackend() {
    reset();
}

void MppEncoderBackend::reset() {
    if (cfg_) {
        mpp_enc_cfg_deinit(static_cast<MppEncCfg>(cfg_));
        cfg_ = nullptr;
    }
    if (input_buf_) {
        mpp_buffer_put(static_cast<MppBuffer>(input_buf_));
        input_buf_ = nullptr;
    }
    if (pkt_buf_) {
        mpp_buffer_put(static_cast<MppBuffer>(pkt_buf_));
        pkt_buf_ = nullptr;
    }
    if (buf_group_) {
        mpp_buffer_group_put(static_cast<MppBufferGroup>(buf_group_));
        buf_group_ = nullptr;
    }
    if (ctx_) {
        if (mpi_) {
            static_cast<MppApiExt*>(mpi_)->reset(static_cast<MppCtx>(ctx_));
        }
        mpp_destroy(static_cast<MppCtx>(ctx_));
        ctx_ = nullptr;
        mpi_ = nullptr;
    }
    if (valloc_fd_ >= 0) {
        ::close(valloc_fd_);
        valloc_fd_ = -1;
    }
    if (mpp_lib_) {
        dlclose(mpp_lib_);
        mpp_lib_ = nullptr;
    }
    set_jpege_chan_id_ = nullptr;
    config_ = MppConfig();
    hor_stride_ = 0;
    ver_stride_ = 0;
    configured_ = false;
    logged_failure_ = false;
}

bool MppEncoderBackend::matches(const MppConfig& config) const {
    return configured_ &&
           config_.width == config.width &&
           config_.height == config.height &&
           config_.format == config.format &&
           config_.codec == config.codec &&
           config_.quality == config.quality &&
           config_.fps == config.fps &&
           config_.rc_mode == config.rc_mode;
}

bool MppEncoderBackend::configure(const MppConfig& config) {
    if (matches(config)) {
        return true;
    }
    reset();
    if (!is_mpp_supported_config(config)) {
        return false;
    }
    if (!initMpp(config)) {
        reset();
        return false;
    }
    config_ = config;
    configured_ = true;
    return true;
}

bool MppEncoderBackend::initMpp(const MppConfig& config) {
    const auto codec = static_cast<MppCodec>(config.codec);
    hor_stride_ = mpp_hor_stride_for_format(config.width, config.format);
    ver_stride_ = mpp_ver_stride_for_format(config.height, config.format);
    if (hor_stride_ <= 0 || ver_stride_ <= 0) {
        return false;
    }

    MppCtx ctx = nullptr;
    MppApi* mpi = nullptr;
    MppEncCfg cfg = nullptr;
    MppBuffer input = nullptr;
    void* lib = dlopen("librockchip_mpp.so", RTLD_LOCAL | RTLD_NOW);
    if (!lib) lib = dlopen("librockchip_mpp.so.1", RTLD_LOCAL | RTLD_NOW);
    if (!lib) lib = dlopen("librockchip_mpp.so.0", RTLD_LOCAL | RTLD_NOW);
    if (!lib) lib = dlopen("/oem/usr/lib/librockchip_mpp.so", RTLD_LOCAL | RTLD_NOW);
    if (!lib) {
        return false;
    }
    auto mpp_create_ext = reinterpret_cast<PFN_mpp_create_ext>(dlsym(lib, "mpp_create_ext"));
    void* set_jpege_chan_id = dlsym(lib, "mpp_frame_set_jpege_chan_id");
    if (!mpp_create_ext) {
        dlclose(lib);
        return false;
    }
    const int valloc_fd = ::open("/dev/mpi/valloc", O_RDWR | O_CLOEXEC);
    if (valloc_fd < 0) {
        dlclose(lib);
        return false;
    }
    if (mpp_create_ext(&ctx, &mpi, 0) != MPP_OK || !ctx || !mpi) {
        ::close(valloc_fd);
        dlclose(lib);
        return false;
    }

    const MppCodingType coding = to_mpp_codec(codec);
    if (mpp_init(ctx, MPP_CTX_ENC, coding) != MPP_OK) {
        mpp_destroy(ctx);
        ::close(valloc_fd);
        dlclose(lib);
        return false;
    }
    if (mpp_enc_cfg_init(&cfg) != MPP_OK) {
        mpi->reset(ctx);
        mpp_destroy(ctx);
        ::close(valloc_fd);
        dlclose(lib);
        return false;
    }
    if (mpi->control(ctx, MPP_ENC_GET_CFG, cfg) != MPP_OK) {
        mpp_enc_cfg_deinit(cfg);
        mpi->reset(ctx);
        mpp_destroy(ctx);
        ::close(valloc_fd);
        dlclose(lib);
        return false;
    }

    mpp_enc_cfg_set_s32(cfg, "codec:type", coding);
    mpp_enc_cfg_set_s32(cfg, "prep:width", config.width);
    mpp_enc_cfg_set_s32(cfg, "prep:height", config.height);
    mpp_enc_cfg_set_s32(cfg, "prep:hor_stride", hor_stride_);
    mpp_enc_cfg_set_s32(cfg, "prep:ver_stride", ver_stride_);
    mpp_enc_cfg_set_s32(cfg, "prep:format", to_mpp_format(config.format));
    mpp_enc_cfg_set_s32(cfg, "prep:range", MPP_FRAME_RANGE_JPEG);

    if (coding == MPP_VIDEO_CodingMJPEG) {
        const int jpeg_q = clamp_quality(config.quality);
        mpp_enc_cfg_set_s32(cfg, "rc:mode", MPP_ENC_RC_MODE_FIXQP);
        mpp_enc_cfg_set_s32(cfg, "rc:fps_in_flex", 1);
        mpp_enc_cfg_set_s32(cfg, "rc:fps_in_num", 1);
        mpp_enc_cfg_set_s32(cfg, "rc:fps_out_flex", 1);
        mpp_enc_cfg_set_s32(cfg, "rc:fps_out_num", 1);
        mpp_enc_cfg_set_s32(cfg, "rc:gop", 1);
        mpp_enc_cfg_set_s32(cfg, "jpeg:q_factor", jpeg_q);
        mpp_enc_cfg_set_s32(cfg, "jpeg:qf_max", 99);
        mpp_enc_cfg_set_s32(cfg, "jpeg:qf_min", 10);
    } else {
        const int fps = clamp_record_fps(config.fps);
        const int bitrate = estimate_bitrate_bps(config.width, config.height, fps, config.quality);
        const int gop = std::min(std::max(fps * 2, 1), 60);
        const bool vbr = static_cast<MppRcMode>(config.rc_mode) == MppRcMode::VBR;

        mpp_enc_cfg_set_s32(cfg, "rc:mode", vbr ? MPP_ENC_RC_MODE_VBR : MPP_ENC_RC_MODE_CBR);
        mpp_enc_cfg_set_s32(cfg, "rc:fps_in_flex", 0);
        mpp_enc_cfg_set_s32(cfg, "rc:fps_in_num", fps);
        mpp_enc_cfg_set_s32(cfg, "rc:fps_out_flex", 0);
        mpp_enc_cfg_set_s32(cfg, "rc:fps_out_num", fps);
        mpp_enc_cfg_set_s32(cfg, "rc:gop", gop);
        mpp_enc_cfg_set_s32(cfg, "rc:bps_target", bitrate);
        mpp_enc_cfg_set_s32(cfg, "rc:bps_max", vbr ? bitrate * 3 / 2 : bitrate * 17 / 16);
        mpp_enc_cfg_set_s32(cfg, "rc:bps_min", vbr ? std::max(2000, bitrate * 7 / 10) : bitrate * 15 / 16);
        mpp_enc_cfg_set_s32(cfg, "rc:qp_init", -1);
        mpp_enc_cfg_set_s32(cfg, "rc:qp_max", 51);
        mpp_enc_cfg_set_s32(cfg, "rc:qp_min", 10);
        mpp_enc_cfg_set_s32(cfg, "rc:qp_max_i", 51);
        mpp_enc_cfg_set_s32(cfg, "rc:qp_min_i", 10);
        mpp_enc_cfg_set_s32(cfg, "rc:qp_ip", 2);

        if (coding == MPP_VIDEO_CodingAVC) {
            mpp_enc_cfg_set_s32(cfg, "h264:profile", 100);
            mpp_enc_cfg_set_s32(cfg, "h264:level", config.height > 720 ? 40 : 31);
            mpp_enc_cfg_set_s32(cfg, "h264:cabac_en", 1);
            mpp_enc_cfg_set_s32(cfg, "h264:cabac_idc", 0);
            mpp_enc_cfg_set_s32(cfg, "h264:trans8x8", 1);
        }
    }

    if (mpi->control(ctx, MPP_ENC_SET_CFG, cfg) != MPP_OK) {
        mpp_enc_cfg_deinit(cfg);
        mpi->reset(ctx);
        mpp_destroy(ctx);
        ::close(valloc_fd);
        dlclose(lib);
        return false;
    }

    if (coding != MPP_VIDEO_CodingMJPEG) {
        MppEncHeaderMode header_mode = MPP_ENC_HEADER_MODE_EACH_IDR;
        if (mpi->control(ctx, MPP_ENC_SET_HEADER_MODE, &header_mode) != MPP_OK) {
            mpp_enc_cfg_deinit(cfg);
            mpi->reset(ctx);
            mpp_destroy(ctx);
            ::close(valloc_fd);
            dlclose(lib);
            return false;
        }
    }

    const size_t input_size = mpp_buffer_size_for_format(hor_stride_, ver_stride_, config.format);
    if (mpp_buffer_get(nullptr, &input, input_size) != MPP_OK) {
        mpp_enc_cfg_deinit(cfg);
        mpi->reset(ctx);
        mpp_destroy(ctx);
        ::close(valloc_fd);
        dlclose(lib);
        return false;
    }

    ctx_ = ctx;
    mpi_ = mpi;
    cfg_ = cfg;
    input_buf_ = input;
    mpp_lib_ = lib;
    set_jpege_chan_id_ = set_jpege_chan_id;
    valloc_fd_ = valloc_fd;
    return true;
}

bool MppEncoderBackend::encodeImage(const ImageBuffer& img, MppEncodedPacket& out_packet) {
    out_packet = MppEncodedPacket();
    if (!configured_ || !img.is_valid() ||
        !is_supported_input_format(static_cast<MppCodec>(config_.codec), img.format)) {
        return false;
    }

    const int fd = img.is_zero_copy() ? img.get_dma_fd() : -1;
    if (fd >= 0) {
        visiong::bufstate::prepare_device_read(img, visiong::bufstate::BufferOwner::MPP);
    } else {
        visiong::bufstate::prepare_cpu_read(img);
    }

    const void* src = img.get_data();
    if (!src && fd < 0) {
        return false;
    }

    return encodeMppBuffer(const_cast<void*>(src),
                           img.width,
                           img.height,
                           img.w_stride > 0 ? img.w_stride : img.width,
                           img.format,
                           out_packet);
}

bool MppEncoderBackend::encodeFrameInfo(const VIDEO_FRAME_INFO_S& frame, MppEncodedPacket& out_packet) {
    out_packet = MppEncodedPacket();
    if (!configured_) {
        return false;
    }

    const VIDEO_FRAME_S& video = frame.stVFrame;
    if (video.pMbBlk == MB_INVALID_HANDLE ||
        !is_supported_input_format(static_cast<MppCodec>(config_.codec), video.enPixelFormat)) {
        return false;
    }

    const int fd = RK_MPI_MB_Handle2Fd(video.pMbBlk);
    void* ptr = RK_MPI_MB_Handle2VirAddr(video.pMbBlk);
    if (fd >= 0) {
        visiong::bufstate::prepare_mb_device_read(video.pMbBlk, fd, visiong::bufstate::BufferOwner::MPP);
    } else if (ptr) {
        visiong::bufstate::prepare_mb_cpu_read(video.pMbBlk, fd);
    }

    return encodeMppBuffer(ptr,
                           static_cast<int>(video.u32Width),
                           static_cast<int>(video.u32Height),
                           video.u32VirWidth ? static_cast<int>(video.u32VirWidth) : static_cast<int>(video.u32Width),
                           video.enPixelFormat,
                           out_packet);
}

bool MppEncoderBackend::encodeMppBuffer(void* src_ptr,
                                        int src_width,
                                        int src_height,
                                        int src_hor_stride,
                                        PIXEL_FORMAT_E src_format,
                                        MppEncodedPacket& out_packet) {
    if (!configured_ || !is_supported_input_format(static_cast<MppCodec>(config_.codec), src_format)) {
        return false;
    }
    if (src_width < config_.width || src_height < config_.height) {
        return false;
    }

    const bool packed_format = is_packed_rgb_format(src_format);
    const int src_stride_pixels = src_hor_stride > 0 ? src_hor_stride : (packed_format ? src_width : hor_stride_);
    const int src_stride_bytes = stride_bytes_for_format(src_format, src_stride_pixels);

    if (!src_ptr || !input_buf_) {
        return false;
    }

    unsigned char* dst = static_cast<unsigned char*>(mpp_buffer_get_ptr(static_cast<MppBuffer>(input_buf_)));
    if (!dst) {
        return false;
    }
    if (packed_format) {
        const size_t copy_row_bytes = row_bytes_for_format(src_format, src_width);
        const size_t src_row_bytes = static_cast<size_t>(src_stride_bytes);
        const size_t dst_row_bytes = static_cast<size_t>(hor_stride_);
        const unsigned char* src_bytes = static_cast<const unsigned char*>(src_ptr);
        for (int y = 0; y < src_height; ++y) {
            unsigned char* dst_row = dst + static_cast<size_t>(y) * dst_row_bytes;
            const unsigned char* src_row = src_bytes + static_cast<size_t>(y) * src_row_bytes;
            std::memcpy(dst_row, src_row, copy_row_bytes);
            if (dst_row_bytes > copy_row_bytes) {
                std::memset(dst_row + copy_row_bytes, 0, dst_row_bytes - copy_row_bytes);
            }
        }
    } else {
        copyYuv420SpToStride(dst,
                             static_cast<const unsigned char*>(src_ptr),
                             config_.width,
                             config_.height,
                             src_stride_pixels,
                             hor_stride_,
                             ver_stride_);
    }

    return encodeCurrentFrame(static_cast<MppBuffer>(input_buf_),
                              hor_stride_,
                              ver_stride_,
                              src_format,
                              false,
                              out_packet);
}

bool MppEncoderBackend::encodeCurrentFrame(void* input_buf_ptr,
                                           int frame_hor_stride,
                                           int frame_ver_stride,
                                           PIXEL_FORMAT_E src_format,
                                           bool imported_input,
                                           MppEncodedPacket& out_packet) {
    out_packet = MppEncodedPacket();
    if (!configured_ || !ctx_ || !mpi_) {
        return false;
    }

    MppBuffer input_buf = static_cast<MppBuffer>(input_buf_ptr);
    if (!input_buf) {
        return false;
    }

    MppFrame frame = nullptr;
    MPP_RET ret = mpp_frame_init(&frame);
    if (ret != MPP_OK || !frame) {
        if (imported_input) {
            mpp_buffer_put(input_buf);
        }
        return false;
    }

    mpp_frame_set_width(frame, static_cast<RK_U32>(config_.width));
    mpp_frame_set_height(frame, static_cast<RK_U32>(config_.height));
    mpp_frame_set_hor_stride(frame, static_cast<RK_U32>(frame_hor_stride));
    mpp_frame_set_ver_stride(frame, static_cast<RK_U32>(frame_ver_stride));
    mpp_frame_set_fmt(frame, to_mpp_format(src_format));
    mpp_frame_set_pts(frame, 0);
    mpp_frame_set_eos(frame, static_cast<MppCodec>(config_.codec) == MppCodec::JPEG ? 1 : 0);
    mpp_frame_set_buffer(frame, input_buf);

    if (static_cast<MppCodec>(config_.codec) == MppCodec::JPEG && set_jpege_chan_id_) {
        reinterpret_cast<PFN_mpp_frame_set_jpege_chan_id>(set_jpege_chan_id_)(frame, -1);
    }

    MppApiExt* mpi = static_cast<MppApiExt*>(mpi_);
    ret = mpi->encode_put_frame(static_cast<MppCtx>(ctx_), frame);
    mpp_frame_deinit(&frame);
    if (ret != MPP_OK) {
        logFailureOnce("encode_put_frame", ret);
        if (imported_input) {
            mpp_buffer_put(input_buf);
        }
        return false;
    }

    MppPacketExt encoded_packet;
    std::memset(&encoded_packet, 0, sizeof(encoded_packet));
    MppPacket output_packet = reinterpret_cast<MppPacket>(&encoded_packet);
    constexpr int kPacketRetries = 40;
    ret = MPP_OK;
    for (int tries = 0; tries < kPacketRetries; ++tries) {
        std::memset(&encoded_packet, 0, sizeof(encoded_packet));
        output_packet = reinterpret_cast<MppPacket>(&encoded_packet);
        ret = mpi->encode_get_packet(static_cast<MppCtx>(ctx_), &output_packet);
        if (ret != MPP_OK) {
            break;
        }
        if (output_packet && encoded_packet.len > 0) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    if (ret != MPP_OK || !output_packet || encoded_packet.len == 0) {
        logFailureOnce("encode_get_packet", ret);
        if (imported_input) {
            mpp_buffer_put(input_buf);
        }
        return false;
    }

    VallocMb mb;
    std::memset(&mb, 0, sizeof(mb));
    mb.struct_size = sizeof(mb);
    mb.mpi_buf_id = static_cast<int>(encoded_packet.u64priv_data);
    if (::ioctl(valloc_fd_, kVallocIoctlMbGetFd, &mb) != 0 || mb.dma_buf_fd < 0) {
        logFailureOnce("valloc_mb_get_fd", MPP_NOK);
        if (mpi->encode_release_packet) {
            mpi->encode_release_packet(static_cast<MppCtx>(ctx_), &output_packet);
        }
        if (imported_input) {
            mpp_buffer_put(input_buf);
        }
        return false;
    }

    void* mapped = mmap(nullptr, encoded_packet.buf_size, PROT_READ, MAP_SHARED, mb.dma_buf_fd, 0);
    if (mapped == MAP_FAILED) {
        logFailureOnce("mmap_packet", MPP_NOK);
        ::close(mb.dma_buf_fd);
        if (mpi->encode_release_packet) {
            mpi->encode_release_packet(static_cast<MppCtx>(ctx_), &output_packet);
        }
        if (imported_input) {
            mpp_buffer_put(input_buf);
        }
        return false;
    }

    const size_t packet_len = static_cast<size_t>(encoded_packet.len);
    const size_t packet_offset = static_cast<size_t>(encoded_packet.offset);
    const size_t packet_buf_size = static_cast<size_t>(encoded_packet.buf_size);
    if (packet_buf_size > 0 && packet_len > 0 && packet_len <= packet_buf_size) {
        const size_t offset_norm = packet_offset % packet_buf_size;
        const size_t first_len = std::min(packet_len, packet_buf_size - offset_norm);
        const auto* packet_bytes = static_cast<const unsigned char*>(mapped);
        out_packet.data.insert(out_packet.data.end(),
                               packet_bytes + offset_norm,
                               packet_bytes + offset_norm + first_len);
        if (first_len < packet_len) {
            out_packet.data.insert(out_packet.data.end(),
                                   packet_bytes,
                                   packet_bytes + (packet_len - first_len));
        }
        out_packet.pack_count = 1;
        out_packet.pack_capacity = 1;
        out_packet.packs_appended = 1;
    }
    munmap(mapped, encoded_packet.buf_size);
    ::close(mb.dma_buf_fd);

    if (out_packet.data.empty()) {
        VISIONG_LOG_WARN("MppEncoderManager",
                         "MPP packet produced empty output for codec="
                             << static_cast<int>(config_.codec)
                             << " len=" << static_cast<unsigned long long>(encoded_packet.len)
                             << " offset=" << static_cast<unsigned long long>(encoded_packet.offset)
                             << " buf_size=" << static_cast<unsigned long long>(encoded_packet.buf_size)
                             << " data_num=" << static_cast<unsigned long long>(encoded_packet.data_num));
    }

    if (mpi->encode_release_packet) {
        mpi->encode_release_packet(static_cast<MppCtx>(ctx_), &output_packet);
    }
    if (imported_input) {
        mpp_buffer_put(input_buf);
    }

    if (!out_packet.data.empty()) {
        bool keyframe = false;
        extract_codec_data_from_annexb(out_packet.data,
                                       static_cast<MppCodec>(config_.codec),
                                       out_packet.codec_data,
                                       keyframe);
        out_packet.is_keyframe = out_packet.is_keyframe || keyframe;
    }
    return !out_packet.data.empty();
}

bool MppEncoderBackend::requestIDR() {
    if (!configured_ || !ctx_ || !mpi_) {
        return false;
    }
    return static_cast<MppApi*>(mpi_)->control(static_cast<MppCtx>(ctx_), MPP_ENC_SET_IDR_FRAME, nullptr) == MPP_OK;
}

void MppEncoderBackend::copyYuv420SpToStride(unsigned char* dst,
                                             const unsigned char* src,
                                             int width,
                                             int height,
                                             int src_stride,
                                             int dst_stride,
                                             int dst_height) const {
    const int src_y_stride = src_stride > 0 ? src_stride : width;
    const size_t dst_y_size = static_cast<size_t>(dst_stride) * dst_height;
    for (int y = 0; y < height; ++y) {
        std::memcpy(dst + static_cast<size_t>(y) * dst_stride,
                    src + static_cast<size_t>(y) * src_y_stride,
                    static_cast<size_t>(width));
        if (dst_stride > width) {
            std::memset(dst + static_cast<size_t>(y) * dst_stride + width, 0,
                        static_cast<size_t>(dst_stride - width));
        }
    }

    unsigned char* dst_uv = dst + dst_y_size;
    const unsigned char* src_uv = src + static_cast<size_t>(src_y_stride) * height;
    for (int y = 0; y < height / 2; ++y) {
        std::memcpy(dst_uv + static_cast<size_t>(y) * dst_stride,
                    src_uv + static_cast<size_t>(y) * src_y_stride,
                    static_cast<size_t>(width));
        if (dst_stride > width) {
            std::memset(dst_uv + static_cast<size_t>(y) * dst_stride + width, 128,
                        static_cast<size_t>(dst_stride - width));
        }
    }
}

void MppEncoderBackend::logFailureOnce(const char* stage, int ret) {
    if (logged_failure_) {
        return;
    }
    logged_failure_ = true;
    VISIONG_LOG_WARN("MppEncoderManager", "MPP encoder failed at " << stage << " ret=" << ret << ".");
}

}  // namespace visiong::mpp
