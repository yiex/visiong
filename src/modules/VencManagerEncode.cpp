// SPDX-License-Identifier: LGPL-3.0-or-later
#include "visiong/modules/VencManager.h"

#include "visiong/core/ImageBuffer.h"
#include "visiong/core/BufferStateMachine.h"
#include "visiong/core/RgaHelper.h"
#include "core/internal/rga_utils.h"
#include "visiong/common/pixel_format.h"
#include "core/internal/logger.h"
#include "core/internal/runtime_init.h"
#include "modules/internal/venc_utils.h"
#include "modules/internal/venc_manager_impl.h"
#include "im2d.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <tuple>
#include <vector>

#include "rk_mpi_mb.h"
#include "rk_mpi_sys.h"
#include "rk_mpi_venc.h"

namespace {

int normalize_quality(int quality) {
    return visiong::venc::clamp_quality(quality);
}

int normalize_encoder_fps(int fps) {
    return visiong::venc::clamp_record_fps(fps);
}

int align_up_to(int value, int base) {
    return (value + base - 1) & (~(base - 1));
}

bool is_yuv420_format(PIXEL_FORMAT_E format) {
    return format == RK_FMT_YUV420SP || format == RK_FMT_YUV420SP_VU;
}

struct VencPackView {
    const unsigned char* data = nullptr;
    size_t len = 0;
};

// Rockchip VENC may expose u32Offset as a global stream offset while pMbBlk / Rockchip VENC 可能把 u32Offset 暴露为全局流偏移，而 pMbBlk
// maps only the current pack payload. In that case, use base directly.
bool resolve_venc_pack_view(const VENC_PACK_S& pack, VencPackView& out_view, bool* out_used_offset = nullptr) {
    out_view = VencPackView{};
    if (out_used_offset) {
        *out_used_offset = false;
    }
    if (pack.pMbBlk == MB_INVALID_HANDLE || pack.u32Len == 0) {
        return false;
    }

    unsigned char* base = static_cast<unsigned char*>(RK_MPI_MB_Handle2VirAddr(pack.pMbBlk));
    if (!base) {
        return false;
    }

    const size_t mb_size = static_cast<size_t>(RK_MPI_MB_GetSize(pack.pMbBlk));
    const size_t offset = static_cast<size_t>(pack.u32Offset);
    const size_t valid_len = static_cast<size_t>(pack.u32Len);
    if (valid_len == 0 || valid_len > mb_size) {
        return false;
    }

    const bool can_use_offset = (offset < mb_size && valid_len <= (mb_size - offset));
    out_view.data = can_use_offset ? (base + offset) : base;
    out_view.len = valid_len;
    if (out_used_offset) {
        *out_used_offset = can_use_offset;
    }
    return true;
}

class ScopedMbBlk {
  public:
    ScopedMbBlk() = default;
    ~ScopedMbBlk() { reset(); }

    ScopedMbBlk(const ScopedMbBlk&) = delete;
    ScopedMbBlk& operator=(const ScopedMbBlk&) = delete;

    void reset(MB_BLK blk = MB_INVALID_HANDLE) {
        if (m_blk != MB_INVALID_HANDLE) {
            RK_MPI_MB_ReleaseMB(m_blk);
        }
        m_blk = blk;
    }

    MB_BLK get() const { return m_blk; }
    bool valid() const { return m_blk != MB_INVALID_HANDLE; }

  private:
    MB_BLK m_blk = MB_INVALID_HANDLE;
};

} // namespace

bool extract_codec_data_from_annexb(const std::vector<unsigned char>& data, VencCodec codec, std::vector<unsigned char>& out_codec, bool& out_keyframe) {
    out_keyframe = false;
    if (codec == VencCodec::JPEG || data.size() < 4) return false;

    auto is_start_code3 = [&](size_t i) {
        return i + 3 <= data.size() && data[i] == 0x00 && data[i + 1] == 0x00 && data[i + 2] == 0x01;
    };
    auto is_start_code4 = [&](size_t i) {
        return i + 4 <= data.size() && data[i] == 0x00 && data[i + 1] == 0x00 && data[i + 2] == 0x00 && data[i + 3] == 0x01;
    };

    size_t i = 0;
    bool added = false;
    while (i + 3 < data.size()) {
        size_t start = std::string::npos;
        size_t start_code_len = 0;
        for (; i + 3 < data.size(); ++i) {
            if (is_start_code4(i)) { start = i; start_code_len = 4; break; }
            if (is_start_code3(i)) { start = i; start_code_len = 3; break; }
        }
        if (start == std::string::npos) break;
        size_t nalu_start = start + start_code_len;
        size_t j = nalu_start;
        for (; j + 3 < data.size(); ++j) {
            if (is_start_code4(j) || is_start_code3(j)) break;
        }
        size_t nalu_end = (j + 3 < data.size()) ? j : data.size();
        if (nalu_start >= nalu_end) {
            i = nalu_end;
            continue;
        }
        const unsigned char* nalu_ptr = data.data() + nalu_start;
        if (codec == VencCodec::H264) {
            unsigned char nalu_type = nalu_ptr[0] & 0x1F;
            if (nalu_type == 7 || nalu_type == 8) {
                out_codec.insert(out_codec.end(), data.begin() + start, data.begin() + nalu_end);
                added = true;
            }
            if (nalu_type == 5) out_keyframe = true;
        } else if (codec == VencCodec::H265) {
            unsigned char nalu_type = (nalu_ptr[0] >> 1) & 0x3F;
            if (nalu_type == 32 || nalu_type == 33 || nalu_type == 34) {
                out_codec.insert(out_codec.end(), data.begin() + start, data.begin() + nalu_end);
                added = true;
            }
            if (nalu_type == 19 || nalu_type == 20) out_keyframe = true;
        }
        i = nalu_end;
    }

    return added;
}

bool VencManager::encodeToVideo(const ImageBuffer& img, VencCodec codec, int quality, VencEncodedPacket& out_packet,
                                int fps, VencRcMode rc_mode) {
    out_packet = VencEncodedPacket();

    if (!visiong_init_sys_if_needed()) {
        VISIONG_LOG_ERROR("VencManager", "System not initialized.");
        return false;
    }

    ImageBuffer temp_buf_owner;
    const ImageBuffer* input_buf = &img;

    if (img.format == visiong::kGray8Format) {
        temp_buf_owner = img.to_format(RK_FMT_YUV420SP);
        input_buf = &temp_buf_owner;
    }

    PIXEL_FORMAT_E current_format = input_buf->format;
    const bool is_supported = (current_format == RK_FMT_YUV420SP || current_format == RK_FMT_YUV420SP_VU ||
                               current_format == RK_FMT_BGR888 || current_format == RK_FMT_RGB888 ||
                               current_format == RK_FMT_RGB565 || current_format == RK_FMT_BGR565);
    if (!is_supported) {
        temp_buf_owner = input_buf->to_format(RK_FMT_YUV420SP);
        input_buf = &temp_buf_owner;
        current_format = input_buf->format;
    }

    std::lock_guard<std::mutex> encode_lock(m_impl->encode_mutex);

    bool initialized_snapshot = false;
    VencConfig current_config_snapshot;
    {
        std::lock_guard<std::mutex> lock(m_impl->mutex);
        initialized_snapshot = m_impl->is_initialized;
        current_config_snapshot = m_impl->current_config;
    }

    PIXEL_FORMAT_E target_format = current_format;
    if (codec != VencCodec::JPEG) {
        if (!is_yuv420_format(current_format)) {
            temp_buf_owner = input_buf->to_format(RK_FMT_YUV420SP);
            input_buf = &temp_buf_owner;
            current_format = input_buf->format;
        }
        target_format = RK_FMT_YUV420SP;
    } else {
        const bool force_venc_bgr888 = false;
        if (force_venc_bgr888 && current_format != RK_FMT_BGR888) {
            temp_buf_owner = input_buf->to_format(RK_FMT_BGR888);
            input_buf = &temp_buf_owner;
            current_format = RK_FMT_BGR888;
        }
        // Follow the current input format for JPEG so camera-originated YUV frames
        // can reach VENC directly instead of being pinned to an older RGB/BGR session.
        // JPEG 路径优先跟随当前输入格式，让摄像头来的 YUV 帧可直接送进 VENC，
        // 不再被旧的 RGB/BGR 会话强行锁住。
        target_format = current_format;
        if (force_venc_bgr888) {
            target_format = RK_FMT_BGR888;
        }
    }

    if (target_format != current_format) {
        temp_buf_owner = input_buf->to_format(target_format);
        input_buf = &temp_buf_owner;
    }

    const int input_w = input_buf->width;
    const int input_h = input_buf->height;
    const int aligned_w = align_up_to(input_w, 16);
    const int aligned_h = align_up_to(input_h, 2);

    VencConfig config;
    config.width = aligned_w;
    config.height = aligned_h;
    config.format = target_format;
    config.codec = static_cast<int>(codec);
    config.quality = normalize_quality(quality);
    config.fps = normalize_encoder_fps(fps);
    config.rc_mode = static_cast<int>(rc_mode);

    if (initialized_snapshot) {
        config.width = std::max(config.width, current_config_snapshot.width);
        config.height = std::max(config.height, current_config_snapshot.height);
    }

    if (config.width <= 0 || config.height <= 0) {
        VISIONG_LOG_ERROR("VencManager", "Image dimensions too small after alignment.");
        return false;
    }

    int vir_w = 0;
    int vir_h = 0;
    MB_POOL input_pool = MB_INVALID_POOLID;
    {
        std::lock_guard<std::mutex> lock(m_impl->mutex);
        if (m_impl->is_initialized) {
            config.width = std::max(config.width, m_impl->current_config.width);
            config.height = std::max(config.height, m_impl->current_config.height);
        }
        if (!ensureVencReady(config)) {
            return false;
        }
        vir_w = m_impl->vir_width;
        vir_h = m_impl->vir_height;
        input_pool = m_impl->input_pool;
    }

    const bool pad_only = (config.width == aligned_w && config.height == aligned_h &&
                           (input_w != config.width || input_h != config.height));
    const bool need_letterbox = (!pad_only && (input_w != config.width || input_h != config.height));

    // Always use an MB from VENC's own pool. / 始终使用 VENC 自身池中的 MB。
    // Camera-wrapped MB/fd may be rejected by VENC in some pipelines.
    ScopedMbBlk mb_blk_guard;

    {
        const size_t mb_size =
            static_cast<size_t>(vir_w) * vir_h * static_cast<size_t>(get_bpp_for_format(target_format)) / 8;
        MB_BLK mb_blk = RK_MPI_MB_GetMB(input_pool, mb_size, RK_TRUE);
        if (mb_blk == MB_INVALID_HANDLE) {
            VISIONG_LOG_ERROR("VencManager", "Failed to get memory block from pool.");
            return false;
        }
        mb_blk_guard.reset(mb_blk);

        void* venc_vir_addr = RK_MPI_MB_Handle2VirAddr(mb_blk_guard.get());
        const int mb_fd = RK_MPI_MB_Handle2Fd(mb_blk_guard.get());
        const size_t mb_bytes = RK_MPI_MB_GetSize(mb_blk_guard.get());
        RgaDmaBuffer dst_dma(mb_fd, venc_vir_addr, mb_bytes, config.width, config.height,
                             static_cast<int>(target_format), vir_w, vir_h);

        const bool input_zero_copy = input_buf->is_zero_copy() && input_buf->get_dma_fd() >= 0;
        unsigned char* dst = static_cast<unsigned char*>(venc_vir_addr);

        auto copy_from_image = [&](const ImageBuffer& src_image, bool clear_padding) {
            const int copy_w = src_image.width;
            const int copy_h = src_image.height;
            const int src_stride = src_image.w_stride;
            visiong::bufstate::prepare_cpu_read(src_image);
            const unsigned char* src_data = static_cast<const unsigned char*>(src_image.get_data());
            visiong::bufstate::prepare_cpu_write(
                dst_dma,
                visiong::bufstate::AccessIntent::Overwrite);

            if (is_yuv420_format(target_format)) {
                if (clear_padding) {
                    std::memset(dst, 0, static_cast<size_t>(vir_w) * vir_h);
                    std::memset(dst + static_cast<size_t>(vir_w) * vir_h, 128, static_cast<size_t>(vir_w) * vir_h / 2);
                }
                const size_t y_plane_src_size = static_cast<size_t>(src_stride) * copy_h;
                copy_data_with_stride(dst, vir_w, src_data, src_stride, copy_h, copy_w);
                copy_data_with_stride(dst + static_cast<size_t>(vir_w) * vir_h, vir_w, src_data + y_plane_src_size,
                                      src_stride, copy_h / 2, copy_w);
            } else {
                const int bpp = get_bpp_for_format(target_format) / 8;
                if (clear_padding) {
                    std::memset(dst, 0, mb_bytes);
                }
                copy_data_with_stride(dst, vir_w * bpp, src_data, src_stride * bpp, copy_h, copy_w * bpp);
            }
            visiong::bufstate::mark_cpu_write(dst_dma);
        };

        auto prepare_for_venc_send = [&]() {
            visiong::bufstate::prepare_mb_device_read(
                mb_blk_guard.get(),
                mb_fd,
                visiong::bufstate::BufferOwner::VENC);
        };

        if (input_zero_copy) {
            RgaDmaBuffer src_wrapper(input_buf->get_dma_fd(), const_cast<void*>(input_buf->get_data()),
                                     input_buf->get_size(), input_w, input_h,
                                     static_cast<int>(input_buf->format), input_buf->w_stride, input_buf->h_stride);

            if (pad_only) {
                visiong::bufstate::prepare_device_read(src_wrapper, visiong::bufstate::BufferOwner::RGA);
                if (is_yuv420_format(target_format)) {
                    rga_letterbox_yuv_op(src_wrapper.get_buffer(), dst_dma, input_w, input_h, 0, 0);
                } else {
                    visiong::bufstate::prepare_device_write(dst_dma,
                                                            visiong::bufstate::BufferOwner::RGA,
                                                            visiong::bufstate::AccessIntent::Overwrite);
                    im_rect full_rect = {0, 0, config.width, config.height};
                    im_rect src_rect = {0, 0, input_w, input_h};
                    im_rect dst_rect = {0, 0, input_w, input_h};
                    if (imfill(dst_dma.get_buffer(), full_rect, 0) != IM_STATUS_SUCCESS ||
                        improcess(src_wrapper.get_buffer(), dst_dma.get_buffer(), {}, src_rect, dst_rect, {}, IM_SYNC) !=
                            IM_STATUS_SUCCESS) {
                        VISIONG_LOG_WARN("VencManager", "RGA pad-only blit failed, fallback to CPU copy.");
                        copy_from_image(*input_buf, true);
                    } else {
                        visiong::bufstate::mark_device_write(dst_dma, visiong::bufstate::BufferOwner::RGA);
                    }
                }
            } else if (need_letterbox) {
                if (is_yuv420_format(target_format)) {
                    const float scale = std::min(static_cast<float>(config.width) / input_w,
                                                 static_cast<float>(config.height) / input_h);
                    const int new_w = static_cast<int>(input_w * scale) & ~1;
                    const int new_h = static_cast<int>(input_h * scale) & ~1;
                    const int pad_x = (config.width - new_w) / 2;
                    const int pad_y = (config.height - new_h) / 2;
                    visiong::bufstate::prepare_device_read(src_wrapper, visiong::bufstate::BufferOwner::RGA);
                    rga_letterbox_yuv_op(src_wrapper.get_buffer(), dst_dma, new_w, new_h, pad_x, pad_y);
                } else {
                    rga_letterbox_op(src_wrapper, dst_dma, std::make_tuple(0, 0, 0));
                }
            } else if (input_buf->format == target_format) {
                visiong::bufstate::prepare_device_read(src_wrapper, visiong::bufstate::BufferOwner::RGA);
                visiong::bufstate::prepare_device_write(dst_dma,
                                                        visiong::bufstate::BufferOwner::RGA,
                                                        visiong::bufstate::AccessIntent::Overwrite);
                im_rect r = {0, 0, config.width, config.height};
                if (improcess(src_wrapper.get_buffer(), dst_dma.get_buffer(), {}, r, r, {}, IM_SYNC) !=
                    IM_STATUS_SUCCESS) {
                    VISIONG_LOG_WARN("VencManager", "RGA improcess (zero-copy) failed, fallback to CPU copy.");
                    copy_from_image(*input_buf, false);
                } else {
                    visiong::bufstate::mark_device_write(dst_dma, visiong::bufstate::BufferOwner::RGA);
                }
            } else {
                const int src_rga = convert_mpi_to_rga_format(input_buf->format);
                const int dst_rga = convert_mpi_to_rga_format(target_format);
                visiong::bufstate::prepare_device_read(src_wrapper, visiong::bufstate::BufferOwner::RGA);
                visiong::bufstate::prepare_device_write(dst_dma,
                                                        visiong::bufstate::BufferOwner::RGA,
                                                        visiong::bufstate::AccessIntent::Overwrite);
                if (imcvtcolor(src_wrapper.get_buffer(), dst_dma.get_buffer(), src_rga, dst_rga) !=
                    IM_STATUS_SUCCESS) {
                    VISIONG_LOG_WARN("VencManager", "RGA imcvtcolor (zero-copy) failed, fallback to CPU convert/copy.");
                    ImageBuffer converted = input_buf->to_format(target_format);
                    copy_from_image(converted, false);
                } else {
                    visiong::bufstate::mark_device_write(dst_dma, visiong::bufstate::BufferOwner::RGA);
                }
            }
            prepare_for_venc_send();
        } else if (need_letterbox) {
            RgaDmaBuffer src_dma(input_w, input_h, static_cast<int>(input_buf->format));
            const int bpp = get_bpp_for_format(input_buf->format);
            visiong::bufstate::prepare_cpu_read(*input_buf);
            copy_data_with_stride(src_dma.get_vir_addr(), src_dma.get_wstride() * bpp / 8, input_buf->get_data(),
                                  input_buf->w_stride * bpp / 8, input_h, input_w * bpp / 8);
            visiong::bufstate::mark_cpu_write(src_dma);
            if (is_yuv420_format(target_format)) {
                const float scale = std::min(static_cast<float>(config.width) / input_w,
                                             static_cast<float>(config.height) / input_h);
                const int new_w = static_cast<int>(input_w * scale) & ~1;
                const int new_h = static_cast<int>(input_h * scale) & ~1;
                const int pad_x = (config.width - new_w) / 2;
                const int pad_y = (config.height - new_h) / 2;
                visiong::bufstate::prepare_device_read(src_dma, visiong::bufstate::BufferOwner::RGA);
                rga_letterbox_yuv_op(src_dma.get_buffer(), dst_dma, new_w, new_h, pad_x, pad_y);
            } else {
                rga_letterbox_op(src_dma, dst_dma, std::make_tuple(0, 0, 0));
            }
            prepare_for_venc_send();
        } else {
            copy_from_image(*input_buf, pad_only);
            prepare_for_venc_send();
        }
    }

    VIDEO_FRAME_INFO_S stSendFrame;
    memset(&stSendFrame, 0, sizeof(stSendFrame));
    stSendFrame.stVFrame.pMbBlk = mb_blk_guard.get();
    stSendFrame.stVFrame.enPixelFormat = config.format;
    stSendFrame.stVFrame.u32Width = config.width;
    stSendFrame.stVFrame.u32Height = config.height;
    stSendFrame.stVFrame.u32VirWidth = static_cast<RK_U32>(vir_w);
    stSendFrame.stVFrame.u32VirHeight = static_cast<RK_U32>(vir_h);

    if (RK_MPI_VENC_SendFrame(VencManagerImpl::kVencChannelId, &stSendFrame, 200) != RK_SUCCESS) {
        VISIONG_LOG_ERROR("VencManager", "Failed to send frame to VENC.");
    } else {
        VENC_STREAM_S stStream;
        memset(&stStream, 0, sizeof(stStream));
        // Keep enough pack slots to avoid silently truncating multi-pack frames. / 保留足够的 pack 槽位，避免多 pack 帧被悄悄截断。
        constexpr RK_U32 kPackCapacity = 128;
        constexpr size_t kMaxPackBytes = 4 * 1024 * 1024;
        constexpr size_t kMaxFrameBytes = 8 * 1024 * 1024;
        std::vector<VENC_PACK_S> packs(kPackCapacity);
        stStream.u32PackCount = static_cast<RK_U32>(packs.size());
        stStream.pstPack = packs.data();
        if (RK_MPI_VENC_GetStream(VencManagerImpl::kVencChannelId, &stStream, 400) == RK_SUCCESS) {
            const RK_U32 reported_pack_count = stStream.u32PackCount;
            const RK_U32 pack_capacity = static_cast<RK_U32>(packs.size());
            const RK_U32 packs_to_process = std::min(reported_pack_count, pack_capacity);
            out_packet.stream_seq = stStream.u32Seq;
            out_packet.pack_count = reported_pack_count;
            out_packet.pack_capacity = pack_capacity;
            out_packet.packs_appended = 0;

            if (reported_pack_count > pack_capacity) {
                VISIONG_LOG_WARN("VencManager",
                                 "VENC returned " << reported_pack_count
                                                  << " packs, but capacity is " << pack_capacity
                                                  << "; dropping frame to avoid truncated bitstream.");
                RK_MPI_VENC_ReleaseStream(VencManagerImpl::kVencChannelId, &stStream);
                return false;
            }

            // Reduce vector growth churn in high-FPS streaming. / 减少高 FPS 流式传输中的 vector 扩容抖动。
            size_t expected_bytes = 0;
            for (RK_U32 i = 0; i < packs_to_process; ++i) {
                const VENC_PACK_S& pack = stStream.pstPack[i];
                expected_bytes += static_cast<size_t>(pack.u32Len);
                if (expected_bytes >= kMaxFrameBytes) {
                    expected_bytes = kMaxFrameBytes;
                    break;
                }
            }
            out_packet.data.reserve(expected_bytes);

            for (RK_U32 i = 0; i < packs_to_process; ++i) {
                const VENC_PACK_S& pack = stStream.pstPack[i];
                if (pack.u32Len == 0) {
                    continue;
                }
                if (pack.pMbBlk == MB_INVALID_HANDLE) {
                    VISIONG_LOG_WARN("VencManager", "VENC pack has invalid MB handle.");
                    continue;
                }

                const int pack_fd = RK_MPI_MB_Handle2Fd(pack.pMbBlk);
                visiong::bufstate::mark_mb_device_write(pack.pMbBlk, pack_fd, visiong::bufstate::BufferOwner::VENC);
                visiong::bufstate::prepare_mb_cpu_read(pack.pMbBlk, pack_fd);
                unsigned char* base = static_cast<unsigned char*>(RK_MPI_MB_Handle2VirAddr(pack.pMbBlk));
                if (!base) {
                    VISIONG_LOG_WARN("VencManager", "Failed to map VENC pack buffer.");
                    continue;
                }

                VencPackView pack_view;
                bool used_offset = false;
                if (!resolve_venc_pack_view(pack, pack_view, &used_offset)) {
                    const size_t mb_size = static_cast<size_t>(RK_MPI_MB_GetSize(pack.pMbBlk));
                    VISIONG_LOG_WARN("VencManager",
                                     "Invalid VENC pack bounds (offset=" << static_cast<size_t>(pack.u32Offset)
                                                                         << ", len=" << static_cast<size_t>(pack.u32Len)
                                                                         << ", mb_size=" << mb_size << ").");
                    continue;
                }
                const size_t valid_len = pack_view.len;
                if (valid_len > kMaxPackBytes) {
                    VISIONG_LOG_WARN("VencManager", "VENC pack too large (" << valid_len << " bytes), skipping.");
                    continue;
                }
                if (out_packet.data.size() + valid_len > kMaxFrameBytes) {
                    VISIONG_LOG_WARN("VencManager", "VENC frame exceeds max budget, truncating at pack " << i << ".");
                    break;
                }

                const unsigned char* pack_data = pack_view.data;
                out_packet.data.insert(out_packet.data.end(), pack_data, pack_data + valid_len);
                ++out_packet.packs_appended;
            }

            if (!out_packet.data.empty()) {
                bool keyframe = false;
                extract_codec_data_from_annexb(out_packet.data, codec, out_packet.codec_data, keyframe);
                out_packet.is_keyframe = keyframe;
            }

            RK_MPI_VENC_ReleaseStream(VencManagerImpl::kVencChannelId, &stStream);
        }
    }

    return !out_packet.data.empty();
}
