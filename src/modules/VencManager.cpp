// SPDX-License-Identifier: LGPL-3.0-or-later

#include "visiong/modules/VencManager.h"
#include "visiong/core/ImageBuffer.h"
#include "core/internal/rga_utils.h"
#include "core/internal/logger.h"
#include "core/internal/runtime_init.h"
#include "modules/internal/venc_utils.h"
#include "modules/internal/venc_manager_impl.h"
#include "im2d.h"
#include <algorithm>
#include <cmath>
#include <vector>
#include <cstring>
#include <string>
#include <sstream>
#include <cstdlib>

#include "rk_mpi_mb.h"
#include "rk_mpi_sys.h"
#include "rk_mpi_venc.h"
#include "rk_comm_rc.h"

namespace {
struct BitrateTier {
    int max_pixels;
    int low_kbps;
    int medium_kbps;
    int high_kbps;
    int ultra_kbps;
};

int clamp_int(int value, int min_v, int max_v) {
    if (value < min_v) return min_v;
    if (value > max_v) return max_v;
    return value;
}

int normalize_quality(int quality) {
    return visiong::venc::clamp_quality(quality);
}

int normalize_encoder_fps(int fps) {
    return visiong::venc::clamp_record_fps(fps);
}

RK_U32 to_jpeg_qfactor(int quality) {
    return static_cast<RK_U32>(std::min(normalize_quality(quality), 99));
}

int quality_to_tier(int quality) {
    int q = normalize_quality(quality);
    if (q <= 35) return 0;      // low
    if (q <= 60) return 1;      // medium
    if (q <= 80) return 2;      // high
    return 3;                   // ultra
}

int estimate_bitrate_kbps(int width, int height, int fps, int quality) {
    const BitrateTier tiers[] = {
        {640 * 360,   300,   600,   900,  1200},
        {1280 * 720,  800,  1500,  2500,  3500},
        {1920 * 1080, 1500, 3000,  5000,  8000},
        {2560 * 1440, 2500, 5000,  8000, 12000},
        {3840 * 2160, 6000, 10000, 16000, 25000}
    };

    const int pixels = width * height;
    const BitrateTier* tier = &tiers[0];
    for (const auto& t : tiers) {
        if (pixels <= t.max_pixels) {
            tier = &t;
            break;
        }
        tier = &t;
    }

    int base_kbps = tier->medium_kbps;
    switch (quality_to_tier(quality)) {
        case 0: base_kbps = tier->low_kbps; break;
        case 1: base_kbps = tier->medium_kbps; break;
        case 2: base_kbps = tier->high_kbps; break;
        case 3: base_kbps = tier->ultra_kbps; break;
        default: break;
    }

    const int use_fps = normalize_encoder_fps(fps);
    double scale = static_cast<double>(use_fps) / 30.0;
    int scaled = static_cast<int>(std::lround(base_kbps * scale));
    return clamp_int(scaled, 100, 200000);
}

void destroy_input_pool(MB_POOL& pool) {
    if (pool == MB_INVALID_POOLID) {
        return;
    }
    RK_MPI_MB_DestroyPool(pool);
    pool = MB_INVALID_POOLID;
}

void destroy_venc_channel(VENC_CHN channel_id, bool recv_started) {
    if (recv_started) {
        RK_MPI_VENC_StopRecvFrame(channel_id);
    }
    RK_MPI_VENC_DestroyChn(channel_id);
}

void release_venc_resources(VENC_CHN channel_id, bool& is_initialized, MB_POOL& pool) {
    if (is_initialized) {
        destroy_venc_channel(channel_id, true);
        is_initialized = false;
    }
    destroy_input_pool(pool);
}
} // namespace

VencManager::VencManager() : m_impl(std::make_unique<VencManagerImpl>()) {}

VencManager::~VencManager() {
    std::lock_guard<std::mutex> encode_lock(m_impl->encode_mutex);
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    release_venc_resources(VencManagerImpl::kVencChannelId, m_impl->is_initialized, m_impl->input_pool);
}

void VencManager::acquireUser() {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    if (m_impl->user_count < 0) m_impl->user_count = 0;
    ++m_impl->user_count;
}

void VencManager::releaseUser() {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    if (m_impl->user_count <= 0) {
        m_impl->user_count = 0;
        return;
    }
    --m_impl->user_count;
}

bool VencManager::requestIDR(bool instant) {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    if (!m_impl->is_initialized) return false;
    if (m_impl->current_config.codec == static_cast<int>(VencCodec::JPEG)) return false;
    return RK_MPI_VENC_RequestIDR(VencManagerImpl::kVencChannelId, instant ? RK_TRUE : RK_FALSE) == RK_SUCCESS;
}

bool VencManager::isInitialized() const {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    return m_impl->is_initialized;
}

int VencManager::getWidth() const {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    return m_impl->current_config.width;
}

int VencManager::getHeight() const {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    return m_impl->current_config.height;
}

PIXEL_FORMAT_E VencManager::getFormat() const {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    return m_impl->current_config.format;
}

VencCodec VencManager::getCodec() const {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    return static_cast<VencCodec>(m_impl->current_config.codec);
}

int VencManager::getFps() const {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    return m_impl->current_config.fps;
}

VencRcMode VencManager::getRcMode() const {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    return static_cast<VencRcMode>(m_impl->current_config.rc_mode);
}

int VencManager::getQuality() const {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    return m_impl->current_config.quality;
}
bool VencManager::ensureVencReady(const VencConfig& new_config) {
    auto describe = [](const VencConfig& c) -> std::string {
        std::ostringstream oss;
        oss << c.width << "x" << c.height
            << " [" << PixelFormatToString(c.format) << "] "
            << (static_cast<VencCodec>(c.codec) == VencCodec::H264 ? "H264" :
                (static_cast<VencCodec>(c.codec) == VencCodec::H265 ? "H265" : "JPEG"))
            << " fps=" << c.fps
            << " rc=" << (static_cast<VencRcMode>(c.rc_mode) == VencRcMode::VBR ? "VBR" : "CBR")
            << " q=" << c.quality;
        return oss.str();
    };

    if (m_impl->is_initialized &&
        m_impl->current_config.width == new_config.width &&
        m_impl->current_config.height == new_config.height &&
        m_impl->current_config.format == new_config.format &&
        m_impl->current_config.codec == new_config.codec &&
        m_impl->current_config.fps == new_config.fps &&
        m_impl->current_config.rc_mode == new_config.rc_mode) {

        if (m_impl->current_config.quality == new_config.quality) {
            return true;
        }
        if (new_config.codec == static_cast<int>(VencCodec::JPEG)) {
            // When shared by multiple users, do NOT allow quality changes (would affect others). / 当被多个使用方共享时，禁止修改画质（会影响其他使用方）。
            if (m_impl->user_count > 1) {
                VISIONG_LOG_ERROR("VencManager",
                    "VENC is shared by multiple users; cannot change JPEG quality. "
                    << "Current: " << describe(m_impl->current_config) << " Requested: " << describe(new_config));
                return false;
            }
            VENC_JPEG_PARAM_S stJpegParam;
            memset(&stJpegParam, 0, sizeof(stJpegParam));
            stJpegParam.u32Qfactor = to_jpeg_qfactor(new_config.quality);
            RK_MPI_VENC_SetJpegParam(VencManagerImpl::kVencChannelId, &stJpegParam);
            m_impl->current_config.quality = new_config.quality;
            return true;
        }
        // H264/H265 quality changes require re-init; forbid if multiple users exist. / H264/H265 画质变更需要重新初始化；若存在多个使用方则禁止修改。
        if (m_impl->user_count > 1) {
            VISIONG_LOG_ERROR("VencManager",
                "VENC is shared by multiple users; cannot reconfigure quality. "
                << "Current: " << describe(m_impl->current_config) << " Requested: " << describe(new_config));
            return false;
        }
    }

    // If multiple users exist, forbid any reconfiguration (codec/size/fps/rc/format/quality). / 如果存在多个使用方，则禁止任何重配置（codec/size/fps/rc/format/quality）。
    if (m_impl->is_initialized && m_impl->user_count > 1) {
        VISIONG_LOG_ERROR("VencManager",
            "VENC is shared by multiple users; cannot reconfigure. "
            << "Current: " << describe(m_impl->current_config) << " Requested: " << describe(new_config));
        return false;
    }

    release_venc_resources(VencManagerImpl::kVencChannelId, m_impl->is_initialized, m_impl->input_pool);

    // VENC requires aligned strides: width to 16, height to 2. / VENC 要求步幅对齐：宽对齐到 16，高对齐到 2。
    m_impl->vir_width = (new_config.width + 15) & ~15;
    m_impl->vir_height = (new_config.height + 1) & ~1;

    VENC_CHN_ATTR_S stAttr;
    memset(&stAttr, 0, sizeof(stAttr));
    if (new_config.codec == static_cast<int>(VencCodec::H264)) {
        stAttr.stVencAttr.enType = RK_VIDEO_ID_AVC;
    } else if (new_config.codec == static_cast<int>(VencCodec::H265)) {
        stAttr.stVencAttr.enType = RK_VIDEO_ID_HEVC;
    } else {
        stAttr.stVencAttr.enType = RK_VIDEO_ID_JPEG;
    }
    stAttr.stVencAttr.enPixelFormat = new_config.format;
    stAttr.stVencAttr.u32PicWidth = new_config.width;
    stAttr.stVencAttr.u32PicHeight = new_config.height;
    stAttr.stVencAttr.u32VirWidth = static_cast<RK_U32>(m_impl->vir_width);
    stAttr.stVencAttr.u32VirHeight = static_cast<RK_U32>(m_impl->vir_height);
    stAttr.stVencAttr.u32StreamBufCnt = 4;
    stAttr.stVencAttr.u32BufSize = static_cast<RK_U32>(m_impl->vir_width * m_impl->vir_height * 3 / 2);
    // Some RV1106 firmware builds reject H264 profile=0 and silently fallback to 100. / 某些 RV1106 固件版本会拒绝 H264 profile=0，并悄悄回退到 100。
    // Set it explicitly to avoid noisy runtime warnings.
    stAttr.stVencAttr.u32Profile =
        (new_config.codec == static_cast<int>(VencCodec::H264)) ? 100 : 0;
    stAttr.stVencAttr.bByFrame = RK_TRUE;
    const int cfg_fps = normalize_encoder_fps(new_config.fps);
    // Keep GOP bounded for faster recovery after reconnect/packet loss. / 限制 GOP 上界，以便在重连或丢包后更快恢复。
    const int gop = std::min(std::min(cfg_fps * 2, 60), 250);
    if (new_config.codec == static_cast<int>(VencCodec::H264)) {
        int bitrate = estimate_bitrate_kbps(new_config.width, new_config.height, cfg_fps, new_config.quality);
        if (new_config.rc_mode == static_cast<int>(VencRcMode::VBR)) {
            stAttr.stRcAttr.enRcMode = VENC_RC_MODE_H264VBR;
            stAttr.stRcAttr.stH264Vbr.u32Gop = gop;
            stAttr.stRcAttr.stH264Vbr.u32SrcFrameRateNum = cfg_fps;
            stAttr.stRcAttr.stH264Vbr.u32SrcFrameRateDen = 1;
            stAttr.stRcAttr.stH264Vbr.fr32DstFrameRateNum = static_cast<float>(cfg_fps);
            stAttr.stRcAttr.stH264Vbr.fr32DstFrameRateDen = 1;
            stAttr.stRcAttr.stH264Vbr.u32BitRate = bitrate;
            stAttr.stRcAttr.stH264Vbr.u32MaxBitRate = clamp_int(static_cast<int>(bitrate * 1.5), bitrate, 200000);
            stAttr.stRcAttr.stH264Vbr.u32MinBitRate = clamp_int(static_cast<int>(bitrate * 0.7), 2, bitrate);
            stAttr.stRcAttr.stH264Vbr.u32StatTime = 3;
        } else {
            stAttr.stRcAttr.enRcMode = VENC_RC_MODE_H264CBR;
            stAttr.stRcAttr.stH264Cbr.u32Gop = gop;
            stAttr.stRcAttr.stH264Cbr.u32SrcFrameRateNum = cfg_fps;
            stAttr.stRcAttr.stH264Cbr.u32SrcFrameRateDen = 1;
            stAttr.stRcAttr.stH264Cbr.fr32DstFrameRateNum = static_cast<float>(cfg_fps);
            stAttr.stRcAttr.stH264Cbr.fr32DstFrameRateDen = 1;
            stAttr.stRcAttr.stH264Cbr.u32BitRate = bitrate;
            stAttr.stRcAttr.stH264Cbr.u32StatTime = 3;
        }
    } else if (new_config.codec == static_cast<int>(VencCodec::H265)) {
        int bitrate = estimate_bitrate_kbps(new_config.width, new_config.height, cfg_fps, new_config.quality);
        if (new_config.rc_mode == static_cast<int>(VencRcMode::VBR)) {
            stAttr.stRcAttr.enRcMode = VENC_RC_MODE_H265VBR;
            stAttr.stRcAttr.stH265Vbr.u32Gop = gop;
            stAttr.stRcAttr.stH265Vbr.u32SrcFrameRateNum = cfg_fps;
            stAttr.stRcAttr.stH265Vbr.u32SrcFrameRateDen = 1;
            stAttr.stRcAttr.stH265Vbr.fr32DstFrameRateNum = static_cast<float>(cfg_fps);
            stAttr.stRcAttr.stH265Vbr.fr32DstFrameRateDen = 1;
            stAttr.stRcAttr.stH265Vbr.u32BitRate = bitrate;
            stAttr.stRcAttr.stH265Vbr.u32MaxBitRate = clamp_int(static_cast<int>(bitrate * 1.5), bitrate, 200000);
            stAttr.stRcAttr.stH265Vbr.u32MinBitRate = clamp_int(static_cast<int>(bitrate * 0.7), 2, bitrate);
            stAttr.stRcAttr.stH265Vbr.u32StatTime = 3;
        } else {
            stAttr.stRcAttr.enRcMode = VENC_RC_MODE_H265CBR;
            stAttr.stRcAttr.stH265Cbr.u32Gop = gop;
            stAttr.stRcAttr.stH265Cbr.u32SrcFrameRateNum = cfg_fps;
            stAttr.stRcAttr.stH265Cbr.u32SrcFrameRateDen = 1;
            stAttr.stRcAttr.stH265Cbr.fr32DstFrameRateNum = static_cast<float>(cfg_fps);
            stAttr.stRcAttr.stH265Cbr.fr32DstFrameRateDen = 1;
            stAttr.stRcAttr.stH265Cbr.u32BitRate = bitrate;
            stAttr.stRcAttr.stH265Cbr.u32StatTime = 3;
        }
    } else {
        stAttr.stRcAttr.enRcMode = VENC_RC_MODE_MJPEGFIXQP;
        stAttr.stRcAttr.stMjpegFixQp.u32Qfactor = to_jpeg_qfactor(new_config.quality);
    }

    if (RK_MPI_VENC_CreateChn(VencManagerImpl::kVencChannelId, &stAttr) != RK_SUCCESS) {
        VISIONG_LOG_ERROR("VencManager", "Failed to create VENC channel.");
        return false;
    }

    if (new_config.codec == static_cast<int>(VencCodec::JPEG)) {
        VENC_JPEG_PARAM_S stJpegParam;
        memset(&stJpegParam, 0, sizeof(stJpegParam));
        stJpegParam.u32Qfactor = to_jpeg_qfactor(new_config.quality);
        RK_MPI_VENC_SetJpegParam(VencManagerImpl::kVencChannelId, &stJpegParam);
    }

    MB_POOL_CONFIG_S pool_config;
    memset(&pool_config, 0, sizeof(pool_config));
    // Use a conservative pool block size that covers RGB/YUV intermediates. / 使用保守的池块大小，覆盖 RGB/YUV 中间结果。
    pool_config.u64MBSize = static_cast<RK_U64>(new_config.width) * new_config.height * 3;
    pool_config.u32MBCnt = 2;
    pool_config.enAllocType = MB_ALLOC_TYPE_DMA;
    pool_config.bPreAlloc = RK_TRUE;
    m_impl->input_pool = RK_MPI_MB_CreatePool(&pool_config);
    if (m_impl->input_pool == MB_INVALID_POOLID) {
        VISIONG_LOG_ERROR("VencManager", "Failed to create persistent memory pool.");
        destroy_venc_channel(VencManagerImpl::kVencChannelId, false);
        return false;
    }

    VENC_RECV_PIC_PARAM_S stRecvParam;
    memset(&stRecvParam, 0, sizeof(stRecvParam));
        stRecvParam.s32RecvPicNum = -1;
    if (RK_MPI_VENC_StartRecvFrame(VencManagerImpl::kVencChannelId, &stRecvParam) != RK_SUCCESS) {
        destroy_input_pool(m_impl->input_pool);
        destroy_venc_channel(VencManagerImpl::kVencChannelId, false);
        VISIONG_LOG_ERROR("VencManager", "Failed to start VENC frame receiving.");
        return false;
    }

    m_impl->current_config = new_config;
    m_impl->is_initialized = true;
    VISIONG_LOG_INFO("VencManager",
                     "VENC channel " << VencManagerImpl::kVencChannelId << " and memory pool are ready for "
                                     << m_impl->current_config.width << "x" << m_impl->current_config.height
                                     << " format " << PixelFormatToString(m_impl->current_config.format));

    return true;
}

std::vector<unsigned char> VencManager::encodeToJpeg(const ImageBuffer& img, int quality) {
    VencEncodedPacket packet;
    if (!encodeToVideo(img, VencCodec::JPEG, quality, packet)) {
        return {};
    }
    return packet.data;
}

int VencManager::clamp_quality(int quality) {
    return normalize_quality(quality);
}

int VencManager::quality_to_qp(int quality) {
    int q = clamp_quality(quality);
    int qp = 51 - (q - 1) * 50 / 99;
    if (qp < 1) qp = 1;
    if (qp > 51) qp = 51;
    return qp;
}

void VencManager::releaseVencIfUnused() {
    std::lock_guard<std::mutex> encode_lock(m_impl->encode_mutex);
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    if (m_impl->user_count > 0) {
        // Still in use by some module(s); don't tear down the single HW channel. / 仍被某些模块使用；不要拆掉唯一的硬件通道。
        return;
    }
    const bool was_initialized = m_impl->is_initialized;
    release_venc_resources(VencManagerImpl::kVencChannelId, m_impl->is_initialized, m_impl->input_pool);
    if (was_initialized) {
        VISIONG_LOG_INFO("VencManager", "VENC channel released by user.");
    }
}
