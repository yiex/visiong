// SPDX-License-Identifier: LGPL-3.0-or-later

#include "visiong/modules/MppEncoderManager.h"

#include "core/internal/logger.h"
#include "core/internal/runtime_init.h"
#include "modules/internal/mpp_encoder_backend.h"
#include "modules/internal/mpp_encoder_manager_impl.h"
#include "modules/internal/mpp_utils.h"
#include "visiong/common/pixel_format.h"
#include "visiong/core/ImageBuffer.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

int align_up_to(int value, int base) {
    return (value + base - 1) & (~(base - 1));
}

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

PIXEL_FORMAT_E normalize_input_format(MppCodec codec, PIXEL_FORMAT_E format) {
    if (format == visiong::kGray8Format) {
        return RK_FMT_YUV420SP;
    }
    return is_supported_input_format(codec, format) ? format : RK_FMT_YUV420SP;
}

MppConfig build_config(int width, int height, PIXEL_FORMAT_E format, MppCodec codec, int quality, int fps, MppRcMode rc_mode) {
    MppConfig config;
    config.width = align_up_to(width, 16);
    config.height = align_up_to(height, 2);
    config.format = format;
    config.codec = static_cast<int>(codec);
    config.quality = visiong::mpp::clamp_quality(quality);
    config.fps = visiong::mpp::clamp_record_fps(fps);
    config.rc_mode = static_cast<int>(rc_mode);
    return config;
}

bool ensure_mpp_backend(MppEncoderManagerImpl::ChannelState& channel, const MppConfig& config) {
    if (!channel.mpp_backend) {
        channel.mpp_backend = std::make_unique<visiong::mpp::MppEncoderBackend>();
    }
    if (!channel.mpp_backend->matches(config)) {
        if (!channel.mpp_backend->configure(config)) {
            channel.is_initialized = false;
            channel.current_config = MppConfig();
            return false;
        }
    }
    channel.current_config = config;
    channel.is_initialized = true;
    return true;
}

}  // namespace

bool extract_codec_data_from_annexb(const std::vector<unsigned char>& data,
                                    MppCodec codec,
                                    std::vector<unsigned char>& out_codec,
                                    bool& out_keyframe) {
    out_keyframe = false;
    if (codec == MppCodec::JPEG || data.size() < 4) {
        return false;
    }

    auto is_start_code3 = [&](size_t i) {
        return i + 3 <= data.size() && data[i] == 0x00 && data[i + 1] == 0x00 && data[i + 2] == 0x01;
    };
    auto is_start_code4 = [&](size_t i) {
        return i + 4 <= data.size() && data[i] == 0x00 && data[i + 1] == 0x00 && data[i + 2] == 0x00 &&
               data[i + 3] == 0x01;
    };

    size_t i = 0;
    bool added = false;
    while (i + 3 < data.size()) {
        size_t start = std::string::npos;
        size_t start_code_len = 0;
        for (; i + 3 < data.size(); ++i) {
            if (is_start_code4(i)) {
                start = i;
                start_code_len = 4;
                break;
            }
            if (is_start_code3(i)) {
                start = i;
                start_code_len = 3;
                break;
            }
        }
        if (start == std::string::npos) {
            break;
        }

        const size_t nalu_start = start + start_code_len;
        size_t j = nalu_start;
        for (; j + 3 < data.size(); ++j) {
            if (is_start_code4(j) || is_start_code3(j)) {
                break;
            }
        }
        const size_t nalu_end = (j + 3 < data.size()) ? j : data.size();
        if (nalu_start >= nalu_end) {
            i = nalu_end;
            continue;
        }

        const unsigned char* nalu_ptr = data.data() + nalu_start;
        if (codec == MppCodec::H264) {
            const unsigned char nalu_type = nalu_ptr[0] & 0x1F;
            if (nalu_type == 7 || nalu_type == 8) {
                out_codec.insert(out_codec.end(), data.begin() + start, data.begin() + nalu_end);
                added = true;
            }
            if (nalu_type == 5) {
                out_keyframe = true;
            }
        } else if (codec == MppCodec::H265) {
            const unsigned char nalu_type = (nalu_ptr[0] >> 1) & 0x3F;
            if (nalu_type == 32 || nalu_type == 33 || nalu_type == 34) {
                out_codec.insert(out_codec.end(), data.begin() + start, data.begin() + nalu_end);
                added = true;
            }
            if (nalu_type == 19 || nalu_type == 20) {
                out_keyframe = true;
            }
        }
        i = nalu_end;
    }

    return added;
}

bool MppEncoderManager::encodeToVideo(const ImageBuffer& img,
                                MppCodec codec,
                                int quality,
                                MppEncodedPacket& out_packet,
                                int fps,
                                MppRcMode rc_mode) {
    return encodeToVideoOnChannel(MppEncoderManagerImpl::kDefaultMppChannelId,
                                  img,
                                  codec,
                                  quality,
                                  out_packet,
                                  fps,
                                  rc_mode);
}

bool MppEncoderManager::encodeToVideoOnChannel(int channel_id,
                                         const ImageBuffer& img,
                                         MppCodec codec,
                                         int quality,
                                         MppEncodedPacket& out_packet,
                                         int fps,
                                         MppRcMode rc_mode) {
    out_packet = MppEncodedPacket();

    if (!visiong_init_sys_if_needed()) {
        VISIONG_LOG_ERROR("MppEncoderManager", "System not initialized.");
        return false;
    }

    if (channel_id < 0 || channel_id >= MppEncoderManagerImpl::kMaxMppChannels) {
        VISIONG_LOG_ERROR("MppEncoderManager", "Invalid MPP channel " << channel_id << ".");
        return false;
    }
    if (!img.is_valid()) {
        return false;
    }

    ImageBuffer temp_buf_owner;
    const ImageBuffer* input_buf = &img;

    if (img.format == visiong::kGray8Format) {
        temp_buf_owner = img.to_format(RK_FMT_YUV420SP);
        if (!temp_buf_owner.is_valid()) {
            return false;
        }
        input_buf = &temp_buf_owner;
    }

    MppConfig config = build_config(input_buf->width, input_buf->height, input_buf->format, codec, quality, fps, rc_mode);
    config.format = normalize_input_format(codec, config.format);
    if (input_buf->format != config.format) {
        temp_buf_owner = input_buf->to_format(config.format);
        if (!temp_buf_owner.is_valid()) {
            return false;
        }
        input_buf = &temp_buf_owner;
        config = build_config(input_buf->width, input_buf->height, input_buf->format, codec, quality, fps, rc_mode);
        config.format = input_buf->format;
    }

    if (!visiong::mpp::is_mpp_supported_config(config)) {
        VISIONG_LOG_ERROR("MppEncoderManager", "Unsupported MPP encode config.");
        return false;
    }

    auto& channel = m_impl->channels[channel_id];
    std::lock_guard<std::mutex> encode_lock(channel.encode_mutex);
    {
        std::lock_guard<std::mutex> lock(m_impl->mutex);
        if (!ensure_mpp_backend(channel, config)) {
            VISIONG_LOG_ERROR("MppEncoderManager", "Failed to configure MPP backend for channel " << channel_id << ".");
            return false;
        }
    }

    if (!channel.mpp_backend->encodeImage(*input_buf, out_packet)) {
        return false;
    }

    return !out_packet.data.empty();
}

bool MppEncoderManager::encodeFrameInfoOnChannel(int channel_id,
                                           const VIDEO_FRAME_INFO_S& frame,
                                           MppCodec codec,
                                           int quality,
                                           MppEncodedPacket& out_packet,
                                           int fps,
                                           MppRcMode rc_mode) {
    out_packet = MppEncodedPacket();

    if (!visiong_init_sys_if_needed()) {
        VISIONG_LOG_ERROR("MppEncoderManager", "System not initialized.");
        return false;
    }

    if (channel_id < 0 || channel_id >= MppEncoderManagerImpl::kMaxMppChannels) {
        VISIONG_LOG_ERROR("MppEncoderManager", "Invalid MPP channel " << channel_id << ".");
        return false;
    }

    const VIDEO_FRAME_S& video = frame.stVFrame;
    if (video.pMbBlk == MB_INVALID_HANDLE || video.u32Width == 0 || video.u32Height == 0) {
        VISIONG_LOG_ERROR("MppEncoderManager", "Invalid VIDEO_FRAME_INFO_S for MPP channel " << channel_id << ".");
        return false;
    }

    if (codec != MppCodec::JPEG && !is_yuv420_format(video.enPixelFormat)) {
        VISIONG_LOG_ERROR("MppEncoderManager", "FrameInfo video encoding requires YUV420SP input.");
        return false;
    }

    MppConfig config = build_config(static_cast<int>(video.u32Width),
                                     static_cast<int>(video.u32Height),
                                     video.enPixelFormat,
                                     codec,
                                     quality,
                                     fps,
                                     rc_mode);
    if (!visiong::mpp::is_mpp_supported_config(config)) {
        VISIONG_LOG_ERROR("MppEncoderManager", "Unsupported MPP frame-info encode config.");
        return false;
    }

    auto& channel = m_impl->channels[channel_id];
    std::lock_guard<std::mutex> encode_lock(channel.encode_mutex);
    {
        std::lock_guard<std::mutex> lock(m_impl->mutex);
        if (!ensure_mpp_backend(channel, config)) {
            VISIONG_LOG_ERROR("MppEncoderManager", "Failed to configure MPP backend for channel " << channel_id << ".");
            return false;
        }
    }

    return channel.mpp_backend->encodeFrameInfo(frame, out_packet) && !out_packet.data.empty();
}
