// SPDX-License-Identifier: LGPL-3.0-or-later
#include "visiong/modules/DisplayUDP.h"

#include "visiong/core/ImageBuffer.h"
#include "visiong/common/pixel_format.h"
#include "visiong/modules/VencManager.h"
#include "core/internal/logger.h"
#include "modules/internal/venc_utils.h"
#include "modules/internal/jpeg_lock_utils.h"
#include "core/internal/runtime_init.h"

#include <algorithm>
#include <arpa/inet.h>
#include <atomic>
#include <cstring>
#include <mutex>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

struct DisplayUDP::Impl {
    std::string udp_ip_address;
    int udp_port_number;
    int jpeg_quality;
    std::atomic<bool> initialized{false};
    std::atomic<bool> venc_user_acquired{false};
    int sockfd = -1;
    struct sockaddr_in server_address {};
    uint16_t picture_index = 0;
    std::mutex lock_mutex;
    PIXEL_FORMAT_E locked_format = RK_FMT_BUTT;
    int locked_width = 0;
    int locked_height = 0;
    int locked_input_priority = -1;
};

namespace {

bool init_sys_if_needed() {
    return visiong_init_sys_if_needed();
}

} // namespace

DisplayUDP::DisplayUDP(const std::string& udp_ip, int udp_port, int jpeg_quality)
    : m_impl(std::make_unique<Impl>()) {
    m_impl->udp_ip_address = udp_ip;
    m_impl->udp_port_number = udp_port;
    m_impl->jpeg_quality = visiong::venc::clamp_quality(jpeg_quality);
    std::memset(&m_impl->server_address, 0, sizeof(m_impl->server_address));
    if (!init(udp_ip, udp_port, jpeg_quality)) {
        VISIONG_LOG_ERROR("DisplayUDP", "Initialization failed in constructor.");
    }
}

DisplayUDP::~DisplayUDP() {
    release();
}

bool DisplayUDP::init(const std::string& udp_ip, int udp_port, int jpeg_quality) {
    Impl& state = *m_impl;
    if (state.initialized.load(std::memory_order_relaxed)) {
        release();
    }
    state.udp_ip_address = udp_ip;
    state.udp_port_number = udp_port;
    state.jpeg_quality = visiong::venc::clamp_quality(jpeg_quality);
    {
        std::lock_guard<std::mutex> lock(state.lock_mutex);
        state.locked_format = RK_FMT_BUTT;
        state.locked_width = 0;
        state.locked_height = 0;
        state.locked_input_priority = -1;
    }

    if (!init_sys_if_needed()) {
        return false;
    }

    if (state.sockfd != -1) {
        close(state.sockfd);
    }
    if ((state.sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        return false;
    }
    state.server_address.sin_family = AF_INET;
    state.server_address.sin_port = htons(state.udp_port_number);
    if (inet_pton(AF_INET, state.udp_ip_address.c_str(), &state.server_address.sin_addr) <= 0) {
        close(state.sockfd);
        state.sockfd = -1;
        return false;
    }

    state.initialized.store(true, std::memory_order_relaxed);
    VISIONG_LOG_INFO("DisplayUDP",
                     "Initialized, ready to stream to " << udp_ip << ":" << udp_port
                                                         << " with JPEG quality " << state.jpeg_quality);
    return true;
}

bool DisplayUDP::display(const ImageBuffer& img_buf) {
    Impl& state = *m_impl;
    if (!state.initialized.load(std::memory_order_relaxed) || !img_buf.is_valid()) {
        return false;
    }

    if (!state.venc_user_acquired.exchange(true, std::memory_order_relaxed)) {
        VencManager::getInstance().acquireUser();
    }

    PIXEL_FORMAT_E locked_format = RK_FMT_BUTT;
    int locked_width = 0;
    int locked_height = 0;
    {
        std::lock_guard<std::mutex> lock(state.lock_mutex);
        const int input_priority = visiong::jpeg_lock::get_color_priority(img_buf.format);
        const PIXEL_FORMAT_E requested_lock_format = visiong::jpeg_lock::choose_lock_format(img_buf.format);

        if (state.locked_format == RK_FMT_BUTT || input_priority > state.locked_input_priority) {
            const PIXEL_FORMAT_E previous_format = state.locked_format;
            state.locked_format = requested_lock_format;
            state.locked_input_priority = input_priority;
            if (state.locked_width > 0 && state.locked_height > 0) {
                visiong::jpeg_lock::normalize_lock_size(state.locked_format, &state.locked_width, &state.locked_height);
            }
            VISIONG_LOG_INFO("DisplayUDP",
                             "JPEG color lock updated: "
                                 << (previous_format == RK_FMT_BUTT ? "unset" : PixelFormatToString(previous_format))
                                 << " -> " << PixelFormatToString(state.locked_format)
                                 << " (priority " << input_priority << ")");
        }

        int candidate_width = img_buf.width;
        int candidate_height = img_buf.height;
        visiong::jpeg_lock::normalize_lock_size(state.locked_format, &candidate_width, &candidate_height);
        if (state.locked_width <= 0 || state.locked_height <= 0 ||
            visiong::jpeg_lock::should_expand_lock_size(state.locked_width,
                                                        state.locked_height,
                                                        candidate_width,
                                                        candidate_height)) {
            const int previous_width = state.locked_width;
            const int previous_height = state.locked_height;
            state.locked_width = std::max(state.locked_width, candidate_width);
            state.locked_height = std::max(state.locked_height, candidate_height);
            visiong::jpeg_lock::normalize_lock_size(state.locked_format, &state.locked_width, &state.locked_height);
            VISIONG_LOG_INFO("DisplayUDP",
                             "JPEG size lock updated: "
                                 << previous_width << "x" << previous_height
                                 << " -> " << state.locked_width << "x" << state.locked_height);
        }

        locked_format = state.locked_format;
        locked_width = state.locked_width;
        locked_height = state.locked_height;
    }

    const ImageBuffer* encode_buf = &img_buf;
    ImageBuffer converted_owner;
    ImageBuffer padded_owner;
    if ((locked_format == RK_FMT_YUV420SP && img_buf.format != RK_FMT_YUV420SP) ||
        (locked_format != RK_FMT_YUV420SP && img_buf.format != locked_format)) {
        converted_owner = img_buf.to_format(locked_format);
        encode_buf = &converted_owner;
    } else if (locked_format == RK_FMT_YUV420SP && img_buf.format == RK_FMT_YUV420SP_VU) {
        converted_owner = img_buf.to_format(RK_FMT_YUV420SP);
        encode_buf = &converted_owner;
    }

    if (encode_buf->width != locked_width || encode_buf->height != locked_height) {
        padded_owner = visiong::jpeg_lock::pad_frame_without_scaling(*encode_buf,
                                                                     locked_format,
                                                                     locked_width,
                                                                     locked_height);
        encode_buf = &padded_owner;
    }

    std::vector<unsigned char> jpeg_data =
        VencManager::getInstance().encodeToJpeg(*encode_buf, state.jpeg_quality);
    if (jpeg_data.empty()) {
        VISIONG_LOG_WARN("DisplayUDP", "Failed to encode image.");
        return false;
    }

    if (state.sockfd < 0) {
        return false;
    }

    char header[64];
    const int header_len = snprintf(header, sizeof(header), "frameData,%zu,%hu,jpeg", jpeg_data.size(), state.picture_index++);
    sendto(state.sockfd, header, header_len, 0,
           reinterpret_cast<const sockaddr*>(&state.server_address), sizeof(state.server_address));
    const unsigned char* ptr = jpeg_data.data();
    size_t left = jpeg_data.size();
    while (left > 0) {
        const size_t chunk = std::min(left, static_cast<size_t>(UDP_SEND_MAX_LEN));
        const ssize_t sent = sendto(state.sockfd, ptr, chunk, 0,
                                    reinterpret_cast<const sockaddr*>(&state.server_address),
                                    sizeof(state.server_address));
        if (sent < 0) {
            break;
        }
        ptr += sent;
        left -= static_cast<size_t>(sent);
    }
    return true;
}

void DisplayUDP::release() {
    Impl& state = *m_impl;
    if (state.venc_user_acquired.exchange(false, std::memory_order_relaxed)) {
        VencManager::getInstance().releaseUser();
    }
    VencManager::getInstance().releaseVencIfUnused();

    if (state.sockfd >= 0) {
        close(state.sockfd);
        state.sockfd = -1;
    }
    {
        std::lock_guard<std::mutex> lock(state.lock_mutex);
        state.locked_format = RK_FMT_BUTT;
        state.locked_width = 0;
        state.locked_height = 0;
        state.locked_input_priority = -1;
    }
    state.initialized.store(false, std::memory_order_relaxed);
}

bool DisplayUDP::is_initialized() const {
    return m_impl->initialized.load(std::memory_order_relaxed);
}

const char* DisplayUDP::PixelFormatToString(int format) {
    return visiong::pixel_format_name(static_cast<PIXEL_FORMAT_E>(format));
}
