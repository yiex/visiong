// SPDX-License-Identifier: LGPL-3.0-or-later
#include "visiong/modules/DisplayRTSP.h"

#include "visiong/core/ImageBuffer.h"
#include "visiong/core/NetUtils.h"
#include "visiong/modules/VencManager.h"
#include "core/internal/logger.h"
#include "modules/internal/venc_utils.h"
#include "rtsp_demo.h"

#include <atomic>
#include <chrono>
#include <fcntl.h>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

DisplayRTSP::RcMode normalize_rc_mode(DisplayRTSP::RcMode mode) {
    return mode == DisplayRTSP::RcMode::VBR ? DisplayRTSP::RcMode::VBR : DisplayRTSP::RcMode::CBR;
}

VencRcMode to_venc_rc_mode(DisplayRTSP::RcMode mode) {
    return normalize_rc_mode(mode) == DisplayRTSP::RcMode::VBR ? VencRcMode::VBR : VencRcMode::CBR;
}

std::vector<std::string> get_local_ip_addresses() {
    return visiong::get_local_ipv4_addresses();
}

int g_null_fd = -1;
int g_saved_stdout = -1;
int g_saved_stderr = -1;
std::mutex g_log_mutex;

class ScopedLogSilence {
  public:
    explicit ScopedLogSilence(bool enable)
        : m_enabled(enable), m_lock(g_log_mutex, std::defer_lock) {
        if (!m_enabled) {
            return;
        }
        m_lock.lock();
        if (g_null_fd < 0) {
            g_null_fd = open("/dev/null", O_WRONLY);
        }
        if (g_null_fd < 0) {
            return;
        }
        g_saved_stdout = dup(STDOUT_FILENO);
        g_saved_stderr = dup(STDERR_FILENO);
        if (g_saved_stdout >= 0) {
            dup2(g_null_fd, STDOUT_FILENO);
        }
        if (g_saved_stderr >= 0) {
            dup2(g_null_fd, STDERR_FILENO);
        }
    }

    ~ScopedLogSilence() {
        if (!m_enabled) {
            return;
        }
        if (g_saved_stdout >= 0) {
            dup2(g_saved_stdout, STDOUT_FILENO);
            close(g_saved_stdout);
            g_saved_stdout = -1;
        }
        if (g_saved_stderr >= 0) {
            dup2(g_saved_stderr, STDERR_FILENO);
            close(g_saved_stderr);
            g_saved_stderr = -1;
        }
        m_lock.unlock();
    }

  private:
    bool m_enabled = false;
    std::unique_lock<std::mutex> m_lock;
};

} // namespace

struct DisplayRTSP::Impl {
    Impl(int port,
         std::string path,
         DisplayRTSP::Codec codec,
         int quality,
         int fps,
         int logs,
         DisplayRTSP::RcMode rc_mode)
        : m_port(port),
          m_path(std::move(path)),
          m_codec(codec),
          m_quality(visiong::venc::clamp_quality(quality)),
          m_rc_mode(static_cast<int>(normalize_rc_mode(rc_mode))),
          m_demo(nullptr),
          m_session(nullptr),
          m_is_running(false),
          m_video_configured(false),
          m_codec_data_sent(false),
          m_max_fps(visiong::venc::clamp_non_negative_fps(fps)),
          m_has_sent(false),
          m_last_send(std::chrono::steady_clock::now()),
          m_suppress_logs(logs == 0),
          m_client_active(false) {}

    void event_loop() {
        while (m_is_running.load(std::memory_order_relaxed)) {
            if (m_demo) {
                ScopedLogSilence silence(m_suppress_logs.load(std::memory_order_relaxed));
                rtsp_do_event(m_demo);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    int m_port;
    std::string m_path;
    DisplayRTSP::Codec m_codec;
    std::atomic<int> m_quality;
    std::atomic<int> m_rc_mode;

    rtsp_demo_handle m_demo;
    rtsp_session_handle m_session;

    std::atomic<bool> m_is_running;
    bool m_video_configured;
    bool m_codec_data_sent;
    std::atomic<int> m_max_fps;
    bool m_has_sent;
    std::chrono::steady_clock::time_point m_last_send;
    std::atomic<bool> m_suppress_logs;
    bool m_client_active;
    std::atomic<bool> m_venc_user_acquired{false};
    mutable std::mutex m_state_mutex;
    std::thread m_event_thread;
};

DisplayRTSP::DisplayRTSP(int port, const std::string& path, int quality, Codec codec, int fps, int logs,
                         RcMode rc_mode)
    : m_impl(std::make_unique<Impl>(port, path, codec, quality, fps, logs, rc_mode)) {}

DisplayRTSP::~DisplayRTSP() {
    stop();
}

void DisplayRTSP::start() {
    Impl& state = *m_impl;
    if (state.m_is_running.load(std::memory_order_relaxed)) {
        VISIONG_LOG_INFO("DisplayRTSP", "Server is already running.");
        return;
    }

    try {
        state.m_demo = rtsp_new_demo(state.m_port);
        if (!state.m_demo) {
            throw std::runtime_error("DisplayRTSP: Failed to create RTSP demo.");
        }

        state.m_session = rtsp_new_session(state.m_demo, state.m_path.c_str());
        if (!state.m_session) {
            throw std::runtime_error("DisplayRTSP: Failed to create RTSP session.");
        }

        int rtsp_codec = RTSP_CODEC_ID_VIDEO_H264;
        if (state.m_codec == Codec::H265) {
            rtsp_codec = RTSP_CODEC_ID_VIDEO_H265;
        }
        {
            ScopedLogSilence silence(state.m_suppress_logs.load(std::memory_order_relaxed));
            rtsp_set_video(state.m_session, rtsp_codec, nullptr, 0);
        }

        {
            std::lock_guard<std::mutex> lk(state.m_state_mutex);
            state.m_video_configured = true;
            state.m_codec_data_sent = false;
            state.m_has_sent = false;
            state.m_last_send = std::chrono::steady_clock::now();
            state.m_client_active = false;
        }

        state.m_venc_user_acquired.store(false, std::memory_order_relaxed);
        state.m_is_running.store(true, std::memory_order_relaxed);
        state.m_event_thread = std::thread([this] { m_impl->event_loop(); });
    } catch (...) {
        state.m_is_running.store(false, std::memory_order_relaxed);
        if (state.m_event_thread.joinable()) {
            state.m_event_thread.join();
        }
        if (state.m_session) {
            rtsp_del_session(state.m_session);
            state.m_session = nullptr;
        }
        if (state.m_demo) {
            rtsp_del_demo(state.m_demo);
            state.m_demo = nullptr;
        }
        {
            std::lock_guard<std::mutex> lk(state.m_state_mutex);
            state.m_video_configured = false;
            state.m_codec_data_sent = false;
            state.m_has_sent = false;
            state.m_client_active = false;
        }
        throw;
    }

    VISIONG_LOG_INFO("DisplayRTSP", "Server started.");
    VISIONG_LOG_INFO("DisplayRTSP",
                     "  Codec: " << (state.m_codec == Codec::H264 ? "H264" : "H265")
                                 << "  Quality: " << state.m_quality.load(std::memory_order_relaxed)
                                 << "  RC: "
                                 << (normalize_rc_mode(static_cast<RcMode>(state.m_rc_mode.load(std::memory_order_relaxed))) ==
                                             RcMode::CBR
                                         ? "CBR"
                                         : "VBR")
                                 << "  FPS: " << state.m_max_fps.load(std::memory_order_relaxed)
                                 << "  Logs: " << (state.m_suppress_logs.load(std::memory_order_relaxed) ? "0" : "1"));
    VISIONG_LOG_INFO("DisplayRTSP", "  Stream URL:");
    auto ips = get_local_ip_addresses();
    if (ips.empty()) {
        VISIONG_LOG_INFO("DisplayRTSP", "  > rtsp://<device-ip>:" << state.m_port << state.m_path);
    } else {
        for (const auto& ip : ips) {
            VISIONG_LOG_INFO("DisplayRTSP", "  > rtsp://" << ip << ":" << state.m_port << state.m_path);
        }
    }
}

void DisplayRTSP::stop() {
    Impl& state = *m_impl;
    if (!state.m_is_running.load(std::memory_order_relaxed) &&
        !state.m_session &&
        !state.m_demo &&
        !state.m_event_thread.joinable()) {
        return;
    }

    state.m_is_running.store(false, std::memory_order_relaxed);
    if (state.m_event_thread.joinable()) {
        state.m_event_thread.join();
    }

    {
        std::lock_guard<std::mutex> lk(state.m_state_mutex);
        if (state.m_session) {
            rtsp_del_session(state.m_session);
            state.m_session = nullptr;
        }
        if (state.m_demo) {
            rtsp_del_demo(state.m_demo);
            state.m_demo = nullptr;
        }
        state.m_video_configured = false;
        state.m_codec_data_sent = false;
        state.m_has_sent = false;
        state.m_client_active = false;
    }

    if (state.m_venc_user_acquired.exchange(false, std::memory_order_relaxed)) {
        VencManager::getInstance().releaseUser();
    }
    VencManager::getInstance().releaseVencIfUnused();
    VISIONG_LOG_INFO("DisplayRTSP", "Server stopped.");
}

bool DisplayRTSP::is_running() const {
    return m_impl->m_is_running.load(std::memory_order_relaxed);
}

void DisplayRTSP::set_fps(int fps) {
    Impl& state = *m_impl;
    fps = visiong::venc::clamp_non_negative_fps(fps);
    state.m_max_fps.store(fps, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(state.m_state_mutex);
    state.m_has_sent = false;
    state.m_last_send = std::chrono::steady_clock::now();
}

int DisplayRTSP::get_fps() const {
    return m_impl->m_max_fps.load(std::memory_order_relaxed);
}

void DisplayRTSP::set_quality(int quality) {
    quality = visiong::venc::clamp_quality(quality);
    m_impl->m_quality.store(quality, std::memory_order_relaxed);
}

int DisplayRTSP::get_quality() const {
    return m_impl->m_quality.load(std::memory_order_relaxed);
}

void DisplayRTSP::set_rc_mode(RcMode mode) {
    m_impl->m_rc_mode.store(static_cast<int>(normalize_rc_mode(mode)), std::memory_order_relaxed);
}

DisplayRTSP::RcMode DisplayRTSP::get_rc_mode() const {
    return normalize_rc_mode(static_cast<RcMode>(m_impl->m_rc_mode.load(std::memory_order_relaxed)));
}

void DisplayRTSP::set_suppress_logs(bool enable) {
    m_impl->m_suppress_logs.store(enable, std::memory_order_relaxed);
}

bool DisplayRTSP::get_suppress_logs() const {
    return m_impl->m_suppress_logs.load(std::memory_order_relaxed);
}

void DisplayRTSP::set_logs(int logs) {
    m_impl->m_suppress_logs.store(logs == 0, std::memory_order_relaxed);
}

int DisplayRTSP::get_logs() const {
    return m_impl->m_suppress_logs.load(std::memory_order_relaxed) ? 0 : 1;
}

bool DisplayRTSP::display(const ImageBuffer& img) {
    Impl& state = *m_impl;
    if (!state.m_is_running.load(std::memory_order_relaxed) || !img.is_valid()) {
        return false;
    }

    const int fps = state.m_max_fps.load(std::memory_order_relaxed);
    if (fps > 0) {
        const auto now = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lk(state.m_state_mutex);
        if (state.m_has_sent) {
            const auto interval = std::chrono::milliseconds(1000 / fps);
            if (now - state.m_last_send < interval) {
                return true;
            }
        }
        state.m_last_send = now;
        state.m_has_sent = true;
    }

    if (!state.m_venc_user_acquired.exchange(true, std::memory_order_relaxed)) {
        VencManager::getInstance().acquireUser();
    }

    const VencCodec venc_codec = (state.m_codec == Codec::H265) ? VencCodec::H265 : VencCodec::H264;
    const int quality = state.m_quality.load(std::memory_order_relaxed);
    const int fps_enc = visiong::venc::clamp_record_fps(fps);
    const VencRcMode venc_rc = to_venc_rc_mode(static_cast<RcMode>(state.m_rc_mode.load(std::memory_order_relaxed)));

    VencEncodedPacket packet;
    if (!VencManager::getInstance().encodeToVideo(img, venc_codec, quality, packet, fps_enc, venc_rc)) {
        return false;
    }
    if (packet.data.empty()) {
        return false;
    }

    rtsp_session_handle session = nullptr;
    bool should_publish_codec_data = false;
    {
        std::lock_guard<std::mutex> lk(state.m_state_mutex);
        if (!state.m_session || !state.m_video_configured) {
            return false;
        }
        session = state.m_session;
        should_publish_codec_data = !state.m_codec_data_sent;
    }

    if (packet.is_keyframe && !packet.codec_data.empty() && should_publish_codec_data) {
        const int rtsp_codec = (state.m_codec == Codec::H265) ? RTSP_CODEC_ID_VIDEO_H265 : RTSP_CODEC_ID_VIDEO_H264;
        int set_video_ret = -1;
        {
            ScopedLogSilence silence(state.m_suppress_logs.load(std::memory_order_relaxed));
            set_video_ret = rtsp_set_video(session, rtsp_codec, packet.codec_data.data(),
                                           static_cast<int>(packet.codec_data.size()));
        }
        if (set_video_ret == 0) {
            std::lock_guard<std::mutex> lk(state.m_state_mutex);
            if (state.m_session == session) {
                state.m_codec_data_sent = true;
            }
        }
    }

    const uint64_t ts = rtsp_get_reltime();
    int ret = -1;
    {
        ScopedLogSilence silence(state.m_suppress_logs.load(std::memory_order_relaxed));
        ret = rtsp_tx_video(session, packet.data.data(), static_cast<int>(packet.data.size()), ts);
    }

    bool log_connected = false;
    bool log_disconnected = false;
    {
        std::lock_guard<std::mutex> lk(state.m_state_mutex);
        if (state.m_session != session) {
            return true;
        }
        if (ret == 0) {
            if (!state.m_client_active) {
                state.m_client_active = true;
                state.m_codec_data_sent = false;
                log_connected = true;
            }
        } else if (state.m_client_active) {
            state.m_client_active = false;
            state.m_has_sent = false;
            state.m_codec_data_sent = false;
            log_disconnected = true;
        }
    }

    if (log_connected) {
        (void)VencManager::getInstance().requestIDR(true);
        if (!state.m_suppress_logs.load(std::memory_order_relaxed)) {
            VISIONG_LOG_INFO("DisplayRTSP", "Client connected (streaming).");
        }
    } else if (log_disconnected) {
        if (!state.m_suppress_logs.load(std::memory_order_relaxed)) {
            VISIONG_LOG_INFO("DisplayRTSP", "Client disconnected.");
        }
    }

    return true;
}

