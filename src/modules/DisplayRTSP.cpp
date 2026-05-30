// SPDX-License-Identifier: LGPL-3.0-or-later
#include "visiong/modules/DisplayRTSP.h"

#include "visiong/core/ImageBuffer.h"
#include "visiong/core/NetUtils.h"
#include "visiong/modules/MppEncoderManager.h"
#include "visiong/core/BufferStateMachine.h"
#include "visiong/core/RgaHelper.h"
#include "core/internal/logger.h"
#include "core/internal/runtime_init.h"
#include "modules/internal/mpp_utils.h"
#include "rtsp_demo.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <fcntl.h>
#include <map>
#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <memory>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

DisplayRTSP::RcMode normalize_rc_mode(DisplayRTSP::RcMode mode) {
    return mode == DisplayRTSP::RcMode::VBR ? DisplayRTSP::RcMode::VBR : DisplayRTSP::RcMode::CBR;
}

MppRcMode to_mpp_rc_mode(DisplayRTSP::RcMode mode) {
    return normalize_rc_mode(mode) == DisplayRTSP::RcMode::VBR ? MppRcMode::VBR : MppRcMode::CBR;
}

std::vector<std::string> get_local_ip_addresses() {
    return visiong::get_local_ipv4_addresses();
}

struct RtspPortServer {
    explicit RtspPortServer(int server_port) : port(server_port) {}
    ~RtspPortServer() { stop(); }

    void start(bool suppress_logs);
    void stop();
    void set_suppress_logs(bool suppress);
    rtsp_session_handle add_session(const std::string& path);
    void remove_session(rtsp_session_handle session);

    int port = 0;
    rtsp_demo_handle demo = nullptr;
    std::atomic<bool> running{false};
    std::thread event_thread;
    std::mutex mutex;
    int ref_count = 0;
    bool suppress_logs = true;
};

std::mutex g_rtsp_server_registry_mutex;
std::map<int, std::weak_ptr<RtspPortServer>> g_rtsp_servers;

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

void RtspPortServer::start(bool suppress_logs_requested) {
    std::lock_guard<std::mutex> lk(mutex);
    suppress_logs = suppress_logs_requested;
    if (running.load(std::memory_order_relaxed)) {
        ++ref_count;
        return;
    }

    demo = rtsp_new_demo(port);
    if (!demo) {
        throw std::runtime_error("DisplayRTSP: Failed to create RTSP demo.");
    }

    ref_count = 1;
    running.store(true, std::memory_order_relaxed);
    event_thread = std::thread([this]() {
        while (running.load(std::memory_order_relaxed)) {
            {
                std::lock_guard<std::mutex> event_lock(mutex);
                if (demo) {
                    ScopedLogSilence silence(suppress_logs);
                    rtsp_do_event(demo);
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    });
}

void RtspPortServer::stop() {
    {
        std::lock_guard<std::mutex> lk(mutex);
        if (ref_count > 0) {
            --ref_count;
            if (ref_count > 0) {
                return;
            }
        }
        running.store(false, std::memory_order_relaxed);
    }

    if (event_thread.joinable()) {
        event_thread.join();
    }

    std::lock_guard<std::mutex> lk(mutex);
    if (demo) {
        rtsp_del_demo(demo);
        demo = nullptr;
    }
    ref_count = 0;
}

void RtspPortServer::set_suppress_logs(bool suppress) {
    std::lock_guard<std::mutex> lk(mutex);
    suppress_logs = suppress;
}

rtsp_session_handle RtspPortServer::add_session(const std::string& path) {
    std::lock_guard<std::mutex> lk(mutex);
    if (!demo) {
        return nullptr;
    }
    return rtsp_new_session(demo, path.c_str());
}

void RtspPortServer::remove_session(rtsp_session_handle session) {
    if (!session) {
        return;
    }
    std::lock_guard<std::mutex> lk(mutex);
    rtsp_del_session(session);
}

std::shared_ptr<RtspPortServer> acquire_rtsp_port_server(int port, bool suppress_logs) {
    std::shared_ptr<RtspPortServer> server;
    {
        std::lock_guard<std::mutex> registry_lock(g_rtsp_server_registry_mutex);
        auto& weak = g_rtsp_servers[port];
        server = weak.lock();
        if (!server) {
            server = std::make_shared<RtspPortServer>(port);
            weak = server;
        }
    }
    server->start(suppress_logs);
    return server;
}

void release_rtsp_port_server(std::shared_ptr<RtspPortServer>& server) {
    if (!server) {
        return;
    }
    const int port = server->port;
    server->stop();
    server.reset();
    {
        std::lock_guard<std::mutex> registry_lock(g_rtsp_server_registry_mutex);
        auto it = g_rtsp_servers.find(port);
        if (it != g_rtsp_servers.end() && it->second.expired()) {
            g_rtsp_servers.erase(it);
        }
    }
}

} // namespace

struct DisplayRTSP::Impl {
    Impl(int port,
         std::string path,
         DisplayRTSP::Codec codec,
         int quality,
         int fps,
         int logs,
         DisplayRTSP::RcMode rc_mode,
         int output_width,
         int output_height,
         int preferred_mpp_channel)
        : m_port(port),
          m_path(std::move(path)),
          m_codec(codec),
          m_quality(visiong::mpp::clamp_quality(quality)),
          m_rc_mode(static_cast<int>(normalize_rc_mode(rc_mode))),
          m_session(nullptr),
          m_is_running(false),
          m_video_configured(false),
          m_codec_data_sent(false),
          m_max_fps(visiong::mpp::clamp_non_negative_fps(fps)),
          m_output_width(std::max(0, output_width)),
          m_output_height(std::max(0, output_height)),
          m_preferred_mpp_channel(preferred_mpp_channel),
          m_mpp_channel(-1),
          m_has_sent(false),
          m_last_send(std::chrono::steady_clock::now()),
          m_suppress_logs(logs == 0),
          m_client_active(false) {}

    int m_port;
    std::string m_path;
    DisplayRTSP::Codec m_codec;
    std::atomic<int> m_quality;
    std::atomic<int> m_rc_mode;

    std::shared_ptr<RtspPortServer> m_server;
    rtsp_session_handle m_session;

    std::atomic<bool> m_is_running;
    bool m_video_configured;
    bool m_codec_data_sent;
    std::atomic<int> m_max_fps;
    std::atomic<int> m_output_width;
    std::atomic<int> m_output_height;
    int m_preferred_mpp_channel;
    int m_mpp_channel;
    bool m_has_sent;
    std::chrono::steady_clock::time_point m_last_send;
    std::atomic<bool> m_suppress_logs;
    bool m_client_active;
    mutable std::mutex m_state_mutex;
};

DisplayRTSP::DisplayRTSP(int port, const std::string& path, int quality, Codec codec, int fps, int logs,
                         RcMode rc_mode, int output_width, int output_height, int mpp_channel)
    : m_impl(std::make_unique<Impl>(port,
                                    path,
                                    codec,
                                    quality,
                                    fps,
                                    logs,
                                    rc_mode,
                                    output_width,
                                    output_height,
                                    mpp_channel)) {}

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
        state.m_mpp_channel = MppEncoderManager::getInstance().acquireDedicatedChannel(state.m_preferred_mpp_channel);
        if (state.m_mpp_channel < 0) {
            throw std::runtime_error("DisplayRTSP: no free MPP channel for this RTSP stream.");
        }

        state.m_server = acquire_rtsp_port_server(state.m_port, state.m_suppress_logs.load(std::memory_order_relaxed));
        state.m_session = state.m_server ? state.m_server->add_session(state.m_path) : nullptr;
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

        state.m_is_running.store(true, std::memory_order_relaxed);
    } catch (...) {
        state.m_is_running.store(false, std::memory_order_relaxed);
        if (state.m_session) {
            if (state.m_server) {
                state.m_server->remove_session(state.m_session);
            } else {
                rtsp_del_session(state.m_session);
            }
            state.m_session = nullptr;
        }
        release_rtsp_port_server(state.m_server);
        if (state.m_mpp_channel >= 0) {
            MppEncoderManager::getInstance().releaseDedicatedChannel(state.m_mpp_channel);
            state.m_mpp_channel = -1;
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
                                 << "  MPP ch: " << state.m_mpp_channel
                                 << "  Logs: " << (state.m_suppress_logs.load(std::memory_order_relaxed) ? "0" : "1"));
    if (state.m_output_width.load(std::memory_order_relaxed) > 0 &&
        state.m_output_height.load(std::memory_order_relaxed) > 0) {
        VISIONG_LOG_INFO("DisplayRTSP",
                         "  Output size: " << state.m_output_width.load(std::memory_order_relaxed) << "x"
                                            << state.m_output_height.load(std::memory_order_relaxed));
    }
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
        !state.m_server &&
        state.m_mpp_channel < 0) {
        return;
    }

    state.m_is_running.store(false, std::memory_order_relaxed);

    {
        std::lock_guard<std::mutex> lk(state.m_state_mutex);
        if (state.m_session) {
            if (state.m_server) {
                state.m_server->remove_session(state.m_session);
            } else {
                rtsp_del_session(state.m_session);
            }
            state.m_session = nullptr;
        }
        state.m_video_configured = false;
        state.m_codec_data_sent = false;
        state.m_has_sent = false;
        state.m_client_active = false;
    }

    release_rtsp_port_server(state.m_server);
    if (state.m_mpp_channel >= 0) {
        MppEncoderManager::getInstance().releaseDedicatedChannel(state.m_mpp_channel);
        state.m_mpp_channel = -1;
    }
    VISIONG_LOG_INFO("DisplayRTSP", "Server stopped.");
}

bool DisplayRTSP::is_running() const {
    return m_impl->m_is_running.load(std::memory_order_relaxed);
}

void DisplayRTSP::set_fps(int fps) {
    Impl& state = *m_impl;
    fps = visiong::mpp::clamp_non_negative_fps(fps);
    state.m_max_fps.store(fps, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(state.m_state_mutex);
    state.m_has_sent = false;
    state.m_last_send = std::chrono::steady_clock::now();
}

int DisplayRTSP::get_fps() const {
    return m_impl->m_max_fps.load(std::memory_order_relaxed);
}

void DisplayRTSP::set_quality(int quality) {
    quality = visiong::mpp::clamp_quality(quality);
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
    Impl& state = *m_impl;
    state.m_suppress_logs.store(enable, std::memory_order_relaxed);
    if (state.m_server) {
        state.m_server->set_suppress_logs(enable);
    }
}

bool DisplayRTSP::get_suppress_logs() const {
    return m_impl->m_suppress_logs.load(std::memory_order_relaxed);
}

void DisplayRTSP::set_logs(int logs) {
    set_suppress_logs(logs == 0);
}

int DisplayRTSP::get_logs() const {
    return m_impl->m_suppress_logs.load(std::memory_order_relaxed) ? 0 : 1;
}

void DisplayRTSP::set_output_size(int width, int height) {
    Impl& state = *m_impl;
    width = std::max(0, width);
    height = std::max(0, height);
    state.m_output_width.store(width, std::memory_order_relaxed);
    state.m_output_height.store(height, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(state.m_state_mutex);
    state.m_codec_data_sent = false;
    state.m_has_sent = false;
}

int DisplayRTSP::get_output_width() const {
    return m_impl->m_output_width.load(std::memory_order_relaxed);
}

int DisplayRTSP::get_output_height() const {
    return m_impl->m_output_height.load(std::memory_order_relaxed);
}

int DisplayRTSP::get_mpp_channel() const {
    return m_impl->m_mpp_channel;
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

    const MppCodec mpp_codec = (state.m_codec == Codec::H265) ? MppCodec::H265 : MppCodec::H264;
    const int quality = state.m_quality.load(std::memory_order_relaxed);
    const int fps_enc = visiong::mpp::clamp_record_fps(fps);
    const MppRcMode mpp_rc = to_mpp_rc_mode(static_cast<RcMode>(state.m_rc_mode.load(std::memory_order_relaxed)));

    ImageBuffer resized_img;
    const ImageBuffer* encode_img = &img;
    const int output_width = state.m_output_width.load(std::memory_order_relaxed);
    const int output_height = state.m_output_height.load(std::memory_order_relaxed);
    const bool needs_resize =
        output_width > 0 && output_height > 0 && (img.width != output_width || img.height != output_height);

    MppEncodedPacket packet;
    if (needs_resize) {
        try {
            resized_img = img.resize(output_width, output_height);
            encode_img = &resized_img;
        } catch (const std::exception& e) {
            VISIONG_LOG_WARN("DisplayRTSP", "Failed to resize frame for RTSP stream: " << e.what());
            return false;
        }
    }

    if (state.m_mpp_channel < 0 ||
        !MppEncoderManager::getInstance().encodeToVideoOnChannel(
            state.m_mpp_channel, *encode_img, mpp_codec, quality, packet, fps_enc, mpp_rc)) {
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
        (void)MppEncoderManager::getInstance().requestIDRForChannel(state.m_mpp_channel, true);
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

