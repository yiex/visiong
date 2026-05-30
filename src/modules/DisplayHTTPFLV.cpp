// SPDX-License-Identifier: LGPL-3.0-or-later
#include "visiong/modules/DisplayHTTPFLV.h"

#include "visiong/core/ImageBuffer.h"
#include "visiong/core/NetUtils.h"
#include "visiong/modules/MppEncoderManager.h"
#include "core/internal/logger.h"
#include "modules/internal/http_socket_utils.h"
#include "modules/internal/mpp_utils.h"

#include <algorithm>
#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <errno.h>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sstream>
#include <stdexcept>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

extern "C" {
#include "flv-header.h"
#include "flv-muxer.h"
#include "flv-proto.h"
#include "flv-writer.h"
}

namespace {

std::string build_html_player(const std::string& flv_path) {
    std::ostringstream os;
    os << "<!DOCTYPE html>\n"
          "<html><head><meta charset=\"utf-8\"/>\n"
          "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>\n"
          "<title>VisionG HTTP-FLV</title>\n"
          "<style>body{margin:0;background:#111;color:#eee;font-family:sans-serif}"
          "#v{width:100%;height:100vh;object-fit:contain;transform:translateZ(0);backface-visibility:hidden;}"
          "</style>\n"
          "</head><body>\n"
          "<video id=\"v\" autoplay muted playsinline controls preload=\"auto\"></video>\n"
          "<script src=\"https://cdn.jsdelivr.net/npm/mpegts.js/dist/mpegts.js\"></script>\n"
          "<script>\n"
          "const v=document.getElementById('v');\n"
          "v.muted=true;v.autoplay=true;v.playsInline=true;\n"
          "v.setAttribute('playsinline','');v.setAttribute('webkit-playsinline','');\n"
          "if (mpegts.getFeatureList().mseLivePlayback) {\n"
          "  const url=(location.origin||(location.protocol+'//'+location.host))+'" << flv_path << "';\n"
          "  const player=mpegts.createPlayer({type:'flv',url:url,isLive:true},{\n"
          "    lazyLoad:false,enableWorker:true,enableStashBuffer:true,stashInitialSize:384*1024,\n"
          "    autoCleanupSourceBuffer:true,autoCleanupMaxBackwardDuration:8,autoCleanupMinBackwardDuration:3\n"
          "  });\n"
          "  player.attachMediaElement(v);player.load();\n"
          "  const play=()=>{const r=v.play();if(r&&r.catch)r.catch(()=>{});};\n"
          "  v.addEventListener('loadedmetadata',play);v.addEventListener('canplay',play);\n"
          "  player.on(mpegts.Events.ERROR,()=>{setTimeout(()=>location.reload(),1000);});\n"
          "  play();\n"
          "} else { v.outerHTML='<p>Browser does not support MSE live playback.</p>'; }\n"
          "</script>\n"
          "</body></html>\n";
    return os.str();
}

DisplayHTTPFLV::RcMode normalize_rc_mode(DisplayHTTPFLV::RcMode mode) {
    return mode == DisplayHTTPFLV::RcMode::VBR ? DisplayHTTPFLV::RcMode::VBR : DisplayHTTPFLV::RcMode::CBR;
}

MppRcMode to_mpp_rc_mode(DisplayHTTPFLV::RcMode mode) {
    return normalize_rc_mode(mode) == DisplayHTTPFLV::RcMode::VBR ? MppRcMode::VBR : MppRcMode::CBR;
}

void set_socket_send_timeout(int fd, int timeout_ms) {
    if (fd < 0 || timeout_ms <= 0) return;
    struct timeval tv;
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    (void)::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}

}  // namespace

struct DisplayHTTPFLV::Impl {
    static constexpr size_t kMaxClients = 8;

    struct Client;

    int port = 8080;
    std::string path = "/live.flv";
    std::atomic<int> codec{static_cast<int>(Codec::H264)};
    std::atomic<int> quality{75};
    std::atomic<int> rc_mode{static_cast<int>(RcMode::CBR)};
    std::atomic<int> max_fps{30};
    std::atomic<bool> running{false};
    std::atomic<bool> need_idr{false};

    int listen_fd = -1;
    std::thread accept_thread;
    std::thread encoder_thread;
    std::mutex clients_mtx;
    std::list<std::shared_ptr<Client>> clients;

    int mpp_channel = -1;
    std::mutex state_mtx;
    bool timestamp_started = false;
    std::chrono::steady_clock::time_point timestamp_start;
    uint32_t last_timestamp_ms = 0;
    bool idr_started = false;
    std::chrono::steady_clock::time_point last_idr_request;

    std::vector<unsigned char> last_codec_data;

    void accept_loop();
    void encoder_loop();
    void cleanup_dead_locked();
    bool has_live_clients();
    uint32_t next_timestamp_ms(std::chrono::steady_clock::time_point frame_time);
    bool should_request_idr(std::chrono::steady_clock::time_point frame_time);
    bool encode_latest_frame(const ImageBuffer& img, std::chrono::steady_clock::time_point frame_time);
    void broadcast_packet(const MppEncodedPacket& packet,
                          Codec packet_codec,
                          int fps,
                          uint32_t timestamp_ms);
    void close_all_clients();

    std::mutex latest_mtx;
    std::condition_variable latest_cv;
    std::shared_ptr<const ImageBuffer> latest_frame;
    uint64_t latest_seq = 0;
};

struct DisplayHTTPFLV::Impl::Client {
    int fd = -1;
    void* writer = nullptr;
    flv_muxer_t* muxer = nullptr;
    std::atomic<bool> alive{true};
    std::mutex queue_mtx;
    std::condition_variable queue_cv;
    std::deque<std::vector<unsigned char>> queue;
    size_t queued_bytes = 0;
    std::thread sender_thread;
    std::vector<unsigned char> codec_data;
    bool timestamp_base_set = false;
    uint32_t timestamp_base = 0;
    uint32_t last_timestamp_ms = 0;
    Codec codec = Codec::H264;
    int width = 0;
    int height = 0;
    int fps = 30;

    static constexpr size_t kMaxQueuedBytes = 1024 * 1024;
    static constexpr size_t kMaxQueuedPackets = 90;

    ~Client() { stop(); }

    static int on_writer_write(void* param, const struct flv_vec_t* vec, int n) {
        auto* client = static_cast<Client*>(param);
        if (!client || !client->alive.load(std::memory_order_relaxed)) return -1;
        if (!vec || n <= 0) return 0;

        size_t total = 0;
        for (int i = 0; i < n; ++i) {
            if (vec[i].len > 0) total += static_cast<size_t>(vec[i].len);
        }
        if (total == 0) return 0;

        std::vector<unsigned char> packet;
        packet.reserve(total);
        for (int i = 0; i < n; ++i) {
            if (vec[i].len <= 0) continue;
            const auto* ptr = static_cast<const unsigned char*>(vec[i].ptr);
            packet.insert(packet.end(), ptr, ptr + vec[i].len);
        }

        {
            std::lock_guard<std::mutex> lk(client->queue_mtx);
            if (client->queue.size() >= kMaxQueuedPackets || client->queued_bytes + packet.size() > kMaxQueuedBytes) {
                client->alive.store(false, std::memory_order_relaxed);
                client->queue_cv.notify_all();
                return -1;
            }
            client->queued_bytes += packet.size();
            client->queue.emplace_back(std::move(packet));
        }
        client->queue_cv.notify_one();
        return 0;
    }

    static int on_muxer_output(void* param, int type, const void* data, size_t bytes, uint32_t timestamp) {
        auto* client = static_cast<Client*>(param);
        if (!client || !client->writer || !client->alive.load(std::memory_order_relaxed)) return -1;
        return flv_writer_input(client->writer, type, data, bytes, timestamp);
    }

    bool start(int client_fd) {
        fd = client_fd;
        writer = flv_writer_create2(0, 1, &Client::on_writer_write, this);
        if (!writer) {
            stop();
            return false;
        }
        muxer = flv_muxer_create(&Client::on_muxer_output, this);
        if (!muxer) {
            stop();
            return false;
        }
        flv_muxer_set_enhanced_rtmp(muxer, 0);
        try {
            sender_thread = std::thread([this]() { sender_loop(); });
        } catch (...) {
            stop();
            return false;
        }
        return true;
    }

    void stop() {
        alive.store(false, std::memory_order_relaxed);
        queue_cv.notify_all();
        if (fd >= 0) {
            ::shutdown(fd, SHUT_RDWR);
        }
        if (sender_thread.joinable()) sender_thread.join();
        if (muxer) {
            flv_muxer_destroy(muxer);
            muxer = nullptr;
        }
        if (writer) {
            flv_writer_destroy(writer);
            writer = nullptr;
        }
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
    }

    void sender_loop() {
        while (alive.load(std::memory_order_relaxed)) {
            std::vector<unsigned char> packet;
            {
                std::unique_lock<std::mutex> lk(queue_mtx);
                queue_cv.wait(lk, [&]() {
                    return !alive.load(std::memory_order_relaxed) || !queue.empty();
                });
                if (!alive.load(std::memory_order_relaxed) && queue.empty()) break;
                if (queue.empty()) continue;
                packet = std::move(queue.front());
                queue.pop_front();
                queued_bytes -= packet.size();
            }
            if (!packet.empty() && !visiong::http::send_all(fd, packet.data(), packet.size())) {
                alive.store(false, std::memory_order_relaxed);
                break;
            }
        }
        alive.store(false, std::memory_order_relaxed);
    }

    bool write_packet(const MppEncodedPacket& packet,
                      Codec packet_codec,
                      int packet_width,
                      int packet_height,
                      int packet_fps,
                      uint32_t timestamp_ms) {
        if (!alive.load(std::memory_order_relaxed) || !muxer || packet.data.empty()) return false;

        const bool stream_changed =
            codec != packet_codec ||
            width != packet_width ||
            height != packet_height ||
            fps != packet_fps ||
            codec_data != packet.codec_data;
        if (stream_changed) {
            codec = packet_codec;
            width = packet_width;
            height = packet_height;
            fps = packet_fps;
            codec_data = packet.codec_data;
            timestamp_base_set = false;
            timestamp_base = timestamp_ms;
            last_timestamp_ms = 0;
            flv_muxer_reset(muxer);
            flv_muxer_set_enhanced_rtmp(muxer, 0);

            flv_metadata_t meta;
            std::memset(&meta, 0, sizeof(meta));
            meta.videocodecid = (codec == Codec::H264) ? FLV_VIDEO_H264 : FLV_VIDEO_H265;
            meta.width = width;
            meta.height = height;
            meta.framerate = static_cast<double>(fps > 0 ? fps : 30);
            (void)flv_muxer_metadata(muxer, &meta);
            if (!codec_data.empty()) {
                const int rc = (codec == Codec::H264)
                                   ? flv_muxer_avc(muxer, codec_data.data(), codec_data.size(), 0, 0)
                                   : flv_muxer_hevc(muxer, codec_data.data(), codec_data.size(), 0, 0);
                if (rc != 0) return false;
            }
        }

        if (!timestamp_base_set) {
            timestamp_base = timestamp_ms;
            timestamp_base_set = true;
        }
        uint32_t ts = timestamp_ms - timestamp_base;
        if (last_timestamp_ms != 0 || ts == 0) {
            ts = std::max(ts, last_timestamp_ms + (last_timestamp_ms == 0 ? 0 : 1));
        }
        last_timestamp_ms = ts;
        const int rc = (packet_codec == Codec::H264)
                           ? flv_muxer_avc(muxer, packet.data.data(), packet.data.size(), ts, ts)
                           : flv_muxer_hevc(muxer, packet.data.data(), packet.data.size(), ts, ts);
        if (rc != 0) {
            alive.store(false, std::memory_order_relaxed);
            return false;
        }
        return true;
    }
};

void DisplayHTTPFLV::Impl::cleanup_dead_locked() {
    for (auto it = clients.begin(); it != clients.end();) {
        if (!*it || !(*it)->alive.load(std::memory_order_relaxed)) {
            it = clients.erase(it);
        } else {
            ++it;
        }
    }
}

bool DisplayHTTPFLV::Impl::has_live_clients() {
    std::lock_guard<std::mutex> lk(clients_mtx);
    cleanup_dead_locked();
    return !clients.empty();
}

uint32_t DisplayHTTPFLV::Impl::next_timestamp_ms(std::chrono::steady_clock::time_point frame_time) {
    if (!timestamp_started) {
        timestamp_start = frame_time;
        timestamp_started = true;
        last_timestamp_ms = 0;
        return 0;
    }
    int64_t ts = std::chrono::duration_cast<std::chrono::milliseconds>(frame_time - timestamp_start).count();
    if (ts <= static_cast<int64_t>(last_timestamp_ms)) {
        ts = static_cast<int64_t>(last_timestamp_ms) + 1;
    }
    if (ts > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        ts = static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
    }
    last_timestamp_ms = static_cast<uint32_t>(ts);
    return last_timestamp_ms;
}

bool DisplayHTTPFLV::Impl::should_request_idr(std::chrono::steady_clock::time_point frame_time) {
    constexpr auto kIdrInterval = std::chrono::milliseconds(2000);
    if (need_idr.exchange(false, std::memory_order_relaxed)) return true;
    if (!idr_started) {
        idr_started = true;
        last_idr_request = frame_time;
        return true;
    }
    if (frame_time - last_idr_request >= kIdrInterval) {
        last_idr_request = frame_time;
        return true;
    }
    return false;
}

void DisplayHTTPFLV::Impl::broadcast_packet(const MppEncodedPacket& packet,
                                            Codec packet_codec,
                                            int fps,
                                            uint32_t timestamp_ms) {
    std::vector<std::shared_ptr<Client>> snapshot;
    {
        std::lock_guard<std::mutex> lk(clients_mtx);
        cleanup_dead_locked();
        snapshot.reserve(clients.size());
        for (auto& client : clients) {
            if (client && client->alive.load(std::memory_order_relaxed)) snapshot.push_back(client);
        }
    }
    if (snapshot.empty()) return;

    const int width = MppEncoderManager::getInstance().getWidth(mpp_channel);
    const int height = MppEncoderManager::getInstance().getHeight(mpp_channel);
    for (auto& client : snapshot) {
        if (!client->write_packet(packet, packet_codec, width, height, fps, timestamp_ms)) {
            client->alive.store(false, std::memory_order_relaxed);
        }
    }
    {
        std::lock_guard<std::mutex> lk(clients_mtx);
        cleanup_dead_locked();
    }
}

bool DisplayHTTPFLV::Impl::encode_latest_frame(const ImageBuffer& img,
                                               std::chrono::steady_clock::time_point frame_time) {
    if (mpp_channel < 0 || !img.is_valid()) return false;

    const int fps = visiong::mpp::clamp_non_negative_fps(max_fps.load(std::memory_order_relaxed));
    const Codec out_codec = static_cast<Codec>(codec.load(std::memory_order_relaxed));
    const MppCodec mpp_codec = (out_codec == Codec::H265) ? MppCodec::H265 : MppCodec::H264;
    const int quality_value = quality.load(std::memory_order_relaxed);
    const int enc_fps = visiong::mpp::clamp_record_fps(fps);
    const auto rc = to_mpp_rc_mode(static_cast<RcMode>(rc_mode.load(std::memory_order_relaxed)));

    {
        std::lock_guard<std::mutex> lk(state_mtx);
        if (should_request_idr(frame_time)) {
            (void)MppEncoderManager::getInstance().requestIDRForChannel(mpp_channel, true);
        }
    }

    MppEncodedPacket packet;
    if (!MppEncoderManager::getInstance().encodeToVideoOnChannel(
            mpp_channel, img, mpp_codec, quality_value, packet, enc_fps, rc) ||
        packet.data.empty()) {
        need_idr.store(true, std::memory_order_relaxed);
        return true;
    }

    if (packet.is_keyframe && !packet.codec_data.empty()) {
        std::lock_guard<std::mutex> lk(state_mtx);
        last_codec_data = packet.codec_data;
    } else if (packet.codec_data.empty()) {
        std::lock_guard<std::mutex> lk(state_mtx);
        packet.codec_data = last_codec_data;
    }

    uint32_t ts = 0;
    {
        std::lock_guard<std::mutex> lk(state_mtx);
        ts = next_timestamp_ms(frame_time);
    }
    broadcast_packet(packet, out_codec, enc_fps, ts);
    return true;
}

void DisplayHTTPFLV::Impl::encoder_loop() {
    uint64_t consumed_seq = 0;
    auto last_encode_time = std::chrono::steady_clock::now() - std::chrono::seconds(1);

    while (running.load(std::memory_order_relaxed)) {
        if (!has_live_clients()) {
            std::unique_lock<std::mutex> lk(latest_mtx);
            latest_cv.wait_for(lk, std::chrono::milliseconds(100), [&]() {
                return !running.load(std::memory_order_relaxed) || latest_seq != consumed_seq;
            });
            consumed_seq = latest_seq;
            last_encode_time = std::chrono::steady_clock::now() - std::chrono::seconds(1);
            continue;
        }

        std::shared_ptr<const ImageBuffer> frame;
        uint64_t frame_seq = consumed_seq;
        {
            std::unique_lock<std::mutex> lk(latest_mtx);
            latest_cv.wait(lk, [&]() {
                return !running.load(std::memory_order_relaxed) ||
                       (latest_frame && latest_seq != consumed_seq);
            });
            if (!running.load(std::memory_order_relaxed)) break;
            frame = latest_frame;
            frame_seq = latest_seq;
        }

        if (!frame || !frame->is_valid()) {
            consumed_seq = frame_seq;
            continue;
        }

        auto frame_time = std::chrono::steady_clock::now();
        const int fps = visiong::mpp::clamp_non_negative_fps(max_fps.load(std::memory_order_relaxed));
        if (fps > 0) {
            const auto interval = std::chrono::microseconds(1000000 / fps);
            const auto next_allowed = last_encode_time + interval;
            if (frame_time < next_allowed) {
                std::unique_lock<std::mutex> lk(latest_mtx);
                latest_cv.wait_until(lk, next_allowed, [&]() {
                    return !running.load(std::memory_order_relaxed) || latest_seq != frame_seq;
                });
                if (!running.load(std::memory_order_relaxed)) break;
                if (latest_seq != frame_seq) {
                    frame = latest_frame;
                    frame_seq = latest_seq;
                }
                frame_time = std::chrono::steady_clock::now();
            }
        }

        consumed_seq = frame_seq;
        last_encode_time = frame_time;
        (void)encode_latest_frame(*frame, frame_time);
    }
}

void DisplayHTTPFLV::Impl::close_all_clients() {
    std::list<std::shared_ptr<Client>> closing;
    {
        std::lock_guard<std::mutex> lk(clients_mtx);
        closing.splice(closing.end(), clients);
    }
    closing.clear();
}

void DisplayHTTPFLV::Impl::accept_loop() {
    while (running.load(std::memory_order_relaxed)) {
        sockaddr_in addr{};
        socklen_t len = sizeof(addr);
        int cfd = ::accept(listen_fd, reinterpret_cast<sockaddr*>(&addr), &len);
        if (cfd < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
            if (!running.load(std::memory_order_relaxed)) break;
            continue;
        }

        struct timeval rtv;
        rtv.tv_sec = 1;
        rtv.tv_usec = 0;
        ::setsockopt(cfd, SOL_SOCKET, SO_RCVTIMEO, &rtv, sizeof(rtv));

        std::string req;
        if (!visiong::http::read_http_request(cfd, req)) {
            ::close(cfd);
            continue;
        }
        std::string req_path;
        if (!visiong::http::parse_http_request_path(req, req_path)) {
            const char* resp400 = "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n";
            visiong::http::send_all(cfd, resp400, std::strlen(resp400));
            ::close(cfd);
            continue;
        }
        const size_t qm = req_path.find('?');
        const std::string clean_path = (qm == std::string::npos) ? req_path : req_path.substr(0, qm);

        if (clean_path == "/" || clean_path == "/index.html") {
            const std::string html = build_html_player(path);
            std::ostringstream hdr;
            hdr << "HTTP/1.1 200 OK\r\n"
                << "Content-Type: text/html; charset=utf-8\r\n"
                << "Content-Length: " << html.size() << "\r\n"
                << "Cache-Control: no-cache\r\n"
                << "Connection: close\r\n\r\n";
            visiong::http::send_all(cfd, hdr.str().c_str(), hdr.str().size());
            visiong::http::send_all(cfd, html.data(), html.size());
            ::close(cfd);
            continue;
        }
        if (clean_path != path && clean_path != path + "/") {
            const char* resp404 = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n";
            visiong::http::send_all(cfd, resp404, std::strlen(resp404));
            ::close(cfd);
            continue;
        }

        std::shared_ptr<Client> client;
        {
            std::lock_guard<std::mutex> lk(clients_mtx);
            cleanup_dead_locked();
            if (clients.size() >= kMaxClients) {
                const char* resp503 = "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
                visiong::http::send_all(cfd, resp503, std::strlen(resp503));
                ::close(cfd);
                VISIONG_LOG_WARN("DisplayHTTPFLV", "Client rejected: too many active clients.");
                continue;
            }
        }

        const char* resp =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: video/x-flv\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: close\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "\r\n";
        if (!visiong::http::send_all(cfd, resp, std::strlen(resp))) {
            ::close(cfd);
            continue;
        }
        int nodelay = 1;
        ::setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
        set_socket_send_timeout(cfd, 3000);

        client = std::make_shared<Client>();
        if (!client->start(cfd)) {
            continue;
        }
        {
            std::lock_guard<std::mutex> lk(clients_mtx);
            cleanup_dead_locked();
            if (clients.size() >= kMaxClients) {
                client->stop();
                VISIONG_LOG_WARN("DisplayHTTPFLV", "Client rejected: too many active clients.");
                continue;
            }
            clients.push_back(client);
        }
        need_idr.store(true, std::memory_order_relaxed);
        VISIONG_LOG_INFO("DisplayHTTPFLV", "Client connected.");
    }
}

DisplayHTTPFLV::DisplayHTTPFLV(int port, const std::string& path, int quality, Codec codec, int fps, RcMode rc_mode)
    : m_impl(std::make_unique<Impl>()) {
    m_impl->port = port;
    m_impl->path = path.empty() ? "/live.flv" : path;
    if (m_impl->path[0] != '/') m_impl->path = "/" + m_impl->path;
    m_impl->codec.store(static_cast<int>(codec), std::memory_order_relaxed);
    m_impl->quality.store(visiong::mpp::clamp_quality(quality), std::memory_order_relaxed);
    m_impl->max_fps.store(visiong::mpp::clamp_non_negative_fps(fps), std::memory_order_relaxed);
    m_impl->rc_mode.store(static_cast<int>(normalize_rc_mode(rc_mode)), std::memory_order_relaxed);
}

DisplayHTTPFLV::~DisplayHTTPFLV() {
    stop();
}

void DisplayHTTPFLV::start() {
    if (!m_impl) return;
    if (m_impl->running.load(std::memory_order_relaxed)) return;

    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) throw std::runtime_error("DisplayHTTPFLV: socket() failed.");

    int on = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
    struct timeval tv;
    tv.tv_sec = 1;
    tv.tv_usec = 0;
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(static_cast<uint16_t>(m_impl->port));

    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        ::close(fd);
        throw std::runtime_error("DisplayHTTPFLV: bind() failed on port " + std::to_string(m_impl->port));
    }
    if (::listen(fd, 16) != 0) {
        ::close(fd);
        throw std::runtime_error("DisplayHTTPFLV: listen() failed.");
    }

    try {
        start_encoder_only();
    } catch (...) {
        ::close(fd);
        throw;
    }

    m_impl->listen_fd = fd;
    try {
        m_impl->accept_thread = std::thread([impl = m_impl.get()]() { impl->accept_loop(); });
    } catch (...) {
        ::close(fd);
        m_impl->listen_fd = -1;
        stop();
        throw;
    }

    Codec c = static_cast<Codec>(m_impl->codec.load(std::memory_order_relaxed));
    VISIONG_LOG_INFO("DisplayHTTPFLV", "Server started.");
    VISIONG_LOG_INFO("DisplayHTTPFLV",
                     "  Codec: " << (c == Codec::H264 ? "H264" : "H265")
                                 << "  Quality: " << m_impl->quality.load(std::memory_order_relaxed)
                                 << "  FPS: " << m_impl->max_fps.load(std::memory_order_relaxed)
                                 << "  MPP ch: " << m_impl->mpp_channel);
    VISIONG_LOG_INFO("DisplayHTTPFLV", "  Stream URL:");
    auto ips = visiong::get_local_ipv4_addresses();
    if (ips.empty()) {
        VISIONG_LOG_INFO("DisplayHTTPFLV", "  > http://<device-ip>:" << m_impl->port << m_impl->path);
    } else {
        for (const auto& ip : ips) {
            VISIONG_LOG_INFO("DisplayHTTPFLV", "  > http://" << ip << ":" << m_impl->port << m_impl->path);
        }
    }
}

void DisplayHTTPFLV::start_encoder_only() {
    if (!m_impl) return;
    if (m_impl->running.load(std::memory_order_relaxed)) return;

    m_impl->mpp_channel = MppEncoderManager::getInstance().acquireDedicatedChannel(-1);
    if (m_impl->mpp_channel < 0) {
        throw std::runtime_error("DisplayHTTPFLV: no free MPP channel.");
    }

    {
        std::lock_guard<std::mutex> lk(m_impl->state_mtx);
        m_impl->timestamp_started = false;
        m_impl->last_timestamp_ms = 0;
        m_impl->idr_started = false;
        m_impl->last_codec_data.clear();
    }
    {
        std::lock_guard<std::mutex> lk(m_impl->latest_mtx);
        m_impl->latest_frame.reset();
        m_impl->latest_seq = 0;
    }
    m_impl->need_idr.store(true, std::memory_order_relaxed);
    m_impl->running.store(true, std::memory_order_relaxed);
    try {
        m_impl->encoder_thread = std::thread([impl = m_impl.get()]() { impl->encoder_loop(); });
    } catch (...) {
        m_impl->running.store(false, std::memory_order_relaxed);
        MppEncoderManager::getInstance().releaseDedicatedChannel(m_impl->mpp_channel);
        m_impl->mpp_channel = -1;
        throw;
    }
}

void DisplayHTTPFLV::add_stream_client(int client_fd) {
    if (!m_impl || !m_impl->running.load(std::memory_order_relaxed)) {
        ::close(client_fd);
        return;
    }
    {
        std::lock_guard<std::mutex> lk(m_impl->clients_mtx);
        m_impl->cleanup_dead_locked();
        if (m_impl->clients.size() >= Impl::kMaxClients) {
            const char* resp503 = "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
            visiong::http::send_all(client_fd, resp503, std::strlen(resp503));
            ::close(client_fd);
            VISIONG_LOG_WARN("DisplayHTTPFLV", "Client rejected: too many active clients.");
            return;
        }
    }
    const char* resp =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: video/x-flv\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: close\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "\r\n";
    if (!visiong::http::send_all(client_fd, resp, std::strlen(resp))) {
        ::close(client_fd);
        return;
    }
    int nodelay = 1;
    ::setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
    set_socket_send_timeout(client_fd, 3000);

    auto client = std::make_shared<Impl::Client>();
    if (!client->start(client_fd)) {
        return;
    }
    {
        std::lock_guard<std::mutex> lk(m_impl->clients_mtx);
        m_impl->cleanup_dead_locked();
        if (m_impl->clients.size() >= Impl::kMaxClients) {
            client->stop();
            VISIONG_LOG_WARN("DisplayHTTPFLV", "Client rejected: too many active clients.");
            return;
        }
        m_impl->clients.push_back(client);
    }
    m_impl->need_idr.store(true, std::memory_order_relaxed);
    VISIONG_LOG_INFO("DisplayHTTPFLV", "Client connected.");
}

std::string DisplayHTTPFLV::get_index_html() const {
    if (!m_impl) return "";
    return build_html_player(m_impl->path);
}

const std::string& DisplayHTTPFLV::get_path() const {
    static const std::string empty;
    return m_impl ? m_impl->path : empty;
}

void DisplayHTTPFLV::stop() {
    if (!m_impl) return;
    if (!m_impl->running.load(std::memory_order_relaxed) &&
        m_impl->listen_fd < 0 &&
        !m_impl->accept_thread.joinable() &&
        m_impl->mpp_channel < 0) {
        return;
    }

    m_impl->running.store(false, std::memory_order_relaxed);
    if (m_impl->listen_fd >= 0) {
        ::shutdown(m_impl->listen_fd, SHUT_RDWR);
        ::close(m_impl->listen_fd);
        m_impl->listen_fd = -1;
    }
    if (m_impl->accept_thread.joinable()) {
        m_impl->accept_thread.join();
    }
    m_impl->latest_cv.notify_all();
    if (m_impl->encoder_thread.joinable()) {
        m_impl->encoder_thread.join();
    }
    m_impl->close_all_clients();
    if (m_impl->mpp_channel >= 0) {
        MppEncoderManager::getInstance().releaseDedicatedChannel(m_impl->mpp_channel);
        m_impl->mpp_channel = -1;
    }
    {
        std::lock_guard<std::mutex> lk(m_impl->latest_mtx);
        m_impl->latest_frame.reset();
    }
    VISIONG_LOG_INFO("DisplayHTTPFLV", "Server stopped.");
}

bool DisplayHTTPFLV::is_running() const {
    return m_impl && m_impl->running.load(std::memory_order_relaxed);
}

void DisplayHTTPFLV::set_fps(int fps) {
    if (!m_impl) return;
    m_impl->max_fps.store(visiong::mpp::clamp_non_negative_fps(fps), std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(m_impl->state_mtx);
    m_impl->idr_started = false;
}

int DisplayHTTPFLV::get_fps() const {
    return m_impl ? m_impl->max_fps.load(std::memory_order_relaxed) : 0;
}

void DisplayHTTPFLV::set_quality(int quality) {
    if (!m_impl) return;
    m_impl->quality.store(visiong::mpp::clamp_quality(quality), std::memory_order_relaxed);
}

int DisplayHTTPFLV::get_quality() const {
    return m_impl ? m_impl->quality.load(std::memory_order_relaxed) : 0;
}

void DisplayHTTPFLV::set_rc_mode(RcMode mode) {
    if (!m_impl) return;
    m_impl->rc_mode.store(static_cast<int>(normalize_rc_mode(mode)), std::memory_order_relaxed);
}

DisplayHTTPFLV::RcMode DisplayHTTPFLV::get_rc_mode() const {
    return m_impl ? normalize_rc_mode(static_cast<RcMode>(m_impl->rc_mode.load(std::memory_order_relaxed)))
                  : RcMode::CBR;
}

bool DisplayHTTPFLV::display(const ImageBuffer& img) {
    if (!m_impl || !m_impl->running.load(std::memory_order_relaxed) || !img.is_valid()) return false;
    if (!m_impl->has_live_clients()) return true;
    if (m_impl->mpp_channel < 0) return false;

    ImageBuffer frame_copy = img.copy();
    if (!frame_copy.is_valid()) return false;
    {
        std::lock_guard<std::mutex> lk(m_impl->latest_mtx);
        m_impl->latest_frame = std::make_shared<ImageBuffer>(std::move(frame_copy));
        ++m_impl->latest_seq;
    }
    m_impl->latest_cv.notify_one();
    return true;
}
