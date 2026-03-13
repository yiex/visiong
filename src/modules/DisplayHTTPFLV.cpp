// SPDX-License-Identifier: LGPL-3.0-or-later
#include "visiong/modules/DisplayHTTPFLV.h"
#include "visiong/core/ImageBuffer.h"
#include "visiong/core/NetUtils.h"
#include "visiong/modules/VencManager.h"
#include "core/internal/logger.h"
#include "modules/internal/http_socket_utils.h"
#include "modules/internal/venc_utils.h"

#include <sstream>
#include <cstring>
#include <cstdio>
#include <vector>
#include <deque>
#include <list>
#include <mutex>
#include <thread>
#include <atomic>
#include <chrono>
#include <memory>
#include <condition_variable>

#include <unistd.h>
#include <errno.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>

extern "C" {
#include "flv-muxer.h"
#include "flv-writer.h"
#include "flv-header.h"
#include "flv-proto.h"
}

namespace {

static std::string build_html_player(const std::string& flv_path) {
    std::ostringstream os;
    os << "<!DOCTYPE html>\n"
          "<html><head><meta charset=\"utf-8\"/>\n"
          "<title>VisionG HTTP-FLV</title>\n"
          "<style>body{margin:0;background:#111;color:#eee;font-family:sans-serif}"
          "#v{width:100%;height:100vh;object-fit:contain;transform:translateZ(0);backface-visibility:hidden;}"
          "</style>\n"
          "</head><body>\n"
          "<video id=\"v\" autoplay muted playsinline controls></video>\n"
          "<script src=\"https://cdn.jsdelivr.net/npm/mpegts.js/dist/mpegts.js\"></script>\n"
          "<script>\n"
          "const v=document.getElementById('v');\n"
          "if (mpegts.getFeatureList().mseLivePlayback) {\n"
          "  const url=(location.origin|| (location.protocol+'//'+location.host)) + '" << flv_path << "';\n"
          "  const p=mpegts.createPlayer({type:'flv',url:url,isLive:true,lazyLoad:false,lazyLoadMaxDuration:1});\n"
          "  p.attachMediaElement(v); p.load(); p.play();\n"
          "} else { v.outerHTML='<p>Browser does not support MSE live playback.</p>'; }\n"
          "</script>\n"
          "</body></html>\n";
    return os.str();
}

/// Per-client streaming context / 每个客户端的流式上下文。
struct FLVClient {
    int fd = -1;
    std::atomic<bool> alive{true};
    void* flv_writer = nullptr;
    std::thread sender_thread;
    std::mutex queue_mtx;
    std::condition_variable queue_cv;
    std::deque<std::vector<uint8_t>> send_queue;
    size_t queued_bytes = 0;

    static constexpr size_t kMaxQueuedBytes = 2 * 1024 * 1024;
    static constexpr size_t kMaxQueuedPackets = 120;

    ~FLVClient() {
        stop_sender();
        if (flv_writer) { flv_writer_destroy(flv_writer); flv_writer = nullptr; }
        if (fd >= 0) { ::close(fd); fd = -1; }
    }

    bool enqueue_flv_chunk(const struct flv_vec_t* vec, int n) {
        if (!alive.load(std::memory_order_relaxed)) return false;
        if (!vec || n <= 0) return true;

        size_t total = 0;
        for (int i = 0; i < n; ++i) {
            if (vec[i].len > 0) total += (size_t)vec[i].len;
        }
        if (0 == total) return true;

        char chunk_hdr[32];
        int hdr_len = snprintf(chunk_hdr, sizeof(chunk_hdr), "%zx\r\n", total);
        if (hdr_len <= 0 || (size_t)hdr_len >= sizeof(chunk_hdr)) {
            alive.store(false, std::memory_order_relaxed);
            queue_cv.notify_all();
            return false;
        }

        const size_t packet_size = static_cast<size_t>(hdr_len) + total + 2;
        std::vector<uint8_t> packet;
        packet.reserve(packet_size);
        packet.insert(packet.end(), (const uint8_t*)chunk_hdr, (const uint8_t*)chunk_hdr + hdr_len);

        for (int i = 0; i < n; ++i) {
            if (vec[i].len <= 0) continue;
            const auto* begin = static_cast<const uint8_t*>(vec[i].ptr);
            packet.insert(packet.end(), begin, begin + vec[i].len);
        }
        packet.push_back('\r');
        packet.push_back('\n');

        {
            std::lock_guard<std::mutex> lk(queue_mtx);
            if (send_queue.size() >= kMaxQueuedPackets || queued_bytes + packet.size() > kMaxQueuedBytes) {
                alive.store(false, std::memory_order_relaxed);
                queue_cv.notify_all();
                return false;
            }
            queued_bytes += packet.size();
            send_queue.emplace_back(std::move(packet));
        }
        queue_cv.notify_one();
        return true;
    }

    static int on_flv_write(void* param, const struct flv_vec_t* vec, int n) {
        auto* c = (FLVClient*)param;
        if (!c || !c->alive.load(std::memory_order_relaxed)) return -1;
        return c->enqueue_flv_chunk(vec, n) ? 0 : -1;
    }

    void sender_loop() {
        while (true) {
            std::vector<uint8_t> packet;
            {
                std::unique_lock<std::mutex> lk(queue_mtx);
                queue_cv.wait(lk, [&]() {
                    return !alive.load(std::memory_order_relaxed) || !send_queue.empty();
                });

                if (send_queue.empty()) {
                    if (!alive.load(std::memory_order_relaxed)) break;
                    continue;
                }

                packet = std::move(send_queue.front());
                send_queue.pop_front();
                queued_bytes -= packet.size();
            }

            if (packet.empty()) continue;
            if (!visiong::http::send_all(fd, packet.data(), packet.size())) {
                alive.store(false, std::memory_order_relaxed);
                break;
            }
        }
        alive.store(false, std::memory_order_relaxed);
    }

    void stop_sender() {
        alive.store(false, std::memory_order_relaxed);
        queue_cv.notify_all();
        if (sender_thread.joinable()) sender_thread.join();
    }

    bool init_writer() {
        flv_writer = flv_writer_create2(0 /*no audio*/, 1 /*video*/, &FLVClient::on_flv_write, this);
        if (!flv_writer) return false;
        try {
            sender_thread = std::thread([this]() { sender_loop(); });
        } catch (...) {
            flv_writer_destroy(flv_writer);
            flv_writer = nullptr;
            return false;
        }
        return true;
    }
};

static DisplayHTTPFLV::RcMode normalize_rc_mode(DisplayHTTPFLV::RcMode mode) {
    return mode == DisplayHTTPFLV::RcMode::VBR ? DisplayHTTPFLV::RcMode::VBR : DisplayHTTPFLV::RcMode::CBR;
}

static VencRcMode to_venc_rc_mode(DisplayHTTPFLV::RcMode mode) {
    return normalize_rc_mode(mode) == DisplayHTTPFLV::RcMode::VBR ? VencRcMode::VBR : VencRcMode::CBR;
}

static void set_socket_send_timeout(int fd, int timeout_ms) {
    if (fd < 0 || timeout_ms <= 0) return;
    struct timeval tv;
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    (void)::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}

} // namespace

struct DisplayHTTPFLV::Impl {
    static constexpr size_t kMaxClients = 8;

    int port = 8080;
    std::string path = "/live.flv";

    std::atomic<int> codec{(int)Codec::H264};
    std::atomic<int> quality{75};
    std::atomic<int> rc_mode{(int)RcMode::CBR};
    std::atomic<int> max_fps{30};

    std::atomic<bool> running{false};
    std::atomic<bool> need_idr{false};
    std::atomic<bool> venc_user_acquired{false};
    int listen_fd = -1;
    std::thread accept_thread;

    std::mutex clients_mtx;
    std::list<std::shared_ptr<FLVClient>> clients;

    // FLV muxer (shared, one muxer produces FLV tags that are distributed to all clients) / 共享 FLV 复用器（单个 muxer 生成 FLV tag，并分发给所有客户端）。
    flv_muxer_t* muxer = nullptr;
    std::mutex muxer_mtx;

    // Codec config distribution: / 编解码配置分发逻辑：
    // - Increment config_seq when a new client connects OR codec_data changes
    // - In display(), if config_seq != applied_config_seq and we have last_codec_data, / - 在 display() 中，如果 config_seq != applied_config_seq 且已持有 last_codec_data，
    //   reset muxer and push codec_data once so all clients can start decoding.
    std::atomic<uint32_t> config_seq{1};
    uint32_t applied_config_seq = 0;
    std::mutex codec_mtx;
    std::vector<unsigned char> last_codec_data;

    // Frame rate limiter / 帧率限制器。
    std::mutex display_mtx;
    bool has_sent = false;
    std::chrono::steady_clock::time_point last_send = std::chrono::steady_clock::now();

    // Timestamp tracking for FLV (frame-based for stable PTS, reduces tearing) / FLV 时间戳跟踪（基于帧生成稳定 PTS，降低撕裂）。
    uint64_t frame_count = 0;

    void accept_loop();
    void cleanup_dead_locked();

    // Called by flv_muxer; distributes FLV tags to all connected clients / 由 flv_muxer 调用；负责把 FLV tag 分发给所有已连接客户端。
    static int on_muxer_output(void* param, int type, const void* data, size_t bytes, uint32_t timestamp) {
        auto* impl = (Impl*)param;
        if (!impl) return -1;

        // Snapshot to avoid holding clients_mtx while sending (send may block) / 先做快照，避免发送期间一直持有 clients_mtx（send 可能阻塞）。
        std::vector<std::shared_ptr<FLVClient>> snapshot;
        snapshot.reserve(8);
        {
            std::lock_guard<std::mutex> lk(impl->clients_mtx);
            for (auto& c : impl->clients) {
                if (c && c->alive.load(std::memory_order_relaxed)) snapshot.push_back(c);
            }
        }

        for (auto& c : snapshot) {
            if (!c || !c->alive.load(std::memory_order_relaxed)) continue;
            if (c->flv_writer) {
                int r = flv_writer_input(c->flv_writer, type, data, bytes, timestamp);
                if (r != 0) {
                    c->alive.store(false, std::memory_order_relaxed);
                }
            }
        }
        return 0;
    }
};

void DisplayHTTPFLV::Impl::cleanup_dead_locked() {
    for (auto it = clients.begin(); it != clients.end(); ) {
        auto& c = *it;
        if (!c || !c->alive.load(std::memory_order_relaxed)) {
            it = clients.erase(it);
        } else {
            ++it;
        }
    }
}

void DisplayHTTPFLV::Impl::accept_loop() {
    while (running.load(std::memory_order_relaxed)) {
        sockaddr_in addr{};
        socklen_t len = sizeof(addr);
        int cfd = ::accept(listen_fd, (sockaddr*)&addr, &len);
        if (cfd < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
            if (!running.load(std::memory_order_relaxed)) break;
            continue;
        }

        // Read HTTP request (consume until \r\n\r\n, with size cap) / 读取 HTTP request (consume until \r\n\r\n, 与 尺寸 cap)
        {
            struct timeval rtv;
            rtv.tv_sec = 1;
            rtv.tv_usec = 0;
            ::setsockopt(cfd, SOL_SOCKET, SO_RCVTIMEO, &rtv, sizeof(rtv));

            std::string req;
            if (!visiong::http::read_http_request(cfd, req)) { ::close(cfd); continue; }
            std::string req_path;
            if (!visiong::http::parse_http_request_path(req, req_path)) {
                const char* resp400 = "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n";
                visiong::http::send_all(cfd, resp400, strlen(resp400));
                ::close(cfd);
                continue;
            }

            // Path match (allow trailing '?' query string) / 路径匹配（允许尾随 ? 查询串）。
            size_t qm = req_path.find('?');
            std::string req_clean = (qm != std::string::npos) ? req_path.substr(0, qm) : req_path;

            // Serve a minimal HTML player at / / 在 / 路径提供一个最小化 HTML 播放页。
            if (req_clean == "/" || req_clean == "/index.html") {
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

            if (req_clean != path && req_clean != path + "/") {
                const char* resp404 = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n";
                visiong::http::send_all(cfd, resp404, strlen(resp404));
                ::close(cfd);
                continue;
            }
        }

        // Send HTTP response header / 发送 HTTP 响应头。
        const char* resp =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: video/x-flv\r\n"
            "Connection: close\r\n"
            "Transfer-Encoding: chunked\r\n"
            "Cache-Control: no-cache\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "\r\n";

        if (!visiong::http::send_all(cfd, resp, strlen(resp))) {
            ::close(cfd);
            continue;
        }

        // Disable Nagle for lower latency / 关闭 Nagle 算法以降低时延。
        int nodelay = 1;
        ::setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
        // Per-client sender thread uses blocking sends with timeout to avoid producer stalls. / 每个客户端的发送线程都使用带超时的阻塞发送，避免生产者卡死。
        set_socket_send_timeout(cfd, 1000);

        auto client = std::make_shared<FLVClient>();
        client->fd = cfd;
        if (!client->init_writer()) {
            ::close(cfd);
            client->fd = -1;
            continue;
        }

        {
            std::lock_guard<std::mutex> lk(clients_mtx);
            cleanup_dead_locked();
            if (clients.size() >= kMaxClients) {
                const char* resp503 =
                    "HTTP/1.1 503 Service Unavailable\r\n"
                    "Content-Length: 0\r\n"
                    "Connection: close\r\n\r\n";
                visiong::http::send_all(cfd, resp503, strlen(resp503));
                ::close(cfd);
                VISIONG_LOG_WARN("DisplayHTTPFLV", "Client rejected: too many active clients.");
                continue;
            }
            clients.push_back(client);
        }
        // New client needs codec config/sequence header / 新客户端需要先收到编解码配置/序列头。
        config_seq.fetch_add(1, std::memory_order_relaxed);
        need_idr.store(true, std::memory_order_relaxed);

        VISIONG_LOG_INFO("DisplayHTTPFLV", "Client connected.");
    }
}

// --- Public API --- / --- 公共 API ---

DisplayHTTPFLV::DisplayHTTPFLV(int port, const std::string& path, int quality, Codec codec, int fps, RcMode rc_mode)
    : m_impl(std::make_unique<Impl>()) {
    m_impl->port = port;
    m_impl->path = path.empty() ? "/live.flv" : path;
    if (m_impl->path[0] != '/') m_impl->path = "/" + m_impl->path;
    m_impl->codec.store((int)codec, std::memory_order_relaxed);
    m_impl->quality.store(visiong::venc::clamp_quality(quality), std::memory_order_relaxed);
    m_impl->max_fps.store(visiong::venc::clamp_non_negative_fps(fps), std::memory_order_relaxed);
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
    addr.sin_port = htons((uint16_t)m_impl->port);

    if (::bind(fd, (sockaddr*)&addr, sizeof(addr)) != 0) {
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
    m_impl->frame_count = 0;
    try {
        m_impl->accept_thread = std::thread([this]() { m_impl->accept_loop(); });
    } catch (...) {
        ::close(fd);
        m_impl->listen_fd = -1;
        stop();
        throw;
    }

    Codec c = (Codec)m_impl->codec.load();
    VISIONG_LOG_INFO("DisplayHTTPFLV", "Server started.");
    VISIONG_LOG_INFO("DisplayHTTPFLV",
                     "  Codec: " << (c == Codec::H264 ? "H264" : "H265")
                                 << "  Quality: " << m_impl->quality.load(std::memory_order_relaxed)
                                 << "  FPS: " << m_impl->max_fps.load(std::memory_order_relaxed));
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
    m_impl->venc_user_acquired.store(false, std::memory_order_relaxed);
    m_impl->need_idr.store(false, std::memory_order_relaxed);

    m_impl->muxer = flv_muxer_create(&Impl::on_muxer_output, m_impl.get());
    if (!m_impl->muxer) throw std::runtime_error("DisplayHTTPFLV: flv_muxer_create() failed.");
    flv_muxer_set_enhanced_rtmp(m_impl->muxer, 1);
    m_impl->config_seq.store(1, std::memory_order_relaxed);
    m_impl->applied_config_seq = 0;
    {
        std::lock_guard<std::mutex> lk(m_impl->codec_mtx);
        m_impl->last_codec_data.clear();
    }

    m_impl->listen_fd = -1;
    m_impl->frame_count = 0;
    m_impl->running.store(true, std::memory_order_relaxed);
}

void DisplayHTTPFLV::add_stream_client(int client_fd) {
    if (!m_impl || !m_impl->running.load(std::memory_order_relaxed)) {
        ::close(client_fd);
        return;
    }
    const char* resp =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: video/x-flv\r\n"
        "Connection: close\r\n"
        "Transfer-Encoding: chunked\r\n"
        "Cache-Control: no-cache\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "\r\n";

    if (!visiong::http::send_all(client_fd, resp, strlen(resp))) {
        ::close(client_fd);
        return;
    }

    int nodelay = 1;
    ::setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
    set_socket_send_timeout(client_fd, 1000);

    auto client = std::make_shared<FLVClient>();
    client->fd = client_fd;
    if (!client->init_writer()) {
        ::close(client_fd);
        client->fd = -1;
        return;
    }

    {
        std::lock_guard<std::mutex> lk(m_impl->clients_mtx);
        m_impl->cleanup_dead_locked();
        if (m_impl->clients.size() >= Impl::kMaxClients) {
            const char* resp503 =
                "HTTP/1.1 503 Service Unavailable\r\n"
                "Content-Length: 0\r\n"
                "Connection: close\r\n\r\n";
            visiong::http::send_all(client_fd, resp503, strlen(resp503));
            ::close(client_fd);
            VISIONG_LOG_WARN("DisplayHTTPFLV", "Client rejected: too many active clients.");
            return;
        }
        m_impl->clients.push_back(client);
    }
    m_impl->config_seq.fetch_add(1, std::memory_order_relaxed);
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
    if (!m_impl->running.load(std::memory_order_relaxed)) return;

    m_impl->running.store(false, std::memory_order_relaxed);

    if (m_impl->listen_fd >= 0) {
        ::shutdown(m_impl->listen_fd, SHUT_RDWR);
        ::close(m_impl->listen_fd);
        m_impl->listen_fd = -1;
    }

    if (m_impl->accept_thread.joinable()) m_impl->accept_thread.join();

    {
        std::lock_guard<std::mutex> lk(m_impl->clients_mtx);
        for (auto& c : m_impl->clients) {
            if (c) {
                c->alive.store(false, std::memory_order_relaxed);
                if (c->fd >= 0) ::shutdown(c->fd, SHUT_RDWR);
            }
        }
        m_impl->clients.clear();
    }

    {
        std::lock_guard<std::mutex> lk(m_impl->muxer_mtx);
        if (m_impl->muxer) {
            flv_muxer_destroy(m_impl->muxer);
            m_impl->muxer = nullptr;
        }
        m_impl->applied_config_seq = 0;
    }
    {
        std::lock_guard<std::mutex> lk(m_impl->codec_mtx);
        m_impl->last_codec_data.clear();
    }

    if (m_impl->venc_user_acquired.exchange(false, std::memory_order_relaxed)) {
        VencManager::getInstance().releaseUser();
    }
    VencManager::getInstance().releaseVencIfUnused();
    VISIONG_LOG_INFO("DisplayHTTPFLV", "Server stopped.");
}

bool DisplayHTTPFLV::is_running() const {
    return m_impl && m_impl->running.load(std::memory_order_relaxed);
}

void DisplayHTTPFLV::set_fps(int fps) {
    if (!m_impl) return;
    fps = visiong::venc::clamp_non_negative_fps(fps);
    m_impl->max_fps.store(fps, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(m_impl->display_mtx);
    m_impl->has_sent = false;
    m_impl->last_send = std::chrono::steady_clock::now();
}

int DisplayHTTPFLV::get_fps() const {
    return m_impl ? m_impl->max_fps.load(std::memory_order_relaxed) : 0;
}

void DisplayHTTPFLV::set_quality(int quality) {
    if (!m_impl) return;
    quality = visiong::venc::clamp_quality(quality);
    m_impl->quality.store(quality, std::memory_order_relaxed);
}

int DisplayHTTPFLV::get_quality() const {
    return m_impl ? m_impl->quality.load(std::memory_order_relaxed) : 0;
}

void DisplayHTTPFLV::set_rc_mode(RcMode mode) {
    if (!m_impl) return;
    m_impl->rc_mode.store(static_cast<int>(normalize_rc_mode(mode)), std::memory_order_relaxed);
}

DisplayHTTPFLV::RcMode DisplayHTTPFLV::get_rc_mode() const {
    return m_impl ? normalize_rc_mode(static_cast<RcMode>(m_impl->rc_mode.load(std::memory_order_relaxed))) : RcMode::CBR;
}

bool DisplayHTTPFLV::display(const ImageBuffer& img) {
    if (!m_impl || !m_impl->running.load(std::memory_order_relaxed)) return false;
    if (!img.is_valid()) return false;

    // Fast path: skip encoding if no clients / 快速路径：没有客户端时跳过编码。
    bool has_clients = false;
    {
        std::lock_guard<std::mutex> lk(m_impl->clients_mtx);
        m_impl->cleanup_dead_locked();
        has_clients = !m_impl->clients.empty();
    }
    if (!has_clients) {
        if (m_impl->venc_user_acquired.exchange(false, std::memory_order_relaxed)) {
            VencManager::getInstance().releaseUser();
        }
        return true;
    }
    if (!m_impl->venc_user_acquired.exchange(true, std::memory_order_relaxed)) {
        VencManager::getInstance().acquireUser();
    }

    // FPS limiter / FPS 限速器。
    {
        std::lock_guard<std::mutex> lk(m_impl->display_mtx);
        int fps = m_impl->max_fps.load(std::memory_order_relaxed);
        if (fps > 0) {
            auto now = std::chrono::steady_clock::now();
            if (m_impl->has_sent) {
                auto interval = std::chrono::milliseconds(1000 / fps);
                if (now - m_impl->last_send < interval) return true;
            }
            m_impl->last_send = now;
            m_impl->has_sent = true;
        }
    }

    // Encode / 编码。
    Codec codec = (Codec)m_impl->codec.load(std::memory_order_relaxed);
    VencCodec venc_codec = (codec == Codec::H265) ? VencCodec::H265 : VencCodec::H264;
    int quality = m_impl->quality.load(std::memory_order_relaxed);
    int fps_enc = visiong::venc::clamp_non_negative_fps(m_impl->max_fps.load(std::memory_order_relaxed));
    VencRcMode venc_rc = to_venc_rc_mode(static_cast<RcMode>(m_impl->rc_mode.load(std::memory_order_relaxed)));

    if (m_impl->need_idr.exchange(false, std::memory_order_relaxed)) {
        (void)VencManager::getInstance().requestIDR(true);
    }

    VencEncodedPacket packet;
    if (!VencManager::getInstance().encodeToVideo(img, venc_codec, quality, packet, fps_enc, venc_rc))
        return false;
    if (packet.data.empty()) return false;

    // Track codec config (SPS/PPS/VPS) and trigger config re-send when it changes / 跟踪编解码配置（SPS/PPS/VPS），变化时触发重新下发。
    if (!packet.codec_data.empty()) {
        bool changed = false;
        {
            std::lock_guard<std::mutex> lk(m_impl->codec_mtx);
            if (m_impl->last_codec_data != packet.codec_data) {
                m_impl->last_codec_data = packet.codec_data;
                changed = true;
            }
        }
        if (changed) m_impl->config_seq.fetch_add(1, std::memory_order_relaxed);
        if (changed) m_impl->need_idr.store(true, std::memory_order_relaxed);
    }

    // Stable frame-based timestamp (regular PTS reduces decoder/display tearing) / 稳定的基于帧的时间戳（规则 PTS 可降低解码/显示撕裂）。
    int target_fps = (fps_enc > 0) ? fps_enc : 30;
    m_impl->frame_count++;
    uint32_t ts_ms = (uint32_t)((m_impl->frame_count * 1000) / (unsigned)target_fps);

    {
        std::lock_guard<std::mutex> lk(m_impl->muxer_mtx);
        if (!m_impl->muxer) return false;

        // If needed, reset muxer and push codec config so late-joining clients can decode / 必要时重置 muxer 并推送 codec 配置，让后加入的客户端也能解码。
        uint32_t seq = m_impl->config_seq.load(std::memory_order_relaxed);
        std::vector<unsigned char> codec_snapshot;
        {
            std::lock_guard<std::mutex> lk2(m_impl->codec_mtx);
            codec_snapshot = m_impl->last_codec_data;
        }
        if (seq != m_impl->applied_config_seq && !codec_snapshot.empty()) {
            flv_muxer_reset(m_impl->muxer);
            flv_muxer_set_enhanced_rtmp(m_impl->muxer, 1);

            // Send basic metadata (width/height/fps) once per reset / 每次重置后发送一次基础元数据（宽/高/fps）。
            flv_metadata_t meta;
            memset(&meta, 0, sizeof(meta));
            meta.videocodecid = (codec == Codec::H264) ? FLV_VIDEO_H264 : FLV_VIDEO_H265;
            meta.width = VencManager::getInstance().getWidth();
            meta.height = VencManager::getInstance().getHeight();
            meta.framerate = (double)((fps_enc > 0) ? fps_enc : 30);
            (void)flv_muxer_metadata(m_impl->muxer, &meta);

            // Push codec config (SPS/PPS/VPS) to generate sequence header / 推送 codec 配置（SPS/PPS/VPS）以生成 sequence header。
            int rcfg = 0;
            if (codec == Codec::H264) {
                rcfg = flv_muxer_avc(m_impl->muxer, codec_snapshot.data(), codec_snapshot.size(), ts_ms, ts_ms);
            } else {
                rcfg = flv_muxer_hevc(m_impl->muxer, codec_snapshot.data(), codec_snapshot.size(), ts_ms, ts_ms);
            }
            if (rcfg != 0) {
                // Don't apply seq on failure; try again next frame / 失败时不要提交 seq，下一帧再重试。
            } else {
                m_impl->applied_config_seq = seq;
            }
        }

        int r = 0;
        if (codec == Codec::H264) {
            r = flv_muxer_avc(m_impl->muxer, packet.data.data(), packet.data.size(), ts_ms, ts_ms);
        } else {
            r = flv_muxer_hevc(m_impl->muxer, packet.data.data(), packet.data.size(), ts_ms, ts_ms);
        }
        if (r != 0) return false;
    }

    // Post-mux cleanup / mux 后清理。
    {
        std::lock_guard<std::mutex> lk(m_impl->clients_mtx);
        m_impl->cleanup_dead_locked();
    }

    return true;
}

