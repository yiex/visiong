// SPDX-License-Identifier: LGPL-3.0-or-later
#include "visiong/modules/DisplayHTTP.h"
#include "visiong/modules/DisplayHTTPFLV.h"
#include "visiong/common/pixel_format.h"
#include "visiong/core/ImageBuffer.h"
#include "visiong/modules/VencManager.h"
#include "visiong/core/NetUtils.h"
#include "common/internal/string_utils.h"
#include "core/internal/logger.h"
#include "modules/internal/http_socket_utils.h"
#include "modules/internal/jpeg_lock_utils.h"
#include "modules/internal/venc_utils.h"

#include <sstream>
#include <cstring>
#include <stdexcept>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <vector>
#include <thread>
#include <mutex>
#include <memory>
#include <list>
#include <condition_variable>
#include <chrono>
#include <atomic>
#include <cstdint>

namespace {
static std::string normalize_flv_codec(const std::string& codec) {
    const std::string normalized = visiong::to_lower_copy(codec);
    if (normalized == "h265" || normalized == "265" || normalized == "hevc") {
        return "h265";
    }
    return "h264";
}

static std::string normalize_flv_rc_mode(const std::string& rc_mode) {
    try {
        return visiong::venc::normalize_rc_mode(rc_mode);
    } catch (const std::invalid_argument&) {
        return "cbr";
    }
}

static DisplayHTTPFLV::Codec to_flv_codec_enum(const std::string& codec) {
    return normalize_flv_codec(codec) == "h265"
               ? DisplayHTTPFLV::Codec::H265
               : DisplayHTTPFLV::Codec::H264;
}

static DisplayHTTPFLV::RcMode to_flv_rc_mode_enum(const std::string& rc_mode) {
    return normalize_flv_rc_mode(rc_mode) == "vbr"
               ? DisplayHTTPFLV::RcMode::VBR
               : DisplayHTTPFLV::RcMode::CBR;
}
} // namespace

struct DisplayHTTP::ClientThread {
    std::atomic<int> socket{-1};
    std::atomic<bool> done{false};
    std::thread thread;
};

struct DisplayHTTP::Impl {
    int port = 8080;
    int flv_fps = 30;
    int server_socket = -1;
    std::atomic<bool> is_running{false};
    std::atomic<int> quality{75};
    std::atomic<int> max_fps{30};
    std::atomic<int> client_count{0};
    std::atomic<bool> venc_user_acquired{false};
    std::string stream_type = "jpg";
    std::string flv_path = "/live.flv";
    std::string flv_codec = "h264";
    std::string flv_rc_mode = "cbr";
    std::unique_ptr<DisplayHTTPFLV> flv;
    std::thread server_thread;
    std::mutex client_mutex;
    std::list<ClientThread> client_threads;
    std::mutex frame_mutex;
    std::condition_variable frame_cv;
    std::shared_ptr<const std::vector<unsigned char>> latest_jpeg_frame;
    uint64_t frame_id = 0;
    std::mutex display_mutex;
    std::mutex jpeg_lock_mutex;
    bool has_sent = false;
    PIXEL_FORMAT_E jpeg_locked_format = RK_FMT_BUTT;
    int jpeg_locked_width = 0;
    int jpeg_locked_height = 0;
    int jpeg_locked_input_priority = -1;
    std::chrono::steady_clock::time_point last_send = std::chrono::steady_clock::now();
};

DisplayHTTP::DisplayHTTP(int port, int quality,
                         const std::string& mode,
                         const std::string& flv_path,
                         const std::string& flv_codec,
                         int flv_fps,
                         const std::string& flv_rc_mode)
    : m_impl(std::make_unique<Impl>()) {
    m_impl->port = port;
    m_impl->flv_fps = visiong::venc::clamp_non_negative_fps(flv_fps);
    m_impl->max_fps.store(m_impl->flv_fps, std::memory_order_relaxed);
    m_impl->quality.store(visiong::venc::clamp_quality(quality), std::memory_order_relaxed);
    m_impl->stream_type = visiong::to_lower_copy(mode);
    if (m_impl->stream_type != "flv") m_impl->stream_type = "jpg";

    m_impl->flv_path = flv_path.empty() ? "/live.flv" : flv_path;
    if (m_impl->flv_path[0] != '/') m_impl->flv_path = "/" + m_impl->flv_path;

    m_impl->flv_codec = normalize_flv_codec(flv_codec);
    m_impl->flv_rc_mode = normalize_flv_rc_mode(flv_rc_mode);

    signal(SIGPIPE, SIG_IGN);
}

DisplayHTTP::~DisplayHTTP() {
    stop();
}

void DisplayHTTP::start() {
    if (m_impl->is_running) {
        VISIONG_LOG_INFO("DisplayHTTP", "Server is already running.");
        return;
    }

    m_impl->client_count.store(0, std::memory_order_relaxed);
    m_impl->venc_user_acquired.store(false, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> frame_lock(m_impl->frame_mutex);
        m_impl->latest_jpeg_frame.reset();
        m_impl->frame_id = 0;
    }
    {
        std::lock_guard<std::mutex> lk(m_impl->display_mutex);
        m_impl->has_sent = false;
        m_impl->last_send = std::chrono::steady_clock::now();
    }
    {
        std::lock_guard<std::mutex> lock(m_impl->jpeg_lock_mutex);
        m_impl->jpeg_locked_format = RK_FMT_BUTT;
        m_impl->jpeg_locked_width = 0;
        m_impl->jpeg_locked_height = 0;
        m_impl->jpeg_locked_input_priority = -1;
    }

    m_impl->server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (m_impl->server_socket < 0) {
        throw std::runtime_error("Failed to create server socket.");
    }

    try {
        int opt = 1;
        setsockopt(m_impl->server_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        sockaddr_in server_addr;
        memset(&server_addr, 0, sizeof(server_addr));
        server_addr.sin_family = AF_INET;
        server_addr.sin_addr.s_addr = htonl(INADDR_ANY);
        server_addr.sin_port = htons(m_impl->port);

        if (bind(m_impl->server_socket, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            throw std::runtime_error("Failed to bind server socket to port " + std::to_string(m_impl->port));
        }

        if (listen(m_impl->server_socket, 5) < 0) {
            throw std::runtime_error("Failed to listen on server socket.");
        }

        if (m_impl->stream_type == "flv") {
            const DisplayHTTPFLV::Codec codec_enum = to_flv_codec_enum(m_impl->flv_codec);
            const DisplayHTTPFLV::RcMode rc_enum = to_flv_rc_mode_enum(m_impl->flv_rc_mode);
            m_impl->flv = std::make_unique<DisplayHTTPFLV>(
                m_impl->port, m_impl->flv_path, m_impl->quality.load(), codec_enum, m_impl->flv_fps, rc_enum);
            m_impl->flv->start_encoder_only();
        }

        m_impl->is_running = true;
        m_impl->server_thread = std::thread(&DisplayHTTP::server_loop, this);
    } catch (...) {
        m_impl->is_running = false;
        if (m_impl->server_socket != -1) {
            shutdown(m_impl->server_socket, SHUT_RDWR);
            close(m_impl->server_socket);
            m_impl->server_socket = -1;
        }
        if (m_impl->flv) {
            m_impl->flv->stop();
            m_impl->flv.reset();
        }
        throw;
    }
    VISIONG_LOG_INFO("DisplayHTTP", "Server started (" << (m_impl->stream_type == "flv" ? "FLV" : "MJPEG")
                                                       << "). View in browser at:");
    auto ips = visiong::get_local_ipv4_addresses();
    if (ips.empty()) {
        VISIONG_LOG_INFO("DisplayHTTP", "  > http://<device-ip>:" << m_impl->port);
    } else {
        for (const auto& ip : ips) {
            VISIONG_LOG_INFO("DisplayHTTP", "  > http://" << ip << ":" << m_impl->port);
        }
    }
}

void DisplayHTTP::stop() {
    if (!m_impl->is_running && m_impl->server_socket == -1 && !m_impl->server_thread.joinable() && !m_impl->flv) return;

    m_impl->is_running = false;

    // Close server socket to unblock accept(), including partial-start states. / 关闭服务器套接字以解除 accept() 阻塞，包括部分启动状态。
    if (m_impl->server_socket != -1) {
        shutdown(m_impl->server_socket, SHUT_RDWR);
        close(m_impl->server_socket);
        m_impl->server_socket = -1;
    }

    // Wake all clients waiting for new frame data. / 唤醒所有等待新帧数据的客户端。
    m_impl->frame_cv.notify_all();

    if (m_impl->server_thread.joinable()) {
        m_impl->server_thread.join();
    }

    std::list<ClientThread> threads_to_join;
    {
        std::lock_guard<std::mutex> lock(m_impl->client_mutex);
        for (auto& t : m_impl->client_threads) {
            const int sock = t.socket.exchange(-1, std::memory_order_relaxed);
            if (sock >= 0) {
                ::shutdown(sock, SHUT_RDWR);
            }
        }
        threads_to_join.splice(threads_to_join.end(), m_impl->client_threads);
    }
    for (auto& t : threads_to_join) {
        if (t.thread.joinable()) {
            t.thread.join();
        }
    }
    if (m_impl->venc_user_acquired.exchange(false, std::memory_order_relaxed)) {
        VencManager::getInstance().releaseUser();
    }
    VencManager::getInstance().releaseVencIfUnused();

    if (m_impl->flv) {
        m_impl->flv->stop();
        m_impl->flv.reset();
    }
    {
        std::lock_guard<std::mutex> lock(m_impl->jpeg_lock_mutex);
        m_impl->jpeg_locked_format = RK_FMT_BUTT;
        m_impl->jpeg_locked_width = 0;
        m_impl->jpeg_locked_height = 0;
        m_impl->jpeg_locked_input_priority = -1;
    }

    VISIONG_LOG_INFO("DisplayHTTP", "Server stopped.");
}

bool DisplayHTTP::is_running() const {
    return m_impl->is_running;
}

bool DisplayHTTP::display(const ImageBuffer& img) {
    if (!m_impl->is_running || !img.is_valid()) {
        return false;
    }

    if (m_impl->stream_type == "flv" && m_impl->flv) {
        return m_impl->flv->display(img);
    }

    if (m_impl->client_count.load(std::memory_order_relaxed) <= 0) {
        if (m_impl->venc_user_acquired.exchange(false, std::memory_order_relaxed)) {
            VencManager::getInstance().releaseUser();
        }
        return true;
    }
    if (!m_impl->venc_user_acquired.exchange(true, std::memory_order_relaxed)) {
        VencManager::getInstance().acquireUser();
    }

    {
        std::lock_guard<std::mutex> lk(m_impl->display_mutex);
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

    VencManager& venc = VencManager::getInstance();
    const int q = m_impl->quality.load(std::memory_order_relaxed);
    bool use_sw = false;
    if (venc.isInitialized() && venc.getCodec() != VencCodec::JPEG) {
        use_sw = true; // VENC is busy with H264/H265
    }

    PIXEL_FORMAT_E locked_format = RK_FMT_BUTT;
    int locked_width = 0;
    int locked_height = 0;
    {
        std::lock_guard<std::mutex> lock(m_impl->jpeg_lock_mutex);
        const int input_priority = visiong::jpeg_lock::get_color_priority(img.format);
        const PIXEL_FORMAT_E requested_lock_format = visiong::jpeg_lock::choose_lock_format(img.format);

        if (m_impl->jpeg_locked_format == RK_FMT_BUTT || input_priority > m_impl->jpeg_locked_input_priority) {
            const PIXEL_FORMAT_E previous_format = m_impl->jpeg_locked_format;
            m_impl->jpeg_locked_format = requested_lock_format;
            m_impl->jpeg_locked_input_priority = input_priority;
            if (m_impl->jpeg_locked_width > 0 && m_impl->jpeg_locked_height > 0) {
                visiong::jpeg_lock::normalize_lock_size(
                    m_impl->jpeg_locked_format, &m_impl->jpeg_locked_width, &m_impl->jpeg_locked_height);
            }
            VISIONG_LOG_INFO("DisplayHTTP",
                             "MJPEG color lock updated: "
                                 << (previous_format == RK_FMT_BUTT ? "unset" : visiong::pixel_format_name(previous_format))
                                 << " -> " << visiong::pixel_format_name(m_impl->jpeg_locked_format)
                                 << " (priority " << input_priority << ")");
        }

        int candidate_width = img.width;
        int candidate_height = img.height;
        visiong::jpeg_lock::normalize_lock_size(m_impl->jpeg_locked_format, &candidate_width, &candidate_height);
        if (m_impl->jpeg_locked_width <= 0 || m_impl->jpeg_locked_height <= 0 ||
            visiong::jpeg_lock::should_expand_lock_size(
                m_impl->jpeg_locked_width, m_impl->jpeg_locked_height, candidate_width, candidate_height)) {
            const int previous_width = m_impl->jpeg_locked_width;
            const int previous_height = m_impl->jpeg_locked_height;
            m_impl->jpeg_locked_width = std::max(m_impl->jpeg_locked_width, candidate_width);
            m_impl->jpeg_locked_height = std::max(m_impl->jpeg_locked_height, candidate_height);
            visiong::jpeg_lock::normalize_lock_size(
                m_impl->jpeg_locked_format, &m_impl->jpeg_locked_width, &m_impl->jpeg_locked_height);
            VISIONG_LOG_INFO("DisplayHTTP",
                             "MJPEG size lock updated: "
                                 << previous_width << "x" << previous_height
                                 << " -> " << m_impl->jpeg_locked_width << "x" << m_impl->jpeg_locked_height);
        }

        locked_format = m_impl->jpeg_locked_format;
        locked_width = m_impl->jpeg_locked_width;
        locked_height = m_impl->jpeg_locked_height;
    }

    const ImageBuffer* encode_buf = &img;
    ImageBuffer converted_owner;
    ImageBuffer padded_owner;
    if ((locked_format == RK_FMT_YUV420SP && img.format != RK_FMT_YUV420SP) ||
        (locked_format != RK_FMT_YUV420SP && img.format != locked_format)) {
        converted_owner = img.to_format(locked_format);
        encode_buf = &converted_owner;
    } else if (locked_format == RK_FMT_YUV420SP && img.format == RK_FMT_YUV420SP_VU) {
        converted_owner = img.to_format(RK_FMT_YUV420SP);
        encode_buf = &converted_owner;
    }
    if (encode_buf->width != locked_width || encode_buf->height != locked_height) {
        padded_owner = visiong::jpeg_lock::pad_frame_without_scaling(*encode_buf,
                                                                     locked_format,
                                                                     locked_width,
                                                                     locked_height);
        encode_buf = &padded_owner;
    }

    std::vector<unsigned char> jpeg_data;
    if (!use_sw) {
        if (!m_impl->venc_user_acquired.exchange(true, std::memory_order_relaxed)) {
            venc.acquireUser();
        }
        jpeg_data = venc.encodeToJpeg(*encode_buf, q);
        if (jpeg_data.empty()) {
            // Hardware JPEG failed; fallback to software / 硬件 JPEG 失败，回退到软件路径。
            use_sw = true;
            if (m_impl->venc_user_acquired.exchange(false, std::memory_order_relaxed)) {
                venc.releaseUser();
                venc.releaseVencIfUnused();
            }
        }
    }

    if (use_sw) {
        ImageBuffer bgr = encode_buf->to_format(RK_FMT_BGR888);
        const int stride_pixels = (bgr.w_stride > 0) ? bgr.w_stride : bgr.width;
        const int step = stride_pixels * 3;
        cv::Mat mat(bgr.height, bgr.width, CV_8UC3, bgr.get_data(), step);
        std::vector<unsigned char> sw_jpeg;
        std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, q};
        if (!cv::imencode(".jpg", mat, sw_jpeg, params) || sw_jpeg.empty()) {
            VISIONG_LOG_WARN("DisplayHTTP", "Failed to encode image (SW JPEG fallback).");
            return false;
        }
        jpeg_data = std::move(sw_jpeg);
    }

    auto shared_jpeg = std::make_shared<const std::vector<unsigned char>>(std::move(jpeg_data));
    {
        std::lock_guard<std::mutex> lock(m_impl->frame_mutex);
        m_impl->latest_jpeg_frame = std::move(shared_jpeg);
        m_impl->frame_id++;
    }

    m_impl->frame_cv.notify_all();
    return true;
}

void DisplayHTTP::serve_flv_index(int client_socket) {
    if (!m_impl->flv) { ::close(client_socket); return; }
    std::string html = m_impl->flv->get_index_html();
    std::ostringstream hdr;
    hdr << "HTTP/1.1 200 OK\r\n"
        << "Content-Type: text/html; charset=utf-8\r\n"
        << "Content-Length: " << html.size() << "\r\n"
        << "Cache-Control: no-cache\r\n"
        << "Connection: close\r\n\r\n";
    visiong::http::send_all(client_socket, hdr.str().c_str(), hdr.str().length());
    visiong::http::send_all(client_socket, html.data(), html.size());
    ::close(client_socket);
}

void DisplayHTTP::server_loop() {
    while (m_impl->is_running) {
        sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_socket = accept(m_impl->server_socket, (struct sockaddr*)&client_addr, &client_len);

        if (client_socket < 0) {
            if (m_impl->is_running) {
                VISIONG_LOG_WARN("DisplayHTTP", "Accept failed.");
            }
            continue;
        }

        if (m_impl->stream_type == "flv") {
            struct timeval rtv;
            rtv.tv_sec = 1;
            rtv.tv_usec = 0;
            ::setsockopt(client_socket, SOL_SOCKET, SO_RCVTIMEO, &rtv, sizeof(rtv));
            std::string req;
            if (!visiong::http::read_http_request(client_socket, req)) {
                ::close(client_socket);
                continue;
            }
            std::string req_path;
            if (!visiong::http::parse_http_request_path(req, req_path)) {
                const char* resp400 = "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n";
                visiong::http::send_all(client_socket, resp400, strlen(resp400));
                ::close(client_socket);
                continue;
            }
            size_t qm = req_path.find('?');
            std::string path_clean = (qm != std::string::npos) ? req_path.substr(0, qm) : req_path;
            if (path_clean == "/" || path_clean == "/index.html") {
                serve_flv_index(client_socket);
                continue;
            }
            if (path_clean != m_impl->flv_path && path_clean != m_impl->flv_path + "/") {
                const char* resp404 = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n";
                visiong::http::send_all(client_socket, resp404, strlen(resp404));
                ::close(client_socket);
                continue;
            }
            if (m_impl->flv) {
                m_impl->flv->add_stream_client(client_socket);
            } else {
                ::close(client_socket);
            }
            continue;
        }

        VISIONG_LOG_INFO("DisplayHTTP", "New client connected.");
        std::lock_guard<std::mutex> lock(m_impl->client_mutex);
        m_impl->client_count.fetch_add(1, std::memory_order_relaxed);
        m_impl->client_threads.emplace_back();
        ClientThread& slot = m_impl->client_threads.back();
        slot.socket.store(client_socket, std::memory_order_relaxed);
        slot.done.store(false, std::memory_order_relaxed);
        slot.thread = std::thread(&DisplayHTTP::client_handler, this, client_socket, &slot);

        for (auto it = m_impl->client_threads.begin(); it != m_impl->client_threads.end(); ) {
            if (it->done.load(std::memory_order_relaxed)) {
                if (it->thread.joinable()) it->thread.join();
                it = m_impl->client_threads.erase(it);
            } else {
                ++it;
            }
        }
    }
}

void DisplayHTTP::client_handler(int client_socket, ClientThread* client_state) {
    // Keep a bounded send timeout so stop() can reliably join client threads. / 保持有界发送超时，确保 stop() 能可靠地回收客户端线程。
    struct timeval stv;
    stv.tv_sec = 1;
    stv.tv_usec = 0;
    ::setsockopt(client_socket, SOL_SOCKET, SO_SNDTIMEO, &stv, sizeof(stv));

    auto close_client_socket = [&]() {
        int sock = client_socket;
        if (client_state) {
            const int tracked = client_state->socket.exchange(-1, std::memory_order_relaxed);
            if (tracked >= 0) {
                sock = tracked;
            }
        }
        if (sock >= 0) {
            ::close(sock);
        }
    };

    auto mark_client_done = [&]() {
        m_impl->client_count.fetch_sub(1, std::memory_order_relaxed);
        if (client_state) {
            client_state->done.store(true, std::memory_order_relaxed);
        }
    };

    // MJPEG path: keep legacy behavior. Do not parse request. / MJPEG 路径：保持历史行为，不解析请求。
    std::string header = "HTTP/1.1 200 OK\r\n"
                         "Content-Type: multipart/x-mixed-replace; boundary=--frame\r\n"
                         "Connection: close\r\n"
                         "Cache-Control: no-store, no-cache, must-revalidate, pre-check=0, post-check=0, max-age=0\r\n"
                         "Pragma: no-cache\r\n"
                         "Expires: -1\r\n\r\n";

    if (!visiong::http::send_all(client_socket, header.c_str(), header.length())) {
        close_client_socket();
        mark_client_done();
        return;
    }

    uint64_t last_sent_frame_id = 0;

    while (m_impl->is_running) {
        std::shared_ptr<const std::vector<unsigned char>> frame_to_send;
        uint64_t current_frame_id;

        {
            std::unique_lock<std::mutex> lock(m_impl->frame_mutex);
            // Wait for a newer frame id than the one already sent to this client. / 等待比当前客户端已发送帧号更新的 frame id。
            m_impl->frame_cv.wait(lock, [&] { return !m_impl->is_running || (m_impl->frame_id > last_sent_frame_id); });

            if (!m_impl->is_running) break;

            frame_to_send = m_impl->latest_jpeg_frame;
            current_frame_id = m_impl->frame_id;
        }
        
        last_sent_frame_id = current_frame_id;
        if (!frame_to_send || frame_to_send->empty()) continue;

        std::string frame_header = "\r\n--frame\r\n"
                                   "Content-Type: image/jpeg\r\n"
                                   "Content-Length: " + std::to_string(frame_to_send->size()) + "\r\n\r\n";
        
        if (!visiong::http::send_all(client_socket, frame_header.c_str(), frame_header.length())) {
            break; 
        }

        if (!visiong::http::send_all(client_socket, frame_to_send->data(), frame_to_send->size())) {
            break;
        }
    }

    close_client_socket();
    mark_client_done();
    VISIONG_LOG_INFO("DisplayHTTP", "Client disconnected.");
}

void DisplayHTTP::set_fps(int fps) {
    if (m_impl->flv) { m_impl->flv->set_fps(fps); return; }
    fps = visiong::venc::clamp_non_negative_fps(fps);
    m_impl->max_fps.store(fps, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(m_impl->display_mutex);
    m_impl->has_sent = false;
    m_impl->last_send = std::chrono::steady_clock::now();
}

int DisplayHTTP::get_fps() const {
    if (m_impl->flv) return m_impl->flv->get_fps();
    return m_impl->max_fps.load(std::memory_order_relaxed);
}

void DisplayHTTP::set_quality(int quality) {
    if (m_impl->flv) { m_impl->flv->set_quality(quality); return; }
    quality = visiong::venc::clamp_quality(quality);
    m_impl->quality.store(quality, std::memory_order_relaxed);
}

int DisplayHTTP::get_quality() const {
    if (m_impl->flv) return m_impl->flv->get_quality();
    return m_impl->quality.load(std::memory_order_relaxed);
}
