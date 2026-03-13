// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_MODULES_DISPLAYHTTPFLV_H
#define VISIONG_MODULES_DISPLAYHTTPFLV_H

#include <memory>
#include <string>

class ImageBuffer;

/// HTTP-FLV live streamer: serves H264/H265 video as FLV over HTTP. / HTTP-FLV 实时推流器：通过 HTTP 以 FLV 形式提供 H264/H265 视频。
/// Browser plays via mpegts.js / flv.js; VLC/ffplay also works.
/// Supports multiple concurrent viewers; low latency (~1 frame). / 支持多个并发观看者；低延迟（约 1 帧）。
class DisplayHTTPFLV {
  public:
    enum class Codec { H264 = 0, H265 = 1 };
    enum class RcMode { CBR = 0, VBR = 1 };

    DisplayHTTPFLV(int port = 8080, const std::string& path = "/live.flv", int quality = 75,
                   Codec codec = Codec::H264, int fps = 30, RcMode rc_mode = RcMode::CBR);
    ~DisplayHTTPFLV();

    DisplayHTTPFLV(const DisplayHTTPFLV&) = delete;
    DisplayHTTPFLV& operator=(const DisplayHTTPFLV&) = delete;

    void start();
    /// Start encoder/muxer only (no listen). Used when DisplayHTTP owns the server and feeds connections via / 仅启动编码器/复用器（不监听端口）。用于 DisplayHTTP 持有服务器并通过下述方式投喂连接时：
    /// add_stream_client.
    void start_encoder_only();
    void stop();
    bool is_running() const;

    /// Handle an already-accepted stream client (path already matched). Sends HTTP + FLV headers and adds to / 处理一个已经 accept 完成的流客户端（路径已匹配）。发送 HTTP + FLV 头并加入到
    /// client list.
    void add_stream_client(int client_fd);
    /// HTML page for browser playback (root path). Open http://ip:port/ to view stream in browser. / 供浏览器播放的 HTML 页面（根路径）。打开 http://ip:port/ 即可在浏览器中观看。
    std::string get_index_html() const;
    const std::string& get_path() const;

    void set_fps(int fps);
    int get_fps() const;
    void set_quality(int quality);
    int get_quality() const;
    void set_rc_mode(RcMode mode);
    RcMode get_rc_mode() const;

    /// Push one frame. Returns true on success. / 推送一帧；成功时返回 true。
    bool display(const ImageBuffer& img);

  private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

#endif  // VISIONG_MODULES_DISPLAYHTTPFLV_H

