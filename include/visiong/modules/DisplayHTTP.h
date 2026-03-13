// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_MODULES_DISPLAYHTTP_H
#define VISIONG_MODULES_DISPLAYHTTP_H

#include <memory>
#include <string>

class ImageBuffer;

class DisplayHTTP {
public:
    DisplayHTTP(int port = 8080,
                int quality = 75,
                const std::string& mode = "jpg",
                const std::string& flv_path = "/live.flv",
                const std::string& flv_codec = "h264",
                int flv_fps = 30,
                const std::string& flv_rc_mode = "cbr");
    ~DisplayHTTP();

    DisplayHTTP(const DisplayHTTP&) = delete;
    DisplayHTTP& operator=(const DisplayHTTP&) = delete;

    void start();
    void stop();
    bool is_running() const;

    bool display(const ImageBuffer& img);
    void set_fps(int fps);
    int get_fps() const;
    void set_quality(int quality);
    int get_quality() const;

private:
    struct ClientThread;
    struct Impl;

    void server_loop();
    void client_handler(int client_socket, ClientThread* client_state);
    void serve_flv_index(int client_socket);

    std::unique_ptr<Impl> m_impl;
};

#endif  // VISIONG_MODULES_DISPLAYHTTP_H

