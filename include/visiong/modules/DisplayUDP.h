// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_MODULES_DISPLAYUDP_H
#define VISIONG_MODULES_DISPLAYUDP_H

#include <memory>
#include <string>

class ImageBuffer;

#define UDP_SEND_MAX_LEN (60 * 1024)

class DisplayUDP {
  public:
    DisplayUDP(const std::string& udp_ip = "172.32.0.100", int udp_port = 8000, int jpeg_quality = 75);
    ~DisplayUDP();

    DisplayUDP(const DisplayUDP&) = delete;
    DisplayUDP& operator=(const DisplayUDP&) = delete;

    bool init(const std::string& udp_ip, int udp_port, int jpeg_quality = 75);
    bool display(const ImageBuffer& img_buf);
    void release();
    bool is_initialized() const;
    static const char* PixelFormatToString(int format);

  private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

#endif  // VISIONG_MODULES_DISPLAYUDP_H

