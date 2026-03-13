// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_MODULES_DISPLAYRTSP_H
#define VISIONG_MODULES_DISPLAYRTSP_H

#include <memory>
#include <string>

class ImageBuffer;

class DisplayRTSP {
  public:
    enum class Codec { H264 = 0, H265 = 1 };

    enum class RcMode { CBR = 0, VBR = 1 };

    DisplayRTSP(int port = 554, const std::string& path = "/live/0", int quality = 75,
                Codec codec = Codec::H264, int fps = 30, int logs = 0, RcMode rc_mode = RcMode::CBR);
    ~DisplayRTSP();

    DisplayRTSP(const DisplayRTSP&) = delete;
    DisplayRTSP& operator=(const DisplayRTSP&) = delete;

    void start();
    void stop();
    bool is_running() const;
    void set_fps(int fps);
    int get_fps() const;
    void set_quality(int quality);
    int get_quality() const;
    void set_rc_mode(RcMode mode);
    RcMode get_rc_mode() const;
    void set_suppress_logs(bool enable);
    bool get_suppress_logs() const;
    void set_logs(int logs);
    int get_logs() const;

    bool display(const ImageBuffer& img);

  private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

#endif  // VISIONG_MODULES_DISPLAYRTSP_H

