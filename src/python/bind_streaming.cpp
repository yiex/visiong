// SPDX-License-Identifier: LGPL-3.0-or-later
#include "internal/bindings_common.h"

void bind_streaming(py::module_& m) {
    py::class_<DisplayHTTP>(m, "DisplayHTTP",
        "HTTP display: MJPEG (mode='jpg') or HTTP-FLV (mode='flv'). "
        "Open http://<device-ip>:port/ in browser to view; FLV uses built-in HTML player.")
        .def(py::init([](int port, int quality, const std::string& mode,
                         const std::string& flv_path, const std::string& flv_codec,
                         int flv_fps, const std::string& flv_rc_mode) {
            auto streamer = std::make_unique<DisplayHTTP>(port, quality, mode,
                flv_path, flv_codec, flv_fps, flv_rc_mode);
            streamer->start();
            return streamer;
        }), "port"_a = 8080, "quality"_a = 75, "mode"_a = "jpg",
            "flv_path"_a = "/live.flv", "flv_codec"_a = "h264", "flv_fps"_a = 30, "flv_rc_mode"_a = "cbr",
             "Initializes and starts the HTTP server.\n\n"
             "Args:\n"
             "    port (int): TCP port for HTTP (default 8080).\n"
             "    quality (int): JPEG/encoding quality 1-100 (default 75).\n"
             "    mode (str): 'jpg' = MJPEG stream with a local JPEG lock (BGR > RGB > YUV > other, size only grows and smaller frames are black-padded); 'flv' = HTTP-FLV (open http://ip:port/ in browser to view).\n"
             "    flv_path (str): URL path for FLV stream when mode='flv' (default '/live.flv').\n"
             "    flv_codec (str): 'h264' or 'h265' when mode='flv'.\n"
             "    flv_fps (int): Max FPS when mode='flv' (default 30).\n"
             "    flv_rc_mode (str): 'cbr' or 'vbr' when mode='flv'.")
        
        .def("stop", &DisplayHTTP::stop,
             py::call_guard<py::gil_scoped_release>(),
             "Stops the HTTP server and disconnects all clients. This is automatically called when the object is garbage collected.")
        .def("display", &DisplayHTTP::display, "img"_a,
             py::call_guard<py::gil_scoped_release>(),
             "Displays an ImageBuffer. In mode='jpg', it encodes to MJPEG. In mode='flv', it updates the latest frame cache; a background encoder pushes HTTP-FLV at the configured flv_fps.")
        .def("is_running", &DisplayHTTP::is_running,
             "Returns True if the server is currently running.")
        .def("set_fps", &DisplayHTTP::set_fps, "fps"_a,
             "Sets max FPS. 0 disables limiting.")
        .def("get_fps", &DisplayHTTP::get_fps,
             "Returns current max FPS.")
        .def("set_quality", &DisplayHTTP::set_quality, "quality"_a,
             "Sets JPEG quality (1-100).")
        .def("get_quality", &DisplayHTTP::get_quality,
             "Returns current JPEG quality (1-100).");

    py::class_<DisplayRTSP>(m, "DisplayRTSP",
        "An RTSP streamer. VisionG handles capture-frame resize, encoder setup, and stream packetization internally.")
        .def(py::init([](int port,
                         const std::string& path,
                         int quality,
                         const std::string& codec,
                         int fps,
                         int logs,
                         const std::string& rc_mode,
                         int width,
                         int height,
                         int mpp_channel) {
            auto streamer = std::make_unique<DisplayRTSP>(port, path, quality,
                                                          parse_rtsp_codec(codec), fps, logs,
                                                          parse_rtsp_rc_mode(rc_mode),
                                                          width, height, mpp_channel);
            streamer->start();
            return streamer;
        }), "port"_a = 554, "path"_a = "/live/0", "quality"_a = 75, "codec"_a = "h264", "fps"_a = 30, "logs"_a = 0, "rc_mode"_a = "cbr",
            "width"_a = 0, "height"_a = 0, "mpp_channel"_a = -1,
        "Initializes and immediately starts the RTSP streamer.\n\n"
        "Args:\n"
        "    port (int): The RTSP port (default 554).\n"
        "    path (str): The RTSP path (default /live/0).\n"
        "    quality (int): Unified quality 1-100 (mapped to bitrate by resolution).\n"
        "    codec (str): 'h264' or 'h265'.\n"
        "    fps (int): Max frames per second (default 30).\n"
        "    logs (int): 1 to enable logs, 0 to suppress.\n"
        "    rc_mode (str): 'cbr' or 'vbr' (default 'cbr').\n"
        "    width (int): Optional encoded output width. 0 keeps input width.\n"
        "    height (int): Optional encoded output height. 0 keeps input height.\n"
        "    mpp_channel (int): Advanced hardware encoder channel override. -1 auto-allocates.")
        .def("stop", &DisplayRTSP::stop,
             py::call_guard<py::gil_scoped_release>(),
             "Stops the RTSP server. This is automatically called when the object is garbage collected.")
        .def("set_fps", &DisplayRTSP::set_fps, "fps"_a,
             "Sets max frames per second. 0 disables limiting.")
        .def("get_fps", &DisplayRTSP::get_fps,
             "Returns current max frames per second.")
        .def("set_quality", &DisplayRTSP::set_quality, "quality"_a,
             "Sets encoding quality (1-100). Takes effect on the next display() call.")
        .def("get_quality", &DisplayRTSP::get_quality,
             "Returns current encoding quality (1-100).")
        .def("set_rc_mode", [](DisplayRTSP& self, const std::string& mode) {
            self.set_rc_mode(parse_rtsp_rc_mode(mode));
        }, "rc_mode"_a,
             "Sets rate control mode: 'cbr' or 'vbr'. Takes effect on next display() call.")
        .def("get_rc_mode", [](const DisplayRTSP& self) -> std::string {
            return self.get_rc_mode() == DisplayRTSP::RcMode::VBR ? "vbr" : "cbr";
        }, "Returns current rate control mode as string ('cbr' or 'vbr').")
        .def("set_logs", &DisplayRTSP::set_logs, "logs"_a,
             "Sets logs: 1 enable, 0 suppress.")
        .def("get_logs", &DisplayRTSP::get_logs,
             "Returns 1 if logs are enabled, otherwise 0.")
        .def("set_output_size", &DisplayRTSP::set_output_size, "width"_a, "height"_a,
             "Sets encoded output size. Use 0, 0 to keep the input frame size.")
        .def("get_output_width", &DisplayRTSP::get_output_width,
             "Returns encoded output width, or 0 if input width is used.")
        .def("get_output_height", &DisplayRTSP::get_output_height,
             "Returns encoded output height, or 0 if input height is used.")
        .def("get_mpp_channel", &DisplayRTSP::get_mpp_channel,
             "Advanced: returns the hardware encoder channel used by this stream.")
        .def("display", &DisplayRTSP::display, "img"_a,
             py::call_guard<py::gil_scoped_release>(),
             "Pushes the frame to all connected RTSP clients. VisionG handles resizing and encoding internally.")
        .def("is_running", &DisplayRTSP::is_running,
             "Returns True if the server is currently running.");

    py::class_<MppRecorder>(m, "MppRecorder", "Advanced hardware recorder (Annex-B raw stream or MP4 mux).")
        .def(py::init([](const std::string& filepath,
                         const std::string& codec,
                         const std::string& container,
                         int quality,
                         const std::string& rc_mode,
                         int fps,
                         bool mp4_faststart,
                         int mpp_channel,
                         const std::string& timestamp_mode) {
            return std::make_unique<MppRecorder>(filepath, parse_mpp_codec(codec),
                                                  parse_mpp_container(container), quality,
                                                  rc_mode, fps, mp4_faststart, mpp_channel,
                                                  timestamp_mode);
        }), "filepath"_a, "codec"_a = "h264", "container"_a = "mp4",
            "quality"_a = 75, "rc_mode"_a = "cbr", "fps"_a = 30, "mp4_faststart"_a = true,
            "mpp_channel"_a = -1, "timestamp_mode"_a = "fixed",
            "Creates a hardware recorder.\n\n"
            "Args:\n"
            "    filepath (str): Output path.\n"
            "    codec (str): 'h264'|'h265'.\n"
            "    container (str): 'mp4'|'annexb'.\n"
            "    quality (int): 1-100.\n"
            "    rc_mode (str): 'cbr'|'vbr'.\n"
            "    fps (int): target fps (1-120).\n"
            "    mp4_faststart (bool): write moov before mdat (faster start).\n"
            "    mpp_channel (int): Advanced hardware encoder channel override. -1 auto-allocates.\n"
            "    timestamp_mode (str): 'fixed' uses fps-derived CFR timestamps; 'wallclock' uses actual write() time for VFR MP4.")
        .def("write", &MppRecorder::write, "img"_a,
             py::call_guard<py::gil_scoped_release>(),
             "Encodes and writes one frame.")
        .def("close", &MppRecorder::close,
             py::call_guard<py::gil_scoped_release>(),
             "Closes and finalizes the output file (required for MP4).")
        .def("is_open", &MppRecorder::is_open,
             "Returns True after the output has been opened by the first write().")
        .def("path", &MppRecorder::path,
             "Returns output filepath.")
        .def("get_mpp_channel", &MppRecorder::get_mpp_channel,
             "Advanced: returns the hardware encoder channel used by this recorder.");

    // Cached MP4 writers used by ImageBuffer.save_video() and advanced save_mpp_* calls.
    m.def("close_mpp_recorder", &close_mpp_recorder, "filepath"_a,
          py::call_guard<py::gil_scoped_release>(),
          "Closes a cached MP4 writer for filepath (finalize MP4).");
    m.def("close_all_mpp_recorders", &close_all_mpp_recorders,
          py::call_guard<py::gil_scoped_release>(),
          "Closes all cached MP4 writers (finalize MP4).");
    m.def("close_video", &close_video, "filepath"_a,
          py::call_guard<py::gil_scoped_release>(),
          "Finalizes a video file opened implicitly by ImageBuffer.save_video().");
    m.def("close_all_videos", &close_all_videos,
          py::call_guard<py::gil_scoped_release>(),
          "Finalizes all video files opened implicitly by ImageBuffer.save_video().");

    py::class_<DisplayHTTPFLV>(m, "DisplayHTTPFLV",
        "HTTP-FLV live streamer: H264/H265 video over HTTP.\n"
        "Play in browser with mpegts.js/flv.js, or with VLC/ffplay.\n"
        "Supports multiple concurrent viewers, low latency (~1 frame).")
        .def(py::init([](int port, const std::string& path, int quality,
                         const std::string& codec, int fps, const std::string& rc_mode) {
            auto s = std::make_unique<DisplayHTTPFLV>(port, path, quality,
                                                      parse_httpflv_codec(codec), fps,
                                                      parse_httpflv_rc_mode(rc_mode));
            s->start();
            return s;
        }), "port"_a = 8080, "path"_a = "/live.flv", "quality"_a = 75,
            "codec"_a = "h264", "fps"_a = 30, "rc_mode"_a = "cbr",
        "Creates and starts an HTTP-FLV streamer.\n\n"
        "Args:\n"
        "    port (int): HTTP port (default 8080).\n"
        "    path (str): URL path (default /live.flv).\n"
        "    quality (int): Encoding quality 1-100.\n"
        "    codec (str): 'h264' or 'h265'.\n"
        "    fps (int): Max frames per second.\n"
        "    rc_mode (str): 'cbr' or 'vbr'.")
        .def("stop", &DisplayHTTPFLV::stop,
             py::call_guard<py::gil_scoped_release>(),
             "Stops the server.")
        .def("display", &DisplayHTTPFLV::display, "img"_a,
             py::call_guard<py::gil_scoped_release>(),
             "Updates the latest frame cache. A background encoder pushes HTTP-FLV at the configured FPS.")
        .def("is_running", &DisplayHTTPFLV::is_running,
             "Returns True if server is running.")
        .def("set_fps", &DisplayHTTPFLV::set_fps, "fps"_a,
             "Sets max FPS. 0 disables limiting.")
        .def("get_fps", &DisplayHTTPFLV::get_fps,
             "Returns current max FPS.")
        .def("set_quality", &DisplayHTTPFLV::set_quality, "quality"_a,
             "Sets encoding quality 1-100.")
        .def("get_quality", &DisplayHTTPFLV::get_quality,
             "Returns current encoding quality.")
        .def("set_rc_mode", [](DisplayHTTPFLV& self, const std::string& mode) {
            self.set_rc_mode(parse_httpflv_rc_mode(mode));
        }, "rc_mode"_a,
             "Sets rate control mode: 'cbr' or 'vbr'. Takes effect on next display() call.")
        .def("get_rc_mode", [](const DisplayHTTPFLV& self) -> std::string {
            return self.get_rc_mode() == DisplayHTTPFLV::RcMode::VBR ? "vbr" : "cbr";
        }, "Returns current rate control mode as string ('cbr' or 'vbr').");
}
