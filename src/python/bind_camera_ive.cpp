// SPDX-License-Identifier: LGPL-3.0-or-later
#include "internal/bindings_common.h"

void bind_camera(py::module_& m) {
    py::class_<Camera>(m, "Camera")
        .def(py::init<int, int, const std::string&, bool, const std::string&>(),
             "target_width"_a,
             "target_height"_a,
             "format"_a = "yuv",
             "hdr"_a = false,
             "crop_mode"_a = "auto",
             "Constructor. format defaults to 'yuv'. Supported values: 'bgr', 'rgb', 'yuv'/'yuv420', or 'gray'. crop_mode: 'auto' (default, follows the camera max-resolution aspect ratio), 'off', or any ratio such as '16:9', '4:3', '1:1', or '3:2'.\n构造函数。format 默认值为 'yuv'。可选值包括 'bgr'、'rgb'、'yuv'/'yuv420' 和 'gray'。crop_mode 可为默认的 'auto'（跟随摄像头最大分辨率比例）、'off'，或任意比例字符串，例如 '16:9'、'4:3'、'1:1'、'3:2'。")
        .def(py::init<>())
        .def("init",
             &Camera::init,
             "target_width"_a,
             "target_height"_a,
             "format"_a = "yuv",
             "hdr"_a = false,
             "crop_mode"_a = "auto",
             py::call_guard<py::gil_scoped_release>(),
             "Initializes the camera. format defaults to 'yuv'. Supported values: 'bgr', 'rgb', 'yuv'/'yuv420', or 'gray'. crop_mode: 'auto' (default, follows the camera max-resolution aspect ratio), 'off', or any ratio such as '16:9', '4:3', '1:1', or '3:2'.\n初始化摄像头。format 默认值为 'yuv'。可选值包括 'bgr'、'rgb'、'yuv'/'yuv420' 和 'gray'。crop_mode 可为默认的 'auto'（跟随摄像头最大分辨率比例）、'off'，或任意比例字符串，例如 '16:9'、'4:3'、'1:1'、'3:2'。")
                .def("skip", &Camera::skip, "num_frames"_a = 10,
             py::call_guard<py::gil_scoped_release>(),
             R"(Reads and discards a specified number of frames from the camera.

             This is highly recommended after initializing the camera to allow the ISP's
             auto exposure, auto white balance, and other algorithms to stabilize
             before you start processing actual frames.

             Args:
                 num_frames (int): The number of frames to skip. Defaults to 10,
                                   which is a reasonable value for most sensors.
             )")
        .def("snapshot", &Camera::snapshot, py::return_value_policy::move,
             py::call_guard<py::gil_scoped_release>(),
             "Captures a single frame from the camera and returns an ImageBuffer.")
        .def("release", &Camera::release,
             py::call_guard<py::gil_scoped_release>(),
             "Releases the camera and frees resources. Safe to call even if not initialized.")
        .def("is_initialized", &Camera::is_initialized,
             "Returns True if the camera has been successfully initialized.")
        .def_property_readonly("target_width", &Camera::target_width, "Get the user-defined target width.")
        .def_property_readonly("target_height", &Camera::target_height, "Get the user-defined target height.")
        .def_property_readonly("actual_width", &Camera::actual_width, "Get the actual hardware capture width after alignment.")
        .def_property_readonly("actual_height", &Camera::actual_height, "Get the actual hardware capture height after alignment.")
        .def_property_readonly("crop_mode", &Camera::get_crop_mode,
                               "Get the active crop mode. Returns 'off', a requested ratio like '1:1' or '3:2', or an auto-derived label such as 'auto(16:9)'.\n获取当前生效的裁切模式，可能是 'off'、用户指定比例（如 '1:1'、'3:2'）或自动推导标签（如 'auto(16:9)'）。")
        .def("get_capture_width", &Camera::get_capture_width,
             "Returns the actual capture width (alias for actual_width).")
        .def("get_capture_height", &Camera::get_capture_height,
             "Returns the actual capture height (alias for actual_height).")
    	.def("set_saturation", &Camera::set_saturation, "value"_a,
             "Sets the image saturation. Range: [0, 255]. Raises ValueError on invalid input.")
    	.def("set_contrast", &Camera::set_contrast, "value"_a,
             "Sets the image contrast. Range: [0, 255]. Raises ValueError on invalid input.")
    	.def("set_brightness", &Camera::set_brightness, "value"_a,
             "Sets the image brightness. Range: [0, 255]. Raises ValueError on invalid input.")
        .def("set_sharpness", &Camera::set_sharpness, "value"_a,
             "Sets the image sharpness. Range: [0, 100]. Raises ValueError on invalid input.")
    	.def("set_hue", &Camera::set_hue, "value"_a,
             "Sets the image hue. Range: [0, 255]. Raises ValueError on invalid input.")

    	.def("set_white_balance_mode", &Camera::set_white_balance_mode, "mode"_a,
             "Sets white balance mode ('auto' or 'manual'). Raises ValueError on invalid mode.")
    	.def("set_white_balance_temperature", &Camera::set_white_balance_temperature, "temp"_a,
             "Sets white balance color temperature in manual mode. Raises ValueError on invalid input or RuntimeError if not in manual mode.")

    	.def("set_exposure_mode", &Camera::set_exposure_mode, "mode"_a,
             "Sets exposure mode ('auto' or 'manual'). Raises ValueError on invalid mode.")
    	.def("set_exposure_time", &Camera::set_exposure_time, "time_in_seconds"_a,
             "Sets manual exposure time in seconds. Must be positive. Raises RuntimeError if not in manual mode.")
    	.def("set_exposure_gain", &Camera::set_exposure_gain, "gain"_a,
             "Sets manual exposure gain. Typical range: [0, 127]. Raises ValueError on invalid input or RuntimeError if not in manual mode.")
        
    	.def("set_spatial_denoise_level", &Camera::set_spatial_denoise_level, "level"_a,
         "Sets the spatial (2D) denoise level. Range: [0, 100]. Raises ValueError on invalid input.")
    	.def("set_temporal_denoise_level", &Camera::set_temporal_denoise_level, "level"_a,
         "Sets the temporal (3D) denoise level. Range: [0, 100]. Raises ValueError on invalid input.")

    	.def("set_frame_rate", &Camera::set_frame_rate, "fps"_a,
         "Sets the camera frame rate. Range: [10, 60] (or 0 for auto). Raises ValueError on invalid input.")
    	.def("set_power_line_frequency", &Camera::set_power_line_frequency, "mode"_a,
         "Sets anti-flicker mode ('50hz', '60hz', or 'off'). Raises ValueError on invalid mode.")
    	.def("set_flip", &Camera::set_flip, "flip"_a, "mirror"_a,
         "Sets image flip (vertical) and mirror (horizontal).")
        .def("get_saturation", &Camera::get_saturation, "Gets the current image saturation (0-255).")
        .def("get_contrast", &Camera::get_contrast, "Gets the current image contrast (0-255).")
        .def("get_brightness", &Camera::get_brightness, "Gets the current image brightness (0-255).")
        .def("get_sharpness", &Camera::get_sharpness, "Gets the current image sharpness (0-100).")
        .def("get_hue", &Camera::get_hue, "Gets the current image hue (0-255).")
        .def("get_white_balance_mode", &Camera::get_white_balance_mode, "Gets the current white balance mode ('auto' or 'manual').")
        .def("get_white_balance_temperature", &Camera::get_white_balance_temperature, "Gets the current white balance color temperature.")
        .def("get_exposure_mode", &Camera::get_exposure_mode, "Gets the current exposure mode ('auto' or 'manual').")
        .def("get_exposure_time", &Camera::get_exposure_time, "Gets the current exposure time in seconds.")
        .def("get_exposure_gain", &Camera::get_exposure_gain, "Gets the current exposure gain.")
        .def("lock_focus", &Camera::lock_focus,
             "Locks the autofocus at its current position. Prevents further automatic adjustments.")
        .def("unlock_focus", &Camera::unlock_focus,
             "Unlocks the autofocus, allowing it to resume its configured mode (e.g., continuous focus).")
        .def("trigger_focus", &Camera::trigger_focus,
             "Performs a single, one-shot autofocus search. This is an action, not a mode.")
        .def("set_focus_mode", &Camera::set_focus_mode, "mode"_a,
             "Sets the autofocus mode. Supported modes:\n"
             " - 'continuous': The camera will continuously try to keep the scene in focus.\n"
             " - 'manual': Disables automatic focus. Use `set_manual_focus()` to set the position.")
        .def("set_manual_focus", &Camera::set_manual_focus, "position"_a,
             "Moves the lens to a specific motor code position. This implicitly sets the mode to 'manual'.\n"
             "The valid range for 'position' is hardware-dependent.")
        .def("get_focus_position", &Camera::get_focus_position,
             "Returns the current motor code position of the lens. Returns -1 on failure.");
}

#if VISIONG_WITH_IVE
void bind_ive(py::module_& m) {
    py::enum_<IVE_IMAGE_TYPE_E>(m, "ImageTypeIVE")
        .value("U8C1", IVE_IMAGE_TYPE_U8C1, "Grayscale, unsigned 8-bit")
        .value("S16C1", IVE_IMAGE_TYPE_S16C1, "Grayscale, signed 16-bit")
        .value("U16C1", IVE_IMAGE_TYPE_U16C1, "Grayscale, unsigned 16-bit")
        .value("U64C1", IVE_IMAGE_TYPE_U64C1, "Grayscale, unsigned 64-bit")
        .value("YUV420SP", IVE_IMAGE_TYPE_YUV420SP)
        .value("YUV422SP", IVE_IMAGE_TYPE_YUV422SP)
        .value("U8C3_PACKAGE", IVE_IMAGE_TYPE_U8C3_PACKAGE)
        .export_values();

    py::enum_<IVE_SOBEL_OUT_CTRL_E>(m, "SobelOutCtrl")
        .value("HOR", IVE_SOBEL_OUT_CTRL_HOR)
        .value("VER", IVE_SOBEL_OUT_CTRL_VER)
        .value("BOTH", IVE_SOBEL_OUT_CTRL_BOTH)
        .export_values();

    py::enum_<IVE_ORD_STAT_FILTER_MODE_E>(m, "OrdStatFilterMode")
        .value("MEDIAN", IVE_ORD_STAT_FILTER_MODE_MEDIAN)
        .value("MAX", IVE_ORD_STAT_FILTER_MODE_MAX)
        .value("MIN", IVE_ORD_STAT_FILTER_MODE_MIN)
        .export_values();

    py::enum_<IVE_SUB_MODE_E>(m, "SubMode")
        .value("ABS", IVE_SUB_MODE_ABS)
        .value("SHIFT", IVE_SUB_MODE_SHIFT)
        .export_values();

    py::enum_<IVE_LOGICOP_MODE_E>(m, "LogicOp")
        .value("AND", IVE_LOGICOP_MODE_AND)
        .value("OR", IVE_LOGICOP_MODE_OR)
        .value("XOR", IVE_LOGICOP_MODE_XOR)
        .export_values();

    py::enum_<IVE_THRESH_MODE_E>(m, "ThreshMode")
        .value("BINARY", IVE_THRESH_MODE_BINARY)
        .value("TRUNC", IVE_THRESH_MODE_TRUNC)
        .value("TO_MINVAL", IVE_THRESH_MODE_TO_MINVAL)
        .export_values();

    py::enum_<IVE_INTEG_OUT_CTRL_E>(m, "IntegOutCtrl")
        .value("COMBINE", IVE_INTEG_OUT_CTRL_COMBINE)
        .value("SUM", IVE_INTEG_OUT_CTRL_SUM)
        .value("SQSUM", IVE_INTEG_OUT_CTRL_SQSUM)
        .export_values();

    py::enum_<IVE_CSC_MODE_E>(m, "CscMode")
        // YUV to RGB / YUV 以 RGB
        .value("YUV2RGB_BT601_LIMITED", IVE_CSC_MODE_LIMIT_BT601_YUV2RGB)
        .value("YUV2RGB_BT709_LIMITED", IVE_CSC_MODE_LIMIT_BT709_YUV2RGB)
        .value("YUV2RGB_BT601_FULL", IVE_CSC_MODE_FULL_BT601_YUV2RGB)
        .value("YUV2RGB_BT709_FULL", IVE_CSC_MODE_FULL_BT709_YUV2RGB)
        // YUV to HSV / YUV 以 HSV
        .value("YUV2HSV_BT601_LIMITED", IVE_CSC_MODE_LIMIT_BT601_YUV2HSV)
        .value("YUV2HSV_BT709_LIMITED", IVE_CSC_MODE_LIMIT_BT709_YUV2HSV)
        .value("YUV2HSV_BT601_FULL", IVE_CSC_MODE_FULL_BT601_YUV2HSV)
        .value("YUV2HSV_BT709_FULL", IVE_CSC_MODE_FULL_BT709_YUV2HSV)
        // RGB to YUV / RGB 以 YUV
        .value("RGB2YUV_BT601_LIMITED", IVE_CSC_MODE_LIMIT_BT601_RGB2YUV)
        .value("RGB2YUV_BT709_LIMITED", IVE_CSC_MODE_LIMIT_BT709_RGB2YUV)
        .value("RGB2YUV_BT601_FULL", IVE_CSC_MODE_FULL_BT601_RGB2YUV)
        .value("RGB2YUV_BT709_FULL", IVE_CSC_MODE_FULL_BT709_RGB2YUV)
        // RGB to HSV / RGB 以 HSV
        .value("RGB2HSV_BT601_LIMITED", IVE_CSC_MODE_LIMIT_BT601_RGB2HSV)
        .value("RGB2HSV_BT709_LIMITED", IVE_CSC_MODE_LIMIT_BT709_RGB2HSV)
        .value("RGB2HSV_BT601_FULL", IVE_CSC_MODE_FULL_BT601_RGB2HSV)
        .value("RGB2HSV_BT709_FULL", IVE_CSC_MODE_FULL_BT709_RGB2HSV)
        .export_values();

    py::enum_<IVE_16BIT_TO_8BIT_MODE_E>(m, "Cast16to8Mode")
        .value("S16_TO_S8", IVE_16BIT_TO_8BIT_MODE_S16_TO_S8)
        .value("S16_TO_U8_ABS", IVE_16BIT_TO_8BIT_MODE_S16_TO_U8_ABS)
        .value("S16_TO_U8_BIAS", IVE_16BIT_TO_8BIT_MODE_S16_TO_U8_BIAS)
        .value("U16_TO_U8", IVE_16BIT_TO_8BIT_MODE_U16_TO_U8)
        .export_values();
        
    py::enum_<IVE_DMA_MODE_E>(m, "DmaMode")
        .value("DIRECT_COPY", IVE_DMA_MODE_DIRECT_COPY)
        .value("INTERVAL_COPY", IVE_DMA_MODE_INTERVAL_COPY)
        .value("SET_3BYTE", IVE_DMA_MODE_SET_3BYTE)
        .value("SET_8BYTE", IVE_DMA_MODE_SET_8BYTE)
        .export_values();

    py::enum_<IVE_LBP_CMP_MODE_E>(m, "LbpCmpMode")
        .value("NORMAL", IVE_LBP_CMP_MODE_NORMAL)
        .value("ABS", IVE_LBP_CMP_MODE_ABS)
        .export_values();

    py::enum_<IVE_SAD_MODE_E>(m, "SadMode")
        .value("MB_4X4", IVE_SAD_MODE_MB_4X4)
        .value("MB_8X8", IVE_SAD_MODE_MB_8X8)
        .value("MB_16X16", IVE_SAD_MODE_MB_16X16)
        .export_values();
        
    py::class_<IVEModel>(m, "IVEModel", "Manages a persistent memory block for stateful IVE algorithms like GMM or background modeling.")
        .def(py::init<int, int, int>(), "width"_a, "height"_a, "model_size"_a = 0,
             "Creates a memory model for IVE algorithms. `model_size` can be specified for exact memory requirements.");

    py::class_<IVEMotionVector>(m, "MotionVector", "Holds the result of a Lucas-Kanade optical flow tracking point.")
        .def_readonly("status", &IVEMotionVector::status, "Tracking status: 0 for success, -1 for failure.")
        .def_readonly("mv_x", &IVEMotionVector::mv_x, "Motion X in S9.7 fixed-point; divide by 128 for pixel displacement.")
        .def_readonly("mv_y", &IVEMotionVector::mv_y, "Motion Y in S9.7 fixed-point; divide by 128 for pixel displacement.")
        .def("__repr__", [](const IVEMotionVector &mv) {
            return "<MotionVector status=" + std::to_string(mv.status) +
                   ", mv=(" + std::to_string(mv.mv_x) + "," + std::to_string(mv.mv_y) + ")>";
        });

    py::class_<IVE, std::unique_ptr<IVE, py::nodelete>>(m, "IVE", "Hardware-accelerated image processing using Rockchip IVE.")
        .def(py::init([]() {
            return std::unique_ptr<IVE, py::nodelete>(&IVE::get_instance());
        }), "Initializes and/or returns the global IVE processor instance.")

        .def_static("set_log_enabled", &IVE::set_log_enabled, "enabled"_a,
             "Enable or disable IVE internal logs (default off; env VISIONG_IVE_LOG=1 enables).")
        .def_static("is_log_enabled", &IVE::is_log_enabled,
             "Returns True if IVE internal logs are enabled.")

        .def("filter", &IVE::filter, "src"_a, "mask"_a,
             py::return_value_policy::move,
             py::call_guard<py::gil_scoped_release>(),
             "5x5 filter. src: GRAY8. mask: 25 ints. Returns GRAY8 - do not pass to cast_16bit_to_8bit. Use a 5x5 Gaussian mask for blur (no separate gaussian_filter).")
        .def("sobel", &IVE::sobel, "src"_a, "out_ctrl"_a = IVE_SOBEL_OUT_CTRL_BOTH, "out_format"_a = IVE_IMAGE_TYPE_S16C1,
             py::return_value_policy::move,
             py::call_guard<py::gil_scoped_release>(),
             "Sobel edge detection. src: GRAY8. Returns (horizontal_edges, vertical_edges). When out_format=S16C1 each result is S16C1 - unpack and pass to cast_16bit_to_8bit for display; when U8C1 each result is GRAY8.")
        .def("canny", &IVE::canny, "src"_a, "high_thresh"_a, "low_thresh"_a,
             py::return_value_policy::move,
             py::call_guard<py::gil_scoped_release>(),
             "Canny edge detection. src must be GRAY8. high_thresh and low_thresh typically in 0-255.")
        .def("mag_and_ang", &IVE::mag_and_ang, "src"_a, "threshold"_a = 0, "return_magnitude"_a = true,
             py::return_value_policy::move,
             py::call_guard<py::gil_scoped_release>(),
             "Gradient magnitude/angle. return_magnitude=True -> S16C1 (use cast_16bit_to_8bit for display); False -> U8C1.")
        .def("dilate", &IVE::dilate, "src"_a, "kernel_size"_a = 3,
             py::return_value_policy::move,
             py::call_guard<py::gil_scoped_release>(),
             "Performs morphological dilation. `src` must be GRAY8. `kernel_size` can be 3 or 5.")
        .def("erode", &IVE::erode, "src"_a, "kernel_size"_a = 3,
             py::return_value_policy::move,
             py::call_guard<py::gil_scoped_release>(),
             "Performs morphological erosion. `src` must be GRAY8. `kernel_size` can be 3 or 5.")
        .def("ordered_stat_filter", &IVE::ordered_stat_filter, "src"_a, "mode"_a,
             py::return_value_policy::move,
             py::call_guard<py::gil_scoped_release>(),
             "Performs an ordered statistic filter (Median, Max, Min). `src` must be GRAY8.")
        .def("add", &IVE::add, "src1"_a, "src2"_a,
             py::return_value_policy::move, py::call_guard<py::gil_scoped_release>(), "Pixel-wise addition. Returns GRAY8; do not pass to cast_16bit_to_8bit.")
        .def("sub", &IVE::sub, "src1"_a, "src2"_a, "mode"_a = IVE_SUB_MODE_ABS,
             py::return_value_policy::move, py::call_guard<py::gil_scoped_release>(), "Pixel-wise subtraction. Returns GRAY8; do not pass to cast_16bit_to_8bit. Mode: ABS or SHIFT.")
        .def("logic_op", &IVE::logic_op, "src1"_a, "src2"_a, "op"_a,
             py::return_value_policy::move, py::call_guard<py::gil_scoped_release>(), "Performs a pixel-wise logical operation (AND, OR, XOR) on two images.")
        .def("threshold",
             [](IVE& self, const ImageBuffer& src, py::args args, py::kwargs kwargs) {
                 int low_thresh = 0;
                 int high_thresh = 255;
                 IVE_THRESH_MODE_E mode = IVE_THRESH_MODE_BINARY;

                 if (args.size() > 0) low_thresh = args[0].cast<int>();
                 if (args.size() > 1) high_thresh = args[1].cast<int>();
                 if (args.size() > 2) mode = args[2].cast<IVE_THRESH_MODE_E>();
                 if (args.size() > 3) {
                     throw std::invalid_argument("threshold accepts at most 3 positional arguments: low, high, mode.");
                 }

                 if (kwargs.contains("low_thresh")) low_thresh = kwargs["low_thresh"].cast<int>();
                 if (kwargs.contains("low")) low_thresh = kwargs["low"].cast<int>();
                 if (kwargs.contains("high_thresh")) high_thresh = kwargs["high_thresh"].cast<int>();
                 if (kwargs.contains("high")) high_thresh = kwargs["high"].cast<int>();
                 if (kwargs.contains("mode")) mode = kwargs["mode"].cast<IVE_THRESH_MODE_E>();

                 auto clamp_u8 = [](int v) -> uint8_t {
                     if (v < 0) return 0;
                     if (v > 255) return 255;
                     return static_cast<uint8_t>(v);
                 };
                 return self.threshold(src, clamp_u8(low_thresh), clamp_u8(high_thresh), mode);
             },
             "src"_a,
             py::return_value_policy::move,
             "Thresholding on GRAY8. kwargs accept low/low_thresh, high/high_thresh, mode. Returns GRAY8; do not pass to cast_16bit_to_8bit.")
        .def("cast_16bit_to_8bit", &IVE::cast_16bit_to_8bit, "src"_a, "mode"_a,
             py::return_value_policy::move, py::call_guard<py::gil_scoped_release>(), "Casts 16-bit (S16C1/U16C1) to 8-bit. Input must be from sobel(S16C1), mag_and_ang(magnitude), norm_grad, or sad (first element); not from filter, add, sub, threshold, gmm.")
        .def("hist", &IVE::hist, "src"_a, py::call_guard<py::gil_scoped_release>(), "Calculates the histogram of a GRAY8 image. Returns a list of 256 integers.")
        .def("equalize_hist", &IVE::equalize_hist, "src"_a, py::return_value_policy::move, py::call_guard<py::gil_scoped_release>(), "Performs histogram equalization on a GRAY8 image.")
        .def("integral", &IVE::integral, "src"_a, "mode"_a = IVE_INTEG_OUT_CTRL_COMBINE,
             py::return_value_policy::move, py::call_guard<py::gil_scoped_release>(), "Integral image. mode: COMBINE, SUM, or SQSUM. Output U64C1.")
        .def("ccl", &IVE::ccl, "src"_a, "min_area"_a = 100,
             py::call_guard<py::gil_scoped_release>(),
             "Connected Components Labeling on a binarized GRAY8 image. Blobs with area < min_area are filtered. Returns list of Blob.")
        .def("ncc",
             [](IVE& self, const ImageBuffer& src1, const ImageBuffer& src2, py::kwargs kwargs) {
                 bool auto_resize = true;
                 if (kwargs.contains("auto_resize")) auto_resize = kwargs["auto_resize"].cast<bool>();
                 if (src1.width == src2.width && src1.height == src2.height) {
                     return self.ncc(src1, src2);
                 }
                 if (!auto_resize) {
                     throw std::invalid_argument("ncc requires images of the same size (set auto_resize=True to resize src2).");
                 }
                 ImageBuffer resized = src2.resize(src1.width, src1.height);
                 return self.ncc(src1, resized);
             },
             "src1"_a, "src2"_a, py::return_value_policy::move,
             "Normalized cross-correlation. If sizes differ, set auto_resize=True (default) to resize src2 to src1 size.")
        .def("csc", &IVE::csc, "src"_a, "mode"_a, py::return_value_policy::move, py::call_guard<py::gil_scoped_release>(), "Performs Color Space Conversion (e.g., YUV to RGB).")
        
        .def("yuv_to_rgb",
             [](IVE& self, const ImageBuffer& src, py::kwargs kwargs) {
                 if (kwargs.contains("mode")) {
                     IVE_CSC_MODE_E mode = kwargs["mode"].cast<IVE_CSC_MODE_E>();
                     if (src.format == RK_FMT_YUV420SP_VU) {
                         ImageBuffer converted = src.to_format(RK_FMT_YUV420SP);
                         return self.csc(converted, mode);
                     }
                     return self.csc(src, mode);
                 }
                 bool full_range = true;
                 if (kwargs.contains("full_range")) full_range = kwargs["full_range"].cast<bool>();
                 return self.yuv_to_rgb(src, full_range);
             },
             "src"_a, py::return_value_policy::move,
             "Converts YUV420SP (or YUV420SP_VU) to RGB. Use full_range=True/False, or pass mode=visiong.CscMode.YUV2RGB_*.")
        .def("yuv_to_hsv",
             [](IVE& self, const ImageBuffer& src, py::kwargs kwargs) {
                 if (kwargs.contains("mode")) {
                     IVE_CSC_MODE_E mode = kwargs["mode"].cast<IVE_CSC_MODE_E>();
                     if (src.format == RK_FMT_YUV420SP_VU) {
                         ImageBuffer converted = src.to_format(RK_FMT_YUV420SP);
                         return self.csc(converted, mode);
                     }
                     return self.csc(src, mode);
                 }
                 bool full_range = true;
                 if (kwargs.contains("full_range")) full_range = kwargs["full_range"].cast<bool>();
                 return self.yuv_to_hsv(src, full_range);
             },
             "src"_a, py::return_value_policy::move,
             "Converts YUV420SP (or YUV420SP_VU) to HSV. Use full_range=True/False, or pass mode=visiong.CscMode.YUV2HSV_*.")
        .def("rgb_to_yuv", &IVE::rgb_to_yuv, "src"_a, "full_range"_a = true, py::return_value_policy::move,
             py::call_guard<py::gil_scoped_release>(),
             "Converts RGB/BGR image to YUV420SP.")
        .def("rgb_to_hsv", &IVE::rgb_to_hsv, "src"_a, "full_range"_a = true, py::return_value_policy::move,
             py::call_guard<py::gil_scoped_release>(),
             "Converts RGB/BGR image to HSV. H:[0,180], S:[0,255], V:[0,255].")

        .def("dma", &IVE::dma, "src"_a, "mode"_a = IVE_DMA_MODE_DIRECT_COPY, py::return_value_policy::move,
             py::call_guard<py::gil_scoped_release>(),
             "Performs a direct memory access operation (e.g., copy).")
        .def("cast_8bit_to_8bit", &IVE::cast_8bit_to_8bit, "src"_a, "bias"_a, "numerator"_a, "denominator"_a, py::return_value_policy::move,
             py::call_guard<py::gil_scoped_release>(),
             "Scales an 8-bit image using the formula: dst = (src * numerator / denominator) + bias.")
        .def("map", &IVE::map, "src"_a, "lut"_a, py::return_value_policy::move,
             py::call_guard<py::gil_scoped_release>(),
             "Pixel LUT mapping. lut: list of 256 integers in [0,255]; output[x] = lut[src[x]].")
        .def("gmm", &IVE::gmm, "src"_a, "model"_a, "first_frame"_a = false,
             "GMM background subtraction. Returns (foreground, background); both GRAY8 - do not pass to cast_16bit_to_8bit.")
        .def("gmm2", &IVE::gmm2, "src"_a, "factor"_a, "model"_a, "first_frame"_a = false,
             "GMM2 with factor image. Returns (foreground, background); both GRAY8.")
        .def("lbp", &IVE::lbp, "src"_a, "abs_mode"_a = false, "threshold"_a = 0, py::return_value_policy::move,
             "Calculates the Local Binary Pattern of an image.")
        .def("norm_grad", &IVE::norm_grad, "src"_a,
             py::call_guard<py::gil_scoped_release>(),
             "Normalized gradient. Returns (horizontal_grad, vertical_grad); both S16C1 - use cast_16bit_to_8bit for display.")
        .def("lk_optical_flow", &IVE::lk_optical_flow, "prev_img"_a, "next_img"_a, "points"_a,
             py::call_guard<py::gil_scoped_release>(),
             "Performs Lucas-Kanade optical flow. points: list of (x,y). Returns list of MotionVector; mv_x/mv_y are S9.7 (divide by 128 for pixels). Up to ~500 points.")
        .def("st_corner", &IVE::st_corner, "src"_a, "max_corners"_a = 200, "min_dist"_a = 10, "quality_level"_a = 25,
             py::call_guard<py::gil_scoped_release>(),
             "Performs Shi-Tomasi corner detection. Returns a list of (x, y) corner tuples.")
        .def("match_bg_model", &IVE::match_bg_model, "current_img"_a, "bg_model"_a, "frame_num"_a, py::return_value_policy::move,
             "Matches the current image against a background model. Returns a foreground flag image.")
        .def("update_bg_model", &IVE::update_bg_model, "current_img"_a, "fg_flag"_a, "bg_model"_a, "frame_num"_a, py::return_value_policy::move,
             "Updates the background model. Returns the new background image.")
        .def("sad", &IVE::sad, "src1"_a, "src2"_a, "mode"_a, "threshold"_a, "min_val"_a = 0, "max_val"_a = 255,
             py::call_guard<py::gil_scoped_release>(),
             "Sum of Absolute Differences. Returns (sad_image U16C1, threshold_image U8C1); cast sad_image for 8-bit display.")
        .def("create_pyramid", &IVE::create_pyramid, "src"_a, "levels"_a,
             py::call_guard<py::gil_scoped_release>(),
             "Builds an image pyramid with levels levels. Returns list of ImageBuffer from full size down.");
}
#endif
