// SPDX-License-Identifier: LGPL-3.0-or-later
#include "internal/bindings_common.h"

void bind_camera(py::module_& m) {
    py::class_<Camera>(m, "Camera")
        .def(py::init<int, int, const std::string&, bool, const std::string&, const std::string&>(),
             "target_width"_a,
             "target_height"_a,
             "format"_a = "yuv",
             "hdr"_a = false,
             "crop_mode"_a = "auto",
             "isp_path"_a = "auto",
             "Create a camera stream. format: yuv, rgb, bgr, or gray. isp_path: auto, main, self, bypass, or /dev/videoN.")
        .def(py::init<>())
        .def("init",
             &Camera::init,
             "target_width"_a,
             "target_height"_a,
             "format"_a = "yuv",
             "hdr"_a = false,
             "crop_mode"_a = "auto",
             "isp_path"_a = "auto",
             py::call_guard<py::gil_scoped_release>(),
             "Initialize the camera stream.")
        .def("skip", &Camera::skip, "num_frames"_a = 10,
             py::call_guard<py::gil_scoped_release>(),
             "Reads and discards warmup frames.")
        .def("snapshot", &Camera::snapshot, py::return_value_policy::move,
             py::call_guard<py::gil_scoped_release>(),
             "Captures a single frame from the camera and returns an ImageBuffer.")
        .def("sub", &Camera::sub,
             "target_width"_a = 640,
             "target_height"_a = 360,
             "format"_a = "auto",
             "crop_mode"_a = "auto",
             "isp_path"_a = "auto",
             py::return_value_policy::reference_internal,
             "Configures a secondary camera stream and returns self.")
        .def("snapshots",
             [](Camera& self) {
                 std::pair<ImageBuffer, ImageBuffer> frames;
                 {
                     py::gil_scoped_release release;
                     frames = self.snapshots();
                 }
                 py::tuple out(2);
                 out[0] = py::cast(std::move(frames.first), py::return_value_policy::move);
                 out[1] = py::cast(std::move(frames.second), py::return_value_policy::move);
                 return out;
             },
             "Captures the main stream and configured sub stream as (main, sub).")
        .def("open_stream", &Camera::open_stream,
             "target_width"_a = 640,
             "target_height"_a = 360,
             "format"_a = "auto",
             "crop_mode"_a = "auto",
             "isp_path"_a = "auto",
             py::return_value_policy::move,
             "Advanced: opens another ISP output as a regular Camera.")
        .def("release", &Camera::release,
             py::call_guard<py::gil_scoped_release>(),
             "Releases the camera and frees resources. Safe to call even if not initialized.")
        .def("close", &Camera::release,
             py::call_guard<py::gil_scoped_release>(),
             "Alias for release().")
        .def("has_sub", &Camera::has_sub,
             "Returns True if a secondary stream is configured.")
        .def("close_sub", &Camera::close_sub,
             py::call_guard<py::gil_scoped_release>(),
             "Closes the configured secondary stream.")
        .def("is_initialized", &Camera::is_initialized,
             "Returns True if the camera has been successfully initialized.")
        .def("__enter__", [](Camera& self) -> Camera& { return self; },
             py::return_value_policy::reference_internal)
        .def("__exit__", [](Camera& self, const py::object&, const py::object&, const py::object&) {
            self.release();
            return false;
        })
        .def_property_readonly("target_width", &Camera::target_width, "Get the user-defined target width.")
        .def_property_readonly("target_height", &Camera::target_height, "Get the user-defined target height.")
        .def_property_readonly("actual_width", &Camera::actual_width, "Get the actual hardware capture width after alignment.")
        .def_property_readonly("actual_height", &Camera::actual_height, "Get the actual hardware capture height after alignment.")
        .def_property_readonly("crop_mode", &Camera::get_crop_mode,
                               "Get the active crop mode.")
        .def("get_capture_width", &Camera::get_capture_width,
             "Returns the actual capture width (alias for actual_width).")
        .def("get_capture_height", &Camera::get_capture_height,
             "Returns the actual capture height (alias for actual_height).")
        .def_property_readonly("format", &Camera::get_format,
                               "Get the normalized output format token.")
        .def_property_readonly("hdr_enabled", &Camera::is_hdr_enabled,
                               "Returns True if HDR mode is enabled.")
        .def_property_readonly("isp_path", &Camera::get_isp_path,
                               "Get the selected ISP output path label.")
        .def_property_readonly("device_path", &Camera::get_device_path,
                               "Get the selected V4L2 device path.")
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
        .value("S8C1", IVE_IMAGE_TYPE_S8C1, "Grayscale, signed 8-bit")
        .value("S16C1", IVE_IMAGE_TYPE_S16C1, "Grayscale, signed 16-bit")
        .value("U16C1", IVE_IMAGE_TYPE_U16C1, "Grayscale, unsigned 16-bit")
        .value("U64C1", IVE_IMAGE_TYPE_U64C1, "Grayscale, unsigned 64-bit")
        .value("YUV420SP", IVE_IMAGE_TYPE_YUV420SP)
        .value("YUV422SP", IVE_IMAGE_TYPE_YUV422SP)
        .value("U8C3_PACKAGE", IVE_IMAGE_TYPE_U8C3_PACKAGE)
        .value("U8C3_PLANAR", IVE_IMAGE_TYPE_U8C3_PLANAR)
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
        .value("MIN_MID_MAX", IVE_THRESH_MODE_MIN_MID_MAX)
        .value("ORI_MID_MAX", IVE_THRESH_MODE_ORI_MID_MAX)
        .value("MIN_MID_ORI", IVE_THRESH_MODE_MIN_MID_ORI)
        .value("MIN_ORI_MAX", IVE_THRESH_MODE_MIN_ORI_MAX)
        .value("ORI_MID_ORI", IVE_THRESH_MODE_ORI_MID_ORI)
        .export_values();

    py::enum_<IVE_THRESH_S16_MODE_E>(m, "ThreshS16Mode")
        .value("S16_TO_S8_MIN_MID_MAX", IVE_THRESH_S16_MODE_S16_TO_S8_MIN_MID_MAX)
        .value("S16_TO_S8_MIN_ORI_MAX", IVE_THRESH_S16_MODE_S16_TO_S8_MIN_ORI_MAX)
        .value("S16_TO_U8_MIN_MID_MAX", IVE_THRESH_S16_MODE_S16_TO_U8_MIN_MID_MAX)
        .value("S16_TO_U8_MIN_ORI_MAX", IVE_THRESH_S16_MODE_S16_TO_U8_MIN_ORI_MAX)
        .export_values();

    py::enum_<IVE_THRESH_U16_MODE_E>(m, "ThreshU16Mode")
        .value("U16_TO_U8_MIN_MID_MAX", IVE_THRESH_U16_MODE_U16_TO_U8_MIN_MID_MAX)
        .value("U16_TO_U8_MIN_ORI_MAX", IVE_THRESH_U16_MODE_U16_TO_U8_MIN_ORI_MAX)
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

    py::enum_<IVE_8BIT_TO_8BIT_MODE_E>(m, "Cast8to8Mode")
        .value("S8_TO_S8", IVE_8BIT_TO_8BIT_MODE_S8_TO_S8)
        .value("S8_TO_U8_ABS", IVE_8BIT_TO_8BIT_MODE_S8_TO_U8_ABS)
        .value("S8_TO_U8_BIAS", IVE_8BIT_TO_8BIT_MODE_S8_TO_U8_BIAS)
        .value("U8_TO_U8", IVE_8BIT_TO_8BIT_MODE_U8_TO_U8)
        .export_values();

    py::enum_<IVE_MAP_MODE_E>(m, "MapMode")
        .value("U8", IVE_MAP_MODE_U8)
        .value("S16", IVE_MAP_MODE_S16)
        .value("U16", IVE_MAP_MODE_U16);

    py::enum_<IVE_EQUALIZE_MODE_E>(m, "EqualizeMode")
        .value("HIST", IVE_EQUALIZE_MODE_HIST)
        .value("EQHIST_WITH_EXT_HIST", IVE_EQUALIZE_MODE_EQHIST_WITH_EXT_HIST)
        .value("EQHIST", IVE_EQUALIZE_MODE_EQHIST);

    py::enum_<IVE_CCL_MODE_E>(m, "CclMode")
        .value("C4", IVE_CCL_MODE_4C)
        .value("C8", IVE_CCL_MODE_8C);

    py::enum_<IVE_NORM_GRAD_OUT_CTRL_E>(m, "NormGradOutCtrl")
        .value("HOR_AND_VER", IVE_NORM_GRAD_OUT_CTRL_HOR_AND_VER)
        .value("HOR", IVE_NORM_GRAD_OUT_CTRL_HOR)
        .value("VER", IVE_NORM_GRAD_OUT_CTRL_VER)
        .value("COMBINE", IVE_NORM_GRAD_OUT_CTRL_COMBINE);
        
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

    py::enum_<IVE_SAD_OUT_MODE_E>(m, "SadOutMode")
        .value("SAD", IVE_SAD_OUT_MODE_SAD)
        .value("BOTH", IVE_SAD_OUT_MODE_BOTH)
        .value("THRESH", IVE_SAD_OUT_MODE_THRESH);

    py::enum_<IVE_SAD_OUT_BITS_E>(m, "SadOutBits")
        .value("BITS8", IVE_SAD_OUT_8_BITS)
        .value("BITS16", IVE_SAD_OUT_16_BITS);
        
    py::class_<IVEModel>(m, "IVEModel", "Manages a persistent memory block for stateful IVE algorithms like GMM or background modeling.")
        .def(py::init<int, int, int>(), "width"_a, "height"_a, "model_size"_a = 0,
             "Creates a model bound to the exact source dimensions. model_size=0 selects automatic capacity.")
        .def_property_readonly("width", &IVEModel::width)
        .def_property_readonly("height", &IVEModel::height)
        .def("reset", &IVEModel::reset, py::call_guard<py::gil_scoped_release>(),
             "Clears model state without changing dimensions. Reinitialize GMM/GMM2 with first_frame=True; retrain codebook models.");

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
        .def("filter_ex", &IVE::filter_ex, "src"_a, "mask"_a, "norm"_a = 0, "out_format"_a = IVE_IMAGE_TYPE_U8C1,
             py::return_value_policy::move, py::call_guard<py::gil_scoped_release>(),
             "One native 5x5 U8-input filter. mask contains 25 signed 8-bit coefficients; norm is an explicit right shift in [0,15]. "
             "Output format is U8C1, S8C1, U16C1 or S16C1. Border and signed rounding follow the hardware.")
        .def("sobel_ex", &IVE::sobel_ex, "src"_a, "mask"_a, "norm"_a = 0,
             "out_ctrl"_a = IVE_SOBEL_OUT_CTRL_BOTH, "out_format"_a = IVE_IMAGE_TYPE_S16C1,
             py::return_value_policy::move, py::call_guard<py::gil_scoped_release>(),
             "One native Sobel operation with explicit 25-coefficient mask and normalization shift in [0,15]. "
             "Returns (horizontal,vertical); an unrequested direction is an invalid ImageBuffer. "
             "U8C1, S8C1, U16C1 and S16C1 outputs retain their native dtype.")
        .def("sobel", &IVE::sobel, "src"_a, "out_ctrl"_a = IVE_SOBEL_OUT_CTRL_BOTH, "out_format"_a = IVE_IMAGE_TYPE_S16C1,
             py::return_value_policy::move,
             py::call_guard<py::gil_scoped_release>(),
             "Sobel edge detection. src: GRAY8. Returns (horizontal_edges, vertical_edges). When out_format=S16C1 each result is S16C1 - unpack and pass to cast_16bit_to_8bit for display; when U8C1 each result is GRAY8.")
        .def("canny", &IVE::canny, "src"_a, "high_thresh"_a, "low_thresh"_a,
             py::return_value_policy::move,
             py::call_guard<py::gil_scoped_release>(),
             "Canny edge detection. src must be GRAY8. high_thresh and low_thresh typically in 0-255.")
        .def("canny_ex", &IVE::canny_ex, "src"_a, "high_thresh"_a, "low_thresh"_a,
             "mask"_a = std::vector<int8_t>{}, py::return_value_policy::move,
             py::call_guard<py::gil_scoped_release>(),
             "Canny with a configurable 25-element signed 8-bit gradient mask. Empty mask selects the default 3x3 Sobel kernel. "
             "Requires low_thresh<=high_thresh; choose thresholds for the gradient scale of the supplied kernel.")
        .def("mag_and_ang", &IVE::mag_and_ang, "src"_a, "threshold"_a = 0, "return_magnitude"_a = true,
             py::return_value_policy::move,
             py::call_guard<py::gil_scoped_release>(),
             "Gradient magnitude/angle. return_magnitude=True returns U16C1 (use Cast16to8Mode.U16_TO_U8 for display); False returns U8C1. "
             "threshold is an inclusive minimum magnitude: values below it and their angles are zeroed on the CPU because the runtime ignores its threshold field.")
        .def("mag_and_ang_ex", &IVE::mag_and_ang_ex, "src"_a, "threshold"_a, "mask"_a,
             py::return_value_policy::move, py::call_guard<py::gil_scoped_release>(),
             "Hardware gradient magnitude/angle with an explicit 25-coefficient mask. Returns (U16C1 magnitude, GRAY8 angle). "
             "threshold is an inclusive minimum; values below it and their angles are suppressed on the CPU.")
        .def("morphology", &IVE::morphology, "src"_a, "mask"_a, "dilate_op"_a,
             py::return_value_policy::move, py::call_guard<py::gil_scoped_release>(),
             "Binary morphology with a 25-element 0/1 mask. src must contain only 0 and 255.")
        .def("dilate", &IVE::dilate, "src"_a, "kernel_size"_a = 3,
             py::return_value_policy::move,
             py::call_guard<py::gil_scoped_release>(),
             "Binary morphological dilation. `src` must be GRAY8 containing only 0 and 255. `kernel_size` can be 3 or 5.")
        .def("erode", &IVE::erode, "src"_a, "kernel_size"_a = 3,
             py::return_value_policy::move,
             py::call_guard<py::gil_scoped_release>(),
             "Binary morphological erosion. `src` must be GRAY8 containing only 0 and 255. `kernel_size` can be 3 or 5.")
        .def("ordered_stat_filter", &IVE::ordered_stat_filter, "src"_a, "mode"_a,
             py::return_value_policy::move,
             py::call_guard<py::gil_scoped_release>(),
             "Performs a 3x3 ordered statistic filter (Median, Max, Min). `src` must be GRAY8.")
        .def("add", &IVE::add, "src1"_a, "src2"_a,
             py::return_value_policy::move, py::call_guard<py::gil_scoped_release>(), "Pixel-wise addition. Returns GRAY8; do not pass to cast_16bit_to_8bit.")
        .def("add_weighted", &IVE::add_weighted, "src1"_a, "src2"_a, "weight1"_a = 32768, "weight2"_a = 32768,
             py::return_value_policy::move, py::call_guard<py::gil_scoped_release>(),
             "One hardware weighted add. Unsigned Q0.16 weights range from 0 to 65535; 32768 represents 0.5. Returns GRAY8.")
        .def("sub", &IVE::sub, "src1"_a, "src2"_a, "mode"_a = IVE_SUB_MODE_ABS,
             py::return_value_policy::move, py::call_guard<py::gil_scoped_release>(), "Pixel-wise subtraction. Returns GRAY8; do not pass to cast_16bit_to_8bit. Mode: ABS or SHIFT.")
        .def("logic_op", &IVE::logic_op, "src1"_a, "src2"_a, "op"_a,
             py::return_value_policy::move, py::call_guard<py::gil_scoped_release>(), "Performs a pixel-wise logical operation (AND, OR, XOR) on two images.")
        .def("threshold",
             [](IVE& self, const ImageBuffer& src, py::args args, py::kwargs kwargs) {
                 int low_thresh = 0;
                 int high_thresh = 255;
                 IVE_THRESH_MODE_E mode = IVE_THRESH_MODE_BINARY;
                 int min_val = 0;
                 int mid_val = 0;
                 int max_val = 255;

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
                 if (kwargs.contains("min_val")) min_val = kwargs["min_val"].cast<int>();
                 if (kwargs.contains("mid_val")) mid_val = kwargs["mid_val"].cast<int>();
                 if (kwargs.contains("max_val")) max_val = kwargs["max_val"].cast<int>();
                 for (const auto& item : kwargs) {
                     const std::string name = item.first.cast<std::string>();
                     if (name != "low_thresh" && name != "low" && name != "high_thresh" &&
                         name != "high" && name != "mode" && name != "min_val" &&
                         name != "mid_val" && name != "max_val") {
                         throw std::invalid_argument("Unknown threshold argument: " + name);
                     }
                 }

                 py::gil_scoped_release release;
                 return self.threshold_u8(src, low_thresh, high_thresh, mode, min_val, mid_val, max_val);
             },
             "src"_a,
             py::return_value_policy::move,
             "Thresholding on GRAY8. kwargs accept low/low_thresh, high/high_thresh, mode, min_val, mid_val, max_val. "
             "Values must be in [0,255]. MIN_MID_MAX and ORI_MID_ORI use one hardware Map operation; other modes use one native threshold. Returns GRAY8.")
        .def("threshold_u8", &IVE::threshold_u8,
             "src"_a, "low_thresh"_a, "high_thresh"_a = 255, "mode"_a = IVE_THRESH_MODE_BINARY,
             "min_val"_a = 0, "mid_val"_a = 0, "max_val"_a = 255,
             py::return_value_policy::move, py::call_guard<py::gil_scoped_release>(),
             "Full U8 threshold controls. Regions are src<=low, low<src<=high, src>high. "
             "MIN_MID_MAX and ORI_MID_ORI use one hardware Map operation to avoid runtime region-selection defects; "
             "other modes use one native threshold. Returns GRAY8.")
        .def("threshold_s16", &IVE::threshold_s16,
             "src"_a, "low_thresh"_a, "high_thresh"_a,
             "mode"_a = IVE_THRESH_S16_MODE_S16_TO_U8_MIN_MID_MAX,
             "min_val"_a = 0, "mid_val"_a = 0, "max_val"_a = 255,
             py::return_value_policy::move, py::call_guard<py::gil_scoped_release>(),
             "One native S16 threshold operation. src must be S16C1; thresholds are in [-32768,32767]. "
             "S8 modes return S8C1 and require constants in [-128,127]; U8 modes return GRAY8 and require [0,255]. "
             "MIN_ORI_MAX preserves the middle region through the runtime's 16-to-8-bit conversion.")
        .def("threshold_u16", &IVE::threshold_u16,
             "src"_a, "low_thresh"_a, "high_thresh"_a,
             "mode"_a = IVE_THRESH_U16_MODE_U16_TO_U8_MIN_MID_MAX,
             "min_val"_a = 0, "mid_val"_a = 0, "max_val"_a = 255,
             py::return_value_policy::move, py::call_guard<py::gil_scoped_release>(),
             "One native U16 threshold operation. src must be U16C1; thresholds are in [0,65535], constants in [0,255]. "
             "Returns GRAY8. MIN_ORI_MAX preserves the middle region through the runtime's 16-to-8-bit conversion.")
        .def("cast_16bit_to_8bit", &IVE::cast_16bit_to_8bit, "src"_a, "mode"_a,
             py::return_value_policy::move, py::call_guard<py::gil_scoped_release>(), "Casts 16-bit (S16C1/U16C1) to 8-bit. Input must be from sobel(S16C1), mag_and_ang(magnitude), norm_grad, or sad (first element); not from filter, add, sub, threshold, gmm.")
        .def("cast_16bit_ex", &IVE::cast_16bit_ex, "src"_a, "mode"_a,
             "numerator"_a = 1, "denominator"_a = 1, "bias"_a = 0,
             py::return_value_policy::move, py::call_guard<py::gil_scoped_release>(),
             "16-bit scaling with explicit ratio and post-scale bias. Signed modes require S16C1; U16_TO_U8 requires U16C1. "
             "ABS applies absolute value before scaling. Numerator is [0,255], denominator [1,65535], bias [-128,127]. "
             "Uses one native unsigned scaling call; signed modes add CPU absolute-value staging and sign restoration. "
             "The Q8.24 ratio and nearest rounding follow the hardware; signed half ties are restored toward positive infinity.")
        .def("cast_8bit_ex", &IVE::cast_8bit_ex, "src"_a, "mode"_a,
             "numerator"_a = 1, "denominator"_a = 1, "bias"_a = 0,
             py::return_value_policy::move, py::call_guard<py::gil_scoped_release>(),
             "8-bit scaling through one hardware Map and a CPU-generated 256-entry LUT. Signed modes require S8C1; U8_TO_U8 requires GRAY8. "
             "ABS applies absolute value first. Rounds the Q8.24-scaled value to nearest, with ties toward positive infinity, "
             "then adds bias and saturates to the output dtype. Numerator is [0,255], denominator [1,255], bias [-128,127].")
        .def("hist", &IVE::hist, "src"_a, py::call_guard<py::gil_scoped_release>(), "Calculates the histogram of a GRAY8 image. Returns a list of 256 integers.")
        .def("equalize_hist", &IVE::equalize_hist, "src"_a, py::return_value_policy::move, py::call_guard<py::gil_scoped_release>(), "Performs histogram equalization on a GRAY8 image.")
        .def("equalize_hist_ex", &IVE::equalize_hist_ex, "src"_a, "mode"_a = IVE_EQUALIZE_MODE_EQHIST,
             "histogram"_a = std::vector<uint32_t>{}, py::return_value_policy::move,
             py::call_guard<py::gil_scoped_release>(),
             "HIST uses one native histogram call and returns (invalid ImageBuffer,256 counts). "
             "EQHIST and EQHIST_WITH_EXT_HIST use one native equalization call and return (GRAY8,empty counts). "
             "External counts require 256 bins summing to the image pixel count. Equalization rounding follows the hardware.")
        .def("integral", &IVE::integral, "src"_a, "mode"_a = IVE_INTEG_OUT_CTRL_COMBINE,
             py::return_value_policy::move, py::call_guard<py::gil_scoped_release>(), "Integral image. mode: COMBINE, SUM, or SQSUM. Output U64C1.")
        .def("ccl", &IVE::ccl, "src"_a, "min_area"_a = 100,
             py::call_guard<py::gil_scoped_release>(),
             "Connected Components Labeling on a binarized GRAY8 image. Blobs with area < min_area are filtered. Returns list of Blob.")
        .def("ccl_config", &IVE::ccl_config, "src"_a, "min_area"_a = 100, "mode"_a = IVE_CCL_MODE_8C,
             "init_area_threshold"_a = 0, "step"_a = 1, py::call_guard<py::gil_scoped_release>(),
             "Connected components with explicit connectivity and hardware area controls. Returns filtered Blob values.")
        .def("ncc_raw", &IVE::ncc_raw, "src1"_a, "src2"_a, py::call_guard<py::gil_scoped_release>(),
             "One native NCC operation. Returns integer (sum(A*B),sum(A*A),sum(B*B)); no mean subtraction or normalization is applied.")
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
        .def("csc", &IVE::csc, "src"_a, "mode"_a, "input_full_range"_a = false, py::return_value_policy::move, py::call_guard<py::gil_scoped_release>(), "Hardware color conversion. YUV input defaults to video range; set input_full_range=True for full-range YUV.")
        .def("csc_bytes",
             [](IVE& self, py::bytes data, int width, int height, IVE_IMAGE_TYPE_E src_type,
                IVE_IMAGE_TYPE_E dst_type, IVE_CSC_MODE_E mode, bool input_full_range, bool bgr_order, bool vu_order) {
                 const std::string storage = data.cast<std::string>();
                 std::vector<uint8_t> input(storage.begin(), storage.end());
                 std::vector<uint8_t> output;
                 {
                     py::gil_scoped_release release;
                     output = self.csc_bytes(std::move(input), width, height, src_type, dst_type, mode,
                                             input_full_range, bgr_order, vu_order);
                 }
                 return py::bytes(reinterpret_cast<const char*>(output.data()), output.size());
             },
             "data"_a, "width"_a, "height"_a, "src_type"_a, "dst_type"_a, "mode"_a,
             "input_full_range"_a = false, "bgr_order"_a = false, "vu_order"_a = false,
             "Compact bytes-to-bytes CSC with explicit ImageTypeIVE layouts, separate from the ImageBuffer csc helper. "
             "YUV420SP is Y plus interleaved chroma and requires even width/height; YUV422SP requires even width. "
             "PACKAGE is interleaved color; PLANAR is three consecutive width*height planes. "
             "Color order defaults to RGB and UV; bgr_order selects BGR on RGB inputs/outputs and vu_order selects VU on YUV. "
             "HSV outputs always keep H,S,V order. input_full_range applies only to YUV input. "
             "Planar layouts use CPU packing/unpacking around one hardware CSC. 422 uses row duplication/extraction "
             "around one 420 CSC and is limited to height<=2048. Input bytes must match the compact layout exactly.")
        
        .def("yuv_to_rgb",
             [](IVE& self, const ImageBuffer& src, py::kwargs kwargs) {
                 const bool input_full_range = kwargs.contains("input_full_range") && kwargs["input_full_range"].cast<bool>();
                 if (kwargs.contains("mode")) {
                     IVE_CSC_MODE_E mode = kwargs["mode"].cast<IVE_CSC_MODE_E>();
                     return self.csc(src, mode, input_full_range);
                 }
                 bool full_range = true;
                 if (kwargs.contains("full_range")) full_range = kwargs["full_range"].cast<bool>();
                 return self.yuv_to_rgb(src, full_range, input_full_range);
             },
             "src"_a, py::return_value_policy::move,
             "Converts video-range NV12/NV21 to RGB. full_range controls RGB output range; set input_full_range=True for full-range YUV input.")
        .def("yuv_to_hsv",
             [](IVE& self, const ImageBuffer& src, py::kwargs kwargs) {
                 const bool input_full_range = kwargs.contains("input_full_range") && kwargs["input_full_range"].cast<bool>();
                 if (kwargs.contains("mode")) {
                     IVE_CSC_MODE_E mode = kwargs["mode"].cast<IVE_CSC_MODE_E>();
                     return self.csc(src, mode, input_full_range);
                 }
                 bool full_range = true;
                 if (kwargs.contains("full_range")) full_range = kwargs["full_range"].cast<bool>();
                 return self.yuv_to_hsv(src, full_range, input_full_range);
             },
             "src"_a, py::return_value_policy::move,
             "Converts video-range NV12/NV21 to HSV888. Set input_full_range=True for full-range YUV input. H, S and V use [0,255].")
        .def("rgb_to_yuv", &IVE::rgb_to_yuv, "src"_a, "full_range"_a = true, py::return_value_policy::move,
             py::call_guard<py::gil_scoped_release>(),
             "Converts RGB/BGR to NV12. full_range selects full-range YUV instead of video-range YUV.")
        .def("rgb_to_hsv", &IVE::rgb_to_hsv, "src"_a, "full_range"_a = true, py::return_value_policy::move,
             py::call_guard<py::gil_scoped_release>(),
             "Converts RGB/BGR to HSV888. H, S and V use [0,255]; NumPy exports preserve HSV channel order.")

        .def("dma", &IVE::dma_transfer, "src"_a, "mode"_a = IVE_DMA_MODE_DIRECT_COPY,
             "value"_a = uint64_t{0}, "hor_seg_size"_a = 4, "elem_size"_a = 1, "ver_seg_rows"_a = 1,
             "output_width"_a = 0, "output_height"_a = 0, py::return_value_policy::move,
             py::call_guard<py::gil_scoped_release>(),
             "Alias for dma_transfer. dma(src) retains the direct-copy default; accepts only GRAY8 byte planes.")
        .def("dma_transfer", &IVE::dma_transfer, "src"_a, "mode"_a = IVE_DMA_MODE_DIRECT_COPY,
             "value"_a = uint64_t{0}, "hor_seg_size"_a = 4, "elem_size"_a = 1, "ver_seg_rows"_a = 1,
             "output_width"_a = 0, "output_height"_a = 0, py::return_value_policy::move,
             py::call_guard<py::gil_scoped_release>(),
             "One hardware DMA operation on a GRAY8 byte plane. DIRECT_COPY preserves dimensions. "
             "INTERVAL_COPY keeps the first elem_size bytes of each hor_seg_size-byte segment and one row "
             "per ver_seg_rows group, starting at row zero; source dimensions must divide exactly. "
             "Its output is (src.width/hor_seg_size*elem_size, src.height/ver_seg_rows). "
             "SET_3BYTE/SET_8BYTE create a new filled image, leaving src intact. value supplies the repeated "
             "little-endian 3-byte/8-byte pattern; SET_3BYTE requires value<=0xffffff. "
             "Zero output dimensions select the derived size; fill modes may specify a different output size. "
             "Other controls are used only by their corresponding mode.")
        .def("cast_8bit_to_8bit", &IVE::cast_8bit_to_8bit, "src"_a, "bias"_a, "numerator"_a, "denominator"_a, py::return_value_policy::move,
             py::call_guard<py::gil_scoped_release>(),
             "Scales an 8-bit image using the formula: dst = (src * numerator / denominator) + bias.")
        .def("map", &IVE::map, "src"_a, "lut"_a, py::return_value_policy::move,
             py::call_guard<py::gil_scoped_release>(),
             "Pixel LUT mapping. lut: list of 256 integers in [0,255]; output[x] = lut[src[x]].")
        .def("map_typed", &IVE::map_typed, "src"_a, "lut"_a, "mode"_a = IVE_MAP_MODE_U8,
             py::return_value_policy::move, py::call_guard<py::gil_scoped_release>(),
             "One native LUT operation on GRAY8 input. Exactly 256 entries; MapMode chooses U8, S16 or U16 output and entry range.")
        .def("gmm", &IVE::gmm, "src"_a, "model"_a, "first_frame"_a = false,
             "GMM background subtraction. Returns (foreground, background); both GRAY8 - do not pass to cast_16bit_to_8bit.")
        .def("gmm2", &IVE::gmm2, "src"_a, "factor"_a, "model"_a, "first_frame"_a = false,
             "GMM2 with factor image. Returns (foreground, background); both GRAY8.")
        .def("gmm_config", &IVE::gmm_config, "src"_a, "model"_a, "max_models"_a = 3, "first_frame"_a = false,
             py::return_value_policy::move, py::call_guard<py::gil_scoped_release>(),
             "GMM with 1..5 models. Returns GRAY8 (foreground, background, matched). "
             "Multi-model operation uses separate state buffers on verified RV1106 runtimes. "
             "Initialize or change model count with first_frame=True.")
        .def("gmm2_config", &IVE::gmm2_config, "src"_a, "factor"_a, "model"_a,
             "max_models"_a = 3, "first_frame"_a = false,
             py::return_value_policy::move, py::call_guard<py::gil_scoped_release>(),
             "GMM2 with a same-size U16C1 factor image and 1..5 models. Returns "
             "GRAY8 (foreground, background, matched). Initialize or change model count with first_frame=True.")
        .def("lbp", &IVE::lbp, "src"_a, "abs_mode"_a = false, "threshold"_a = 8, py::return_value_policy::move,
             "Local Binary Pattern. NORMAL threshold is in [-128,127]; ABS threshold is in [0,255]. Explicit zero is preserved.")
        .def("norm_grad", &IVE::norm_grad, "src"_a,
             py::call_guard<py::gil_scoped_release>(),
             "Normalized gradient. Returns (horizontal_grad, vertical_grad); both S16C1 - use cast_16bit_to_8bit for display.")
        .def("norm_grad_config", &IVE::norm_grad_config, "src"_a,
             "out_ctrl"_a = IVE_NORM_GRAD_OUT_CTRL_HOR_AND_VER, "norm"_a = 7, "mask"_a = std::vector<int8_t>{},
             py::return_value_policy::move, py::call_guard<py::gil_scoped_release>(),
             "Returns (S8 horizontal,S8 vertical,U16 combined); unrequested outputs are invalid ImageBuffers. "
             "COMBINE packs the hardware H/V results in software: low byte H, high byte V.")
        .def("lk_optical_flow", &IVE::lk_optical_flow, "prev_img"_a, "next_img"_a, "points"_a,
             py::call_guard<py::gil_scoped_release>(),
             "Performs pyramidal Lucas-Kanade optical flow. points: integer (x,y) coordinates inside the image. "
             "Returns MotionVector values with status=0 for success and -1 for failure; mv_x/mv_y are S9.7 "
             "(divide by 128 for pixel displacement). Up to 500 points; image dimensions must be at least 64x64.")
        .def("lk_optical_flow_single", &IVE::lk_optical_flow_single, "prev_img"_a, "next_img"_a, "points"_a,
             "iterations"_a = 20, "min_eigenvalue"_a = 100, "epsilon"_a = 32,
             py::call_guard<py::gil_scoped_release>(), "Single-level hardware LK through max_level=0 of the public configurable entry, with explicit convergence controls.")
        .def("lk_optical_flow_config", &IVE::lk_optical_flow_config, "prev_img"_a, "next_img"_a, "points"_a,
             "max_level"_a = 3, "iterations"_a = 20, "min_eigenvalue"_a = 100, "epsilon"_a = 32,
             "initial_points"_a = std::vector<std::tuple<int, int>>{}, py::call_guard<py::gil_scoped_release>(),
             "Pyramidal LK controls. Returns (motion vectors,U9Q7 errors); divide errors by 128. "
             "Nonempty initial_points supplies initial absolute coordinates and must match points in length.")
        .def("st_corner", &IVE::st_corner, "src"_a, "max_corners"_a = 200, "min_dist"_a = 10, "quality_level"_a = 25,
             py::call_guard<py::gil_scoped_release>(),
             "Performs Shi-Tomasi corner detection. Returns a list of (x, y) corner tuples.")
        .def("st_corner_candidates", &IVE::st_corner_candidates, "src"_a, "quality_level"_a = 25,
             py::call_guard<py::gil_scoped_release>(), "Returns hardware Shi-Tomasi candidates as (x,y,lambda).")
        .def("st_corner_sorted", &IVE::st_corner_sorted, "src"_a, "max_corners"_a = 200,
             "min_dist"_a = 10, "quality_level"_a = 25, py::call_guard<py::gil_scoped_release>(),
             "Returns selected Shi-Tomasi corners as (x,y,lambda).")
        .def("match_bg_model", &IVE::match_bg_model, "current_img"_a, "bg_model"_a, "frame_num"_a,
             "Trains the background codebook using a GRAY8 frame. Updates bg_model in place and returns None.")
        .def("update_bg_model", &IVE::update_bg_model, "current_img"_a, "bg_model"_a, "frame_num"_a, py::return_value_policy::move,
             "Updates the background codebook and returns (foreground_status, background_image), both GRAY8.")
        .def("sad", &IVE::sad, "src1"_a, "src2"_a, "mode"_a, "threshold"_a, "min_val"_a = 0, "max_val"_a = 255,
             py::call_guard<py::gil_scoped_release>(),
             "Sum of Absolute Differences. Returns (sad_image U16C1, threshold_image U8C1); cast sad_image for 8-bit display.")
        .def("sad_config", &IVE::sad_config, "src1"_a, "src2"_a, "mode"_a, "threshold"_a,
             "min_val"_a = 0, "max_val"_a = 255, "out_mode"_a = IVE_SAD_OUT_MODE_BOTH,
             "out_bits"_a = IVE_SAD_OUT_16_BITS, py::return_value_policy::move,
             py::call_guard<py::gil_scoped_release>(),
             "SAD with explicit output selection; unrequested tuple elements are invalid ImageBuffers. "
             "8-bit output is the full U16 SAD shifted right by 8. Thresholding always uses the full U16 value. "
             "Threshold output uses a hardware SAD call followed by a hardware U16 threshold call.")
        .def("create_pyramid", &IVE::create_pyramid, "src"_a, "levels"_a,
             py::call_guard<py::gil_scoped_release>(),
             "Builds an image pyramid with levels levels. Returns list of ImageBuffer from full size down.");
}
#endif
