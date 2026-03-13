// SPDX-License-Identifier: LGPL-3.0-or-later
#include "internal/bindings_common.h"
#include "visiong/common/pixel_format.h"
#include "visiong/core/BufferStateMachine.h"
#include <cstring>
#include <memory>

void bind_core_types(py::module_& m) {
    py::class_<Blob>(m, "Blob", "Connected component from find_blobs or IVE ccl: bounding box (x,y,w,h), center (cx,cy), pixel count, optional label code.")
        .def(py::init<int, int, int, int, int, int, int, unsigned int>(), "x"_a, "y"_a, "w"_a, "h"_a, "cx"_a, "cy"_a, "pixels"_a, "code"_a = 0)
        .def_property_readonly("x",      [](const Blob &b) { return b.x; })
        .def_property_readonly("y",      [](const Blob &b) { return b.y; })
        .def_property_readonly("w",      [](const Blob &b) { return b.w; })
        .def_property_readonly("h",      [](const Blob &b) { return b.h; })
        .def_property_readonly("cx",     [](const Blob &b) { return b.cx; })
        .def_property_readonly("cy",     [](const Blob &b) { return b.cy; })
        .def_property_readonly("pixels", [](const Blob &b) { return b.pixels; })
        .def_property_readonly("code",   [](const Blob &b) { return b.code; })
        .def_property_readonly("rect",   &Blob::rect)
        .def_property_readonly("area",   &Blob::area)
        .def("__repr__", [](const Blob &b) {
            return "<Blob rect=(" + std::to_string(b.x) + "," + std::to_string(b.y) + "," + std::to_string(b.w) + "," + std::to_string(b.h) + ")>";
        });

    py::class_<Line>(m, "Line", "Line segment from find_lines: endpoints (x1,y1)-(x2,y2), magnitude, theta (angle), rho.")
        .def_readonly("x1", &Line::x1).def_readonly("y1", &Line::y1).def_readonly("x2", &Line::x2).def_readonly("y2", &Line::y2).def_readonly("magnitude", &Line::magnitude).def_readonly("theta", &Line::theta).def_readonly("rho", &Line::rho)
        .def("__repr__", [](const Line &l) { return "<Line p1=(" + std::to_string(l.x1) + "," + std::to_string(l.y1) + ") p2=(" + std::to_string(l.x2) + "," + std::to_string(l.y2) + ") mag=" + std::to_string(l.magnitude) + ">"; });

    py::class_<Circle>(m, "Circle", "Circle from find_circles: center (cx,cy), radius r, magnitude.")
        .def_readonly("cx", &Circle::cx).def_readonly("cy", &Circle::cy).def_readonly("r", &Circle::r).def_readonly("magnitude", &Circle::magnitude)
        .def("__repr__", [](const Circle &c) { return "<Circle center=(" + std::to_string(c.cx) + "," + std::to_string(c.cy) + ") r=" + std::to_string(c.r) + ">"; });
    
    py::class_<QRCode>(m, "QRCode")
        .def_property_readonly("corners", [](const QRCode& qr){ return qr.corners; }, "A list of (x, y) tuples representing the four corners of the QR code.")
        .def_property_readonly("payload", [](const QRCode& qr){ return py::bytes(qr.payload); }, "The decoded content of the QR code as a bytes object.")
        .def("__repr__", [](const QRCode &q) {
            return "<QRCode payload='" + q.payload + "'>";
        });
}


namespace {

py::array image_to_numpy_array(ImageBuffer& img, bool copy) {
    py::array arr = make_image_buffer_numpy_view(img);
    if (copy) {
        return arr.attr("copy")().cast<py::array>();
    }
    return arr;
}

ImageBuffer image_buffer_from_numpy_impl(py::array arr, const std::string& fmt, bool copy, bool strict_zero_copy) {
    py::array prepared;
    if (copy) {
        auto tmp = py::array_t<unsigned char, py::array::c_style | py::array::forcecast>::ensure(arr);
        if (!tmp) {
            throw std::runtime_error("from_numpy: expected a uint8 NumPy array.");
        }
        prepared = tmp;
    } else {
        prepared = py::array::ensure(arr);
        if (!prepared) {
            throw std::runtime_error("from_numpy_zero_copy: expected a NumPy array.");
        }
        if (!(prepared.flags() & py::array::c_style)) {
            throw std::runtime_error("from_numpy_zero_copy: input array must be C-contiguous.");
        }
        py::buffer_info check = prepared.request();
        if (check.format != py::format_descriptor<unsigned char>::format() ||
            check.itemsize != static_cast<ssize_t>(sizeof(unsigned char))) {
            throw std::runtime_error("from_numpy_zero_copy: input array must have dtype=uint8.");
        }
    }

    py::buffer_info info = prepared.request();
    if (info.ndim < 2 || info.ndim > 3) {
        throw std::runtime_error("from_numpy: Input NumPy array must be 2D or 3D.");
    }

    const int h = static_cast<int>(info.shape[0]);
    const int w = static_cast<int>(info.shape[1]);
    const int channels = (info.ndim == 3) ? static_cast<int>(info.shape[2]) : 1;

    std::string fmt_lower = visiong::to_lower_copy(fmt);

    PIXEL_FORMAT_E format;
    if (fmt_lower == "auto") {
        if (info.ndim == 2 || channels == 1) {
            format = kGray8;
        } else if (channels == 3) {
            format = RK_FMT_BGR888;
        } else if (channels == 4) {
            format = RK_FMT_BGRA8888;
        } else {
            throw std::runtime_error("from_numpy: format='auto' requires 2D (grayscale), 3-channel, or 4-channel array.");
        }
    } else {
        format = visiong::parse_pixel_format(fmt);
        int expected_ch = 1;
        if (format == RK_FMT_BGR888 || format == RK_FMT_RGB888) expected_ch = 3;
        else if (format == RK_FMT_BGRA8888 || format == RK_FMT_RGBA8888) expected_ch = 4;
        else if (format == kGray8) expected_ch = 1;
        if (expected_ch != channels) {
            throw std::runtime_error("from_numpy: format '" + fmt + "' expects " + std::to_string(expected_ch) +
                                     " channel(s), but array has " + std::to_string(channels) + ".");
        }
    }

    unsigned char* ptr = static_cast<unsigned char*>(info.ptr);
    const size_t data_size = static_cast<size_t>(info.size) * static_cast<size_t>(info.itemsize);

    if ((w % 2 != 0) || (h % 2 != 0)) {
        if (!copy || strict_zero_copy) {
            throw std::runtime_error("from_numpy_zero_copy: width and height must both be even for zero-copy hardware-friendly ImageBuffer wrapping.");
        }
        const int padded_w = w + (w % 2);
        const int padded_h = h + (h % 2);
        std::vector<unsigned char> padded_data(static_cast<size_t>(padded_w) * padded_h * channels, 0);
        for (int y = 0; y < h; ++y) {
            std::memcpy(padded_data.data() + static_cast<size_t>(y) * padded_w * channels,
                        ptr + static_cast<size_t>(y) * w * channels,
                        static_cast<size_t>(w) * channels);
        }
        return ImageBuffer(padded_w, padded_h, format, std::move(padded_data));
    }

    if (!copy) {
        std::shared_ptr<void> keep_alive(new py::object(prepared), [](void* p) { delete static_cast<py::object*>(p); });
        return ImageBuffer(w, h, format, ptr, data_size, std::move(keep_alive));
    }

    std::vector<unsigned char> data(ptr, ptr + data_size);
    return ImageBuffer(w, h, format, std::move(data));
}

}  // namespace

void bind_image_buffer(py::module_& m) {
    py::class_<ImageBuffer>(m, "ImageBuffer", py::buffer_protocol())
        .def(py::init<>())
        .def_static("create", [](int w, int h, const std::string& fmt, py::object color_obj) {
            PIXEL_FORMAT_E pixel_format = visiong::parse_pixel_format(fmt);
        
            if (!py::isinstance<py::tuple>(color_obj)) {
                throw py::type_error("Color must be a tuple of 3 (RGB) or 4 (RGBA) integers.");
            }
        
            py::tuple color_tuple = color_obj.cast<py::tuple>();
            size_t len = color_tuple.size();
        
            if (len == 3) {
                auto rgb = color_tuple.cast<std::tuple<unsigned char, unsigned char, unsigned char>>();
                return ImageBuffer::create(w, h, pixel_format, rgb);
            } else if (len == 4) {
                auto rgba = color_tuple.cast<std::tuple<unsigned char, unsigned char, unsigned char, unsigned char>>();
                return ImageBuffer::create(w, h, pixel_format, rgba);
            } else {
                throw py::type_error("Color tuple must have 3 (RGB) or 4 (RGBA) elements.");
            }
        }, "width"_a, "height"_a, "format"_a, "color"_a = py::make_tuple(0, 0, 0), py::return_value_policy::move,
        "Creates a new ImageBuffer filled with the given color. format: e.g. 'rgb888', 'bgr888', 'gray8'. color: (R,G,B) or (R,G,B,A).")
        .def_buffer([](ImageBuffer &img) -> py::buffer_info {
            if (!img.is_valid()) throw std::runtime_error("Cannot create a buffer from an invalid ImageBuffer.");
            visiong::bufstate::prepare_cpu_read(img);
            int bpp = get_bpp_for_format(img.format);
            if (bpp == 0 || bpp % 8 != 0) throw std::runtime_error("Buffer protocol not supported for this format.");
            int itemsize = bpp / 8;
            std::string format_py;
            ssize_t ndim;
            std::vector<ssize_t> shape, strides;

            switch (static_cast<int>(img.format)) {
                case static_cast<int>(kGray8):
                    format_py = py::format_descriptor<unsigned char>::format();
                    ndim = 2;
                    shape = { (ssize_t)img.height, (ssize_t)img.width };
                    strides = { (ssize_t)(img.w_stride * itemsize), (ssize_t)itemsize };
                    break;
                case RK_FMT_BGR888: case RK_FMT_RGB888:
                    format_py = py::format_descriptor<unsigned char>::format();
                    ndim = 3;
                    shape = { (ssize_t)img.height, (ssize_t)img.width, 3 };
                    strides = { (ssize_t)(img.w_stride * itemsize), (ssize_t)(3 * sizeof(unsigned char)), (ssize_t)sizeof(unsigned char) };
                    break;
                default:
                    format_py = py::format_descriptor<unsigned char>::format();
                    ndim = 1;
                    shape = { (ssize_t)img.get_size() };
                    strides = { (ssize_t)1 };
                    break;
            }
            // Keep buffer-protocol exports read-only so np.asarray(img) cannot / 保持 缓冲区-protocol exports read-only so np.asarray(img) cannot
            // mutate storage without passing through explicit write boundaries.
            return py::buffer_info(img.get_data(), sizeof(unsigned char), format_py, ndim, shape, strides, true);
        })
        .def("__array__", [](ImageBuffer &img, py::object dtype, py::object copy) -> py::object {
            py::array arr = image_to_numpy_array(img, false);
            const bool force_copy = !copy.is_none() && copy.cast<bool>();
            if (!dtype.is_none()) {
                return arr.attr("astype")(dtype, py::arg("copy") = force_copy);
            }
            if (force_copy) {
                return py::object(arr.attr("copy")());
            }
            return py::object(arr);
        }, "dtype"_a = py::none(), "copy"_a = py::none(),
        "Allows NumPy conversion via np.asarray(img). copy=False returns a read-only array pinned to the ImageBuffer lifetime; color exports use a BGR-shaped view over the current CPU-readable storage, which may be a cached converted backing store for non-BGR sources.")
        .def("to_numpy", [](ImageBuffer &img, bool copy) -> py::array {
            if (!img.is_valid()) throw std::runtime_error("to_numpy: Invalid ImageBuffer.");
            return image_to_numpy_array(img, copy);
        }, "copy"_a=false, "Returns a NumPy array: 2D (H,W) for grayscale, 3D (H,W,3) BGR-shaped for color. copy=False returns a read-only lifetime-pinned view over the current CPU-readable storage.")
        .def("numpy_view", [](ImageBuffer &img) -> py::array {
            if (!img.is_valid()) throw std::runtime_error("numpy_view: Invalid ImageBuffer.");
            return image_to_numpy_array(img, false);
        }, "Returns a read-only lifetime-pinned NumPy view over the current CPU-readable storage.")
        .def_static("from_numpy", [](py::array arr, const std::string& fmt, bool copy) {
            return image_buffer_from_numpy_impl(arr, fmt, copy, false);
        }, "array"_a, "format"_a = "auto", "copy"_a = true, py::return_value_policy::move,
        "Creates an ImageBuffer from a NumPy array. format='auto' infers: 2D or 1-channel -> GRAY8, 3-channel -> BGR888, 4-channel -> BGRA8888. copy=False uses zero-copy when the input is uint8, C-contiguous, and has even width/height.")
        .def_static("from_numpy_zero_copy", [](py::array arr, const std::string& fmt) {
            return image_buffer_from_numpy_impl(arr, fmt, false, true);
        }, "array"_a, "format"_a = "auto", py::return_value_policy::move,
        "Strict zero-copy import from a NumPy array. Requires dtype=uint8, C-contiguous layout, and even width/height; otherwise raises instead of silently copying.")
        .def_property_readonly("data", [](const ImageBuffer& img) {
             std::string raw;
             {
                 py::gil_scoped_release release;
                 raw = get_image_buffer_compact_bytes(img);
             }
             return py::bytes(raw);
         },
             "Raw bytes of the image (compact, no stride padding). Format depends on pixel format.")
        .def_readonly("width", &ImageBuffer::width)
        .def_readonly("height", &ImageBuffer::height)
        .def_property_readonly("format", [](const ImageBuffer& buf){
#if VISIONG_WITH_IVE
            if (buf.format == static_cast<PIXEL_FORMAT_E>(IVE_IMAGE_TYPE_S16C1)) {
                return std::string("S16C1");
            }
#endif
            return std::string(visiong::pixel_format_name(buf.format));
        })
        .def("is_valid", &ImageBuffer::is_valid,
             "Returns True if the buffer holds valid image data.")
        .def("copy", &ImageBuffer::copy, 
         py::call_guard<py::gil_scoped_release>(),
         py::return_value_policy::move, 
         "Creates and returns a deep copy of the image buffer.")

		.def_static("load", &ImageBuffer::load, "filepath"_a, 
                    py::call_guard<py::gil_scoped_release>(),
                    py::return_value_policy::move,
                    "Loads an image from a file into a new ImageBuffer. Supports JPEG, PNG, and BMP (via stb_image).")
		.def("save", &ImageBuffer::save, "filepath"_a, "quality"_a = 75,
             py::call_guard<py::gil_scoped_release>(),
	             "Saves the ImageBuffer to a file. Supports JPEG, PNG, and BMP. Quality (1-100) applies to JPEG/PNG. Uses software encoding (stb_image).")

	    .def("save_hsv_bin", &ImageBuffer::save_hsv_bin, "filepath"_a,
             py::call_guard<py::gil_scoped_release>(),
	         "Converts YUV420SP image to HSV using IVE hardware and saves raw HSV binary data to the given filepath. Input must be YUV420SP or YUV420SP_VU.")
	         
	    .def("save_venc_jpg", &ImageBuffer::save_venc_jpg, 
	         "filepath"_a, "quality"_a = 75,
             py::call_guard<py::gil_scoped_release>(),
	         "Saves image as JPEG using VENC hardware. If VENC is in use (e.g. by DisplayUDP), "
	         "it checks if size and format match; otherwise, it auto-initializes the encoder.")
	    .def("save_venc_h264", &ImageBuffer::save_venc_h264,
	         "filepath"_a, "quality"_a = 75, "rc_mode"_a = "cbr", "fps"_a = 30, "append"_a = true,
	         "container"_a = "auto", "mp4_faststart"_a = true,
             py::call_guard<py::gil_scoped_release>(),
	         "Saves H264 video using VENC hardware. rc_mode: 'cbr' or 'vbr'. "
	         "container: 'auto'|'annexb'|'mp4'. If container='mp4' (or filepath endswith .mp4), "
	         "it muxes into MP4 and caches the writer until close_venc_recorder/close_all_venc_recorders or process exit.")
		.def("save_venc_h265", &ImageBuffer::save_venc_h265,
	         "filepath"_a, "quality"_a = 75, "rc_mode"_a = "cbr", "fps"_a = 30, "append"_a = true,
	         "container"_a = "auto", "mp4_faststart"_a = true,
             py::call_guard<py::gil_scoped_release>(),
	         "Saves H265 video using VENC hardware. rc_mode: 'cbr' or 'vbr'. "
	         "container: 'auto'|'annexb'|'mp4'. If container='mp4' (or filepath endswith .mp4), "
	         "it muxes into MP4 and caches the writer until close_venc_recorder/close_all_venc_recorders or process exit.")
		.def("to_format", [](const ImageBuffer& self, const std::string& fmt) {
	            return self.to_format(visiong::parse_pixel_format(fmt));
	        }, "new_format"_a, py::call_guard<py::gil_scoped_release>(), py::return_value_policy::move,
	        "Converts the image to the given pixel format string (e.g. 'rgb888', 'bgr888', 'gray8', 'yuv420sp').")

	        .def("to_grayscale", [](const ImageBuffer& self) {
	            return self.to_grayscale();
	        }, py::call_guard<py::gil_scoped_release>(), py::return_value_policy::move,
	        "Converts the image to GRAY8. Uses RGA for color images.")
	        
	        .def("resize", [](const ImageBuffer& self, int w, int h) {
	            return self.resize(w, h);
	        }, "new_width"_a, "new_height"_a, py::call_guard<py::gil_scoped_release>(), py::return_value_policy::move,
	        "Resizes the image to new_width x new_height using RGA.")

	        .def("crop", [](const ImageBuffer& self, const std::tuple<int, int, int, int>& rect) {
	            return self.crop(rect);
	        }, "rect_tuple"_a, py::call_guard<py::gil_scoped_release>(), py::return_value_policy::move,
	        "Crops to the region (x, y, w, h). rect_tuple: (x, y, w, h). Uses RGA.")
	        
	        .def("crop", [](const ImageBuffer& self, int x, int y, int w, int h) {
	            return self.crop(std::make_tuple(x, y, w, h));
	        }, "x"_a, "y"_a, "w"_a, "h"_a, py::call_guard<py::gil_scoped_release>(), py::return_value_policy::move,
	        "Crops to the region (x, y, w, h). Uses RGA.")

	        .def("rotate", &ImageBuffer::rotate, "angle_degrees"_a, py::call_guard<py::gil_scoped_release>(), py::return_value_policy::move, "Rotates the image by 90, 180, or 270 degrees using hardware acceleration.")
	        .def("flip", &ImageBuffer::flip, "horizontal"_a, "vertical"_a, py::call_guard<py::gil_scoped_release>(), py::return_value_policy::move, "Flips the image horizontally and/or vertically using hardware acceleration.")

        .def("find_blobs", [](const ImageBuffer& self, const std::vector<std::tuple<int,int,int,int,int,int>>& thresholds,
            bool invert, const std::tuple<int,int,int,int>& roi, int x_stride, int y_stride,
            int area_threshold, int pixels_threshold, bool merge, int margin, int mode, int erode_size, int dilate_size) {
            return self.find_blobs(thresholds, invert, roi, x_stride, y_stride, area_threshold, pixels_threshold, merge, margin, mode, erode_size, dilate_size);
        },
            "thresholds"_a,
            "invert"_a = false,
            "roi"_a = std::make_tuple(0,0,0,0),
            "x_stride"_a = 2,
            "y_stride"_a = 2,
            "area_threshold"_a = 10,
            "pixels_threshold"_a = 10,
            "merge"_a = true,
	            "margin"_a = 10,
	            "mode"_a = 0,
	            "erode_size"_a = 0,
	            "dilate_size"_a = 0,
                py::call_guard<py::gil_scoped_release>(),
	            R"(Finds blobs by color thresholds. thresholds: list of 6-tuples (H_min,H_max,S_min,S_max,V_min,V_max) for HSV; mode 0=HSV, 1=LAB. For grayscale use the overload with [(gray_min, gray_max)].)")
        .def("find_blobs", [](const ImageBuffer& self, const std::vector<std::tuple<int,int>>& gray_thresholds,
            bool invert, const std::tuple<int,int,int,int>& roi, int x_stride, int y_stride,
            int area_threshold, int pixels_threshold, bool merge, int margin, int mode, int erode_size, int dilate_size) {
            return self.find_blobs(gray_thresholds, invert, roi, x_stride, y_stride, area_threshold, pixels_threshold, merge, margin, mode, erode_size, dilate_size);
        },
            "thresholds"_a,
            "invert"_a = false,
            "roi"_a = std::make_tuple(0,0,0,0),
            "x_stride"_a = 2,
            "y_stride"_a = 2,
            "area_threshold"_a = 10,
            "pixels_threshold"_a = 10,
            "merge"_a = true,
	            "margin"_a = 10,
	            "mode"_a = 0,
	            "erode_size"_a = 0,
	            "dilate_size"_a = 0,
                py::call_guard<py::gil_scoped_release>(),
	            R"(Grayscale only: find_blobs([(gray_min, gray_max)]). Same args as color; mode ignored. Image must be grayscale.)")
	        
	        .def("find_lines", [](const ImageBuffer& self, const std::tuple<int, int, int, int>& roi, int xs, int ys, int t, double r, double th, int cl, int ch) {
	            return self.find_lines(roi, xs, ys, t, r, th, cl, ch);
	        }, "roi"_a = std::make_tuple(0,0,0,0), "x_stride"_a=1, "y_stride"_a=1, "threshold"_a=60, "rho_resolution_px"_a=1.0, "theta_resolution_deg"_a=1.5, "canny_low_thresh"_a=50, "canny_high_thresh"_a=150, py::call_guard<py::gil_scoped_release>())
	        
	        .def("find_lines", [](const ImageBuffer& self, int x, int y, int w, int h, int x_stride, int y_stride, int threshold, double rho, double theta, int canny_l, int canny_h) {
	            return self.find_lines(std::make_tuple(x,y,w,h), x_stride, y_stride, threshold, rho, theta, canny_l, canny_h);
	        }, "x"_a, "y"_a, "w"_a, "h"_a, "x_stride"_a=1, "y_stride"_a=1, "threshold"_a=60, "rho_resolution_px"_a=1.0, "theta_resolution_deg"_a=1.5, "canny_low_thresh"_a=50, "canny_high_thresh"_a=150, py::call_guard<py::gil_scoped_release>())
	        
	        .def("find_circles", [](const ImageBuffer& self, const std::tuple<int,int,int,int>& roi, int xs, int ys, int t, int rmin, int rmax, int rstep, int cl, int ch){
	            return self.find_circles(roi, xs, ys, t, rmin, rmax, rstep, cl, ch);
	        }, "roi"_a = std::make_tuple(0,0,0,0), "x_stride"_a=1, "y_stride"_a=1, "threshold"_a=35, "r_min"_a=10, "r_max"_a=0, "r_step"_a=2, "canny_low_thresh"_a=50, "canny_high_thresh"_a=100, py::call_guard<py::gil_scoped_release>())
	        
	        .def("find_circles", [](const ImageBuffer& self, int x, int y, int w, int h, int x_s, int y_s, int th, int r_min, int r_max, int r_step, int can_l, int can_h) {
	            return self.find_circles(std::make_tuple(x,y,w,h), x_s, y_s, th, r_min, r_max, r_step, can_l, can_h);
	        }, "x"_a, "y"_a, "w"_a, "h"_a, "x_stride"_a=1, "y_stride"_a=1, "threshold"_a=35, "r_min"_a=10, "r_max"_a=0, "r_step"_a=2, "canny_low_thresh"_a=50, "canny_high_thresh"_a=100, py::call_guard<py::gil_scoped_release>())

        .def("find_polygons", [](const ImageBuffer& self, const std::tuple<int,int,int,int>& roi, int mina, int maxa, int mins, int maxs, const std::string& accuracy){
            std::string mode = visiong::to_lower_copy(accuracy);
            int acc = 1; // normal
            if (mode == "fast") acc = 0;
            else if (mode == "accurate") acc = 2;
            else if (mode != "normal") throw std::invalid_argument("accuracy must be 'fast', 'normal', or 'accurate'.");
            return self.find_polygons(roi, mina, maxa, mins, maxs, acc);
	        }, "roi"_a = std::make_tuple(0,0,0,0), "min_area"_a = 100, "max_area"_a = 100000, "min_sides"_a = 3, "max_sides"_a = 10, "accuracy"_a = "normal", py::call_guard<py::gil_scoped_release>())
        
        .def("find_polygons", [](const ImageBuffer& self, int x, int y, int w, int h, int min_a, int max_a, int min_s, int max_s, const std::string& accuracy){
            std::string mode = visiong::to_lower_copy(accuracy);
            int acc = 1; // normal
            if (mode == "fast") acc = 0;
            else if (mode == "accurate") acc = 2;
            else if (mode != "normal") throw std::invalid_argument("accuracy must be 'fast', 'normal', or 'accurate'.");
            return self.find_polygons(std::make_tuple(x,y,w,h), min_a, max_a, min_s, max_s, acc);
	        }, "x"_a, "y"_a, "w"_a, "h"_a, "min_area"_a = 100, "max_area"_a = 100000, "min_sides"_a = 3, "max_sides"_a = 10, "accuracy"_a = "normal", py::call_guard<py::gil_scoped_release>())

	        .def("find_qrcodes", &ImageBuffer::find_qrcodes, py::call_guard<py::gil_scoped_release>(),
	             "Finds and decodes QR codes in the image. The image is automatically converted to grayscale if needed.")

		.def("find_squares", &ImageBuffer::find_squares,
             "roi"_a = std::make_tuple(0,0,0,0),
             "threshold_val"_a,
             "min_area"_a = 500,
             "approx_epsilon"_a = 0.02,
             "corner_sample_radius"_a = 10,
             "corner_ratio_thresh"_a = 2.0,
             "edge_check_offset"_a = 5,
             "area_sample_points"_a = 50,
             "area_white_thresh"_a = 0.90,
	             "area_morph_close_kernel_size"_a = 0,
	             "duplicate_center_thresh"_a = 20.0,
	             "duplicate_area_thresh"_a = 0.2,
                 py::call_guard<py::gil_scoped_release>(),
	             R"(Finds squares in the image using a robust corner-based algorithm.

             This method is more computationally intensive than find_polygons but offers
             higher accuracy for square-like objects in noisy or complex backgrounds.

             Args:
                 roi (tuple): Region of interest (x, y, w, h) to search in.
                 threshold_val (int): Grayscale value for binary thresholding. Pixels darker than this become the target.
                 min_area (int): Minimum contour area to consider.
                 approx_epsilon (float): Precision factor for polygon approximation.
                 corner_sample_radius (int): Radius for sampling around potential corners to validate them.
                 corner_ratio_thresh (float): Minimum ratio of background to foreground pixels for a valid corner.
                 edge_check_offset (int): Offset distance to check if an edge is on the object's exterior.
                 area_sample_points (int): Number of random points to sample inside a candidate square.
                 area_white_thresh (float): Minimum ratio of foreground pixels required inside a valid square.
                 area_morph_close_kernel_size (int): Kernel size for morphological close before area sampling; 0 to disable.
                 duplicate_center_thresh (float): Maximum distance between centers to consider squares as duplicates.
                 duplicate_area_thresh (float): Maximum relative area difference to consider squares as duplicates.

             Returns:
                 list[list[tuple[int, int]]]: A list of found squares, where each square is a list of four (x, y) vertex tuples.
             )"
        )
	.def("binarize", &ImageBuffer::binarize,
         "method"_a = "otsu",
         "threshold_range"_a = std::make_tuple(128, 255),
         "invert"_a = false,
	         "adaptive_block_size"_a = 11,
	         "adaptive_c"_a = 2,
	         "pre_blur_kernel_size"_a = 0,
	         "post_morph_kernel_size"_a = 0,
             py::call_guard<py::gil_scoped_release>(),
	         py::return_value_policy::move,
	         R"(Performs image binarization with adjustable denoising strength.

The input image is automatically converted to grayscale.

Args:
    method (str): The binarization method. "manual", "otsu", "adaptive_mean", "adaptive_gaussian".
    threshold_range (tuple[int, int]): Range for "manual" mode.
    invert (bool): If True, inverts the black and white pixels.
    adaptive_block_size (int): Block size for adaptive methods (odd, >1).
    adaptive_c (int): Constant C for adaptive methods.
    pre_blur_kernel_size (int): Kernel size for pre-binarization Gaussian blur.
        Set to 0 to disable. If enabled, must be an odd integer >= 3 (e.g., 3, 5, 7).
        A larger kernel results in stronger blurring. Defaults to 0 (off).
    post_morph_kernel_size (int): Kernel size for post-binarization morphological opening.
        Set to 0 to disable. If enabled, must be an integer >= 2 (e.g., 3, 5, 7).
        A larger kernel will remove larger noise elements. Defaults to 3 (light denoising).

This function preserves the standard OpenCV binarization behavior for all methods.
If you want hardware acceleration, use IVE directly (filter/threshold/erode/dilate) in your own pipeline.

Returns:
    ImageBuffer: A new binarized image buffer in GRAY8 format.
)")

	        .def("warp_perspective", &ImageBuffer::warp_perspective,
	            "quad"_a, "out_width"_a, "out_height"_a,
                py::call_guard<py::gil_scoped_release>(),
	            py::return_value_policy::move,
            R"(Performs a perspective warp transformation.

            Transforms a quadrilateral region from the source image into a
            rectangular output image of a specified size.

            Args:
                quad (list[tuple[int, int]]): A list of four (x, y) tuples
                    representing the corners of the quadrilateral in the source image.
                    The order should be top-left, top-right, bottom-right, bottom-left.
                out_width (int): The width of the destination rectangular image.
                out_height (int): The height of the destination rectangular image.

            Returns:
                ImageBuffer: A new image buffer containing the warped result.
            )")

	        .def("letterbox", &ImageBuffer::letterbox, "target_width"_a, "target_height"_a, "color"_a = std::make_tuple(128,128,128), py::call_guard<py::gil_scoped_release>(), py::return_value_policy::move,
	             "Scales the image to fit inside target dimensions while preserving aspect ratio, then pads with color to fill target_width x target_height. Uses RGA.")

        .def("draw_line", &ImageBuffer::draw_line, "x0"_a, "y0"_a, "x1"_a, "y1"_a, "color"_a = std::make_tuple(255, 255, 255), "thickness"_a = 1, 
            py::return_value_policy::reference_internal, "Draws a line on the image in-place and returns itself.")

        .def("draw_rectangle", static_cast<ImageBuffer& (ImageBuffer::*)(int, int, int, int, std::tuple<unsigned char, unsigned char, unsigned char>, int, bool)>(&ImageBuffer::draw_rectangle), "x"_a, "y"_a, "w"_a, "h"_a, "color"_a = std::make_tuple(255, 255, 255), "thickness"_a = 1, "fill"_a = false, 
            py::return_value_policy::reference_internal, "Draws a rectangle on the image in-place and returns itself.")

        .def("draw_rectangle", static_cast<ImageBuffer& (ImageBuffer::*)(const std::tuple<int,int,int,int>&, std::tuple<unsigned char, unsigned char, unsigned char>, int, bool)>(&ImageBuffer::draw_rectangle), "rect_tuple"_a, "color"_a = std::make_tuple(255, 255, 255), "thickness"_a = 1, "fill"_a = false,
            py::return_value_policy::reference_internal, "Draws a rectangle on the image in-place and returns itself.")

        .def("draw_circle", &ImageBuffer::draw_circle, "cx"_a, "cy"_a, "radius"_a, "color"_a = std::make_tuple(255, 255, 255), "thickness"_a = 1, "fill"_a = false, 
            py::return_value_policy::reference_internal, "Draws a circle on the image in-place and returns itself.")

        .def("draw_string", &ImageBuffer::draw_string, "x"_a, "y"_a, "text"_a, "color"_a = std::make_tuple(255, 255, 255), "scale"_a = 1.0, "thickness"_a = 1,
            py::return_value_policy::reference_internal, "Draws text at (x,y). color (R,G,B). scale and thickness affect size. In-place, returns self.")
        .def_static("set_text_font", &ImageBuffer::set_text_font,
            "font_path"_a = "", "predefine_chars"_a = "", "glyph_budget"_a = 6623,
            "Configures the shared UTF-8 font used by draw_string (e.g. for Chinese). "
            "When predefine_chars is empty, glyph_budget limits baked glyph count.")
        .def_static("clear_text_font", &ImageBuffer::clear_text_font,
            "Clears draw_string shared font configuration.")

        .def("draw_cross", &ImageBuffer::draw_cross, "cx"_a, "cy"_a, "color"_a = std::make_tuple(255, 255, 255), "size"_a = 5, "thickness"_a = 1,
            py::return_value_policy::reference_internal, "Draws a cross on the image in-place and returns itself.")
        .def("paste", &ImageBuffer::paste, "img_to_paste"_a, "x"_a, "y"_a,
            py::return_value_policy::reference_internal,
         "Pastes another image onto this one at the specified (x, y) coordinates.")
        .def("blend", &ImageBuffer::blend, "img_to_blend"_a, "x"_a = 0, "y"_a = 0,
                 py::return_value_policy::reference_internal,
                 "Blends an RGBA image onto this image using its alpha channel. This is a CPU operation.")
        .def("__repr__", [](const ImageBuffer &buf) { return "<ImageBuffer " + std::to_string(buf.width) + "x" + std::to_string(buf.height) + ", format " + (py::cast(buf).attr("format").cast<std::string>()) + (buf.is_zero_copy() ? " zero-copy>" : ">"); });
}

