// SPDX-License-Identifier: LGPL-3.0-or-later
#include "visiong/core/Camera.h"
#include "visiong/core/BufferStateMachine.h"
#include "visiong/modules/ISPController.h"
#include "visiong/core/RgaHelper.h"
#include "im2d.hpp"
#include "rk_mpi_mb.h"
#include "rk_mpi_sys.h"
#include "visiong/common/pixel_format.h"
#include "common/internal/string_utils.h"
#include "core/internal/log_filter.h"
#include "core/internal/logger.h"
#include "core/internal/runtime_init.h"

#include <array>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <numeric>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <mutex>
#include <stdio.h>
#include <unistd.h>
#include <vector>

// V4L2 and AIQ headers / V4L2 与 AIQ 相关头文件。
#include "rk_aiq_algo_des.h"
#include "rk_aiq_user_api2_sysctl.h"
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

namespace {

ISPController& require_isp_controller(const std::unique_ptr<ISPController>& controller,
                                      const char* method_name) {
    if (!controller) {
        throw std::runtime_error(std::string(method_name) +
                                 ": camera is not initialized or ISP controller is unavailable.");
    }
    return *controller;
}

struct V4L2FdState {
    std::atomic<int> fd{-1};
    std::atomic<uint64_t> generation{0};
};

std::string normalize_camera_format_token(const std::string& format) {
    const std::string key = visiong::to_lower_copy(format);
    if (key == "nv12" || key == "yuv420sp" || key == "yuv420") {
        return "yuv";
    }
    if (key == "rgb888") {
        return "rgb";
    }
    if (key == "bgr888") {
        return "bgr";
    }
    if (key == "gray8") {
        return "gray";
    }
    return format;
}

struct CameraCropMode {
    enum class Strategy {
        Auto,
        FixedAspect,
        Off,
    };

    Strategy strategy = Strategy::Auto;
    double aspect_ratio = 0.0;
    std::string label = "auto";
};

struct CaptureGeometry {
    int width;
    int height;
};

struct CameraResolutionCaps {
    int max_width = 0;
    int max_height = 0;
    std::string ratio_label = "unknown";
    std::string source = "unknown";

    bool is_valid() const {
        return max_width > 0 && max_height > 0;
    }
};

struct CameraGeometryPlan {
    int target_width = 0;
    int target_height = 0;
    int capture_width = 0;
    int capture_height = 0;
};

int align_up(int value, int alignment) {
    if (alignment <= 0) {
        return value;
    }
    return ((value + alignment - 1) / alignment) * alignment;
}

int align_even(int value) {
    return (value + 1) & ~1;
}

std::string trim_copy(const std::string& value) {
    const auto begin = std::find_if_not(value.begin(), value.end(),
                                        [](unsigned char c) { return std::isspace(c) != 0; });
    const auto end = std::find_if_not(value.rbegin(), value.rend(),
                                      [](unsigned char c) { return std::isspace(c) != 0; })
                         .base();
    if (begin >= end) {
        return std::string();
    }
    return std::string(begin, end);
}

std::string compact_token(const std::string& value) {
    std::string out;
    out.reserve(value.size());
    for (unsigned char ch : value) {
        if (!std::isspace(ch)) {
            out.push_back(static_cast<char>(ch));
        }
    }
    return out;
}

bool parse_positive_double(const std::string& token, double* value_out) {
    if (!value_out || token.empty()) {
        return false;
    }

    char* end_ptr = nullptr;
    errno = 0;
    const double parsed = std::strtod(token.c_str(), &end_ptr);
    if (errno != 0 || end_ptr == token.c_str() || *end_ptr != '\0' || parsed <= 0.0) {
        return false;
    }

    *value_out = parsed;
    return true;
}

bool is_decimal_integer_token(const std::string& token) {
    return !token.empty() &&
           std::all_of(token.begin(), token.end(),
                       [](unsigned char c) { return std::isdigit(c) != 0; });
}

std::string make_ratio_label_from_dimensions(int width, int height) {
    if (width <= 0 || height <= 0) {
        return "unknown";
    }
    const int divisor = std::gcd(width, height);
    std::ostringstream oss;
    oss << (width / divisor) << ':' << (height / divisor);
    return oss.str();
}

bool parse_ratio_expression(const std::string& token,
                            double* aspect_ratio_out,
                            std::string* label_out) {
    if (!aspect_ratio_out || !label_out) {
        return false;
    }

    const std::string compact = compact_token(token);
    if (compact.empty()) {
        return false;
    }

    const std::size_t separator_pos = compact.find_first_of(":xX/");
    if (separator_pos == std::string::npos || separator_pos == 0 || separator_pos + 1 >= compact.size()) {
        return false;
    }

    const std::string lhs_token = compact.substr(0, separator_pos);
    const std::string rhs_token = compact.substr(separator_pos + 1);
    double lhs = 0.0;
    double rhs = 0.0;
    if (!parse_positive_double(lhs_token, &lhs) || !parse_positive_double(rhs_token, &rhs)) {
        return false;
    }

    *aspect_ratio_out = lhs / rhs;
    if (is_decimal_integer_token(lhs_token) && is_decimal_integer_token(rhs_token)) {
        *label_out = make_ratio_label_from_dimensions(std::stoi(lhs_token), std::stoi(rhs_token));
    } else {
        *label_out = lhs_token + ":" + rhs_token;
    }
    return true;
}

CameraCropMode parse_camera_crop_mode(const std::string& crop_mode) {
    const std::string token = trim_copy(crop_mode);
    const std::string key = visiong::to_lower_copy(token);

    if (token.empty() || key == "auto" || key == "default" || key == "native" || key == "sensor" ||
        key == "sensor_max" || key == "max" || key == "on" || key == "true" || key == "enabled") {
        return {CameraCropMode::Strategy::Auto, 0.0, "auto"};
    }
    if (key == "off" || key == "none" || key == "false" || key == "disabled" || key == "0") {
        return {CameraCropMode::Strategy::Off, 0.0, "off"};
    }

    double aspect_ratio = 0.0;
    std::string ratio_label;
    if (parse_ratio_expression(token, &aspect_ratio, &ratio_label)) {
        return {CameraCropMode::Strategy::FixedAspect, aspect_ratio, ratio_label};
    }

    throw std::invalid_argument("Invalid crop mode '" + crop_mode +
                                "'. Use 'auto', 'off', or a ratio such as '16:9', '4:3', '1:1', or '3:2'.");
}

CameraCropMode resolve_camera_crop_mode(const CameraCropMode& requested_mode,
                                        const CameraResolutionCaps& sensor_caps) {
    if (requested_mode.strategy == CameraCropMode::Strategy::Off) {
        return requested_mode;
    }
    if (requested_mode.strategy == CameraCropMode::Strategy::FixedAspect && requested_mode.aspect_ratio > 0.0) {
        return requested_mode;
    }
    if (sensor_caps.is_valid()) {
        return {CameraCropMode::Strategy::FixedAspect,
                static_cast<double>(sensor_caps.max_width) / static_cast<double>(sensor_caps.max_height),
                std::string("auto(") + sensor_caps.ratio_label + ')'};
    }
    return {CameraCropMode::Strategy::FixedAspect, 16.0 / 9.0, "auto(16:9)"};
}

CaptureGeometry compute_capture_geometry(int target_width,
                                         int target_height,
                                         const CameraCropMode& crop_mode) {
    if (crop_mode.strategy == CameraCropMode::Strategy::Off || crop_mode.aspect_ratio <= 0.0) {
        return {align_up(target_width, 16), align_even(target_height)};
    }

    const double aspect_ratio = crop_mode.aspect_ratio;
    int capture_width = align_up(target_width, 16);
    int capture_height = align_even(
        static_cast<int>(std::ceil(static_cast<double>(capture_width) / aspect_ratio)));

    if (capture_height < target_height) {
        capture_height = align_even(target_height);
        capture_width = align_up(
            static_cast<int>(std::ceil(static_cast<double>(capture_height) * aspect_ratio)), 16);
    }

    return {capture_width, capture_height};
}

bool should_center_crop(const CameraCropMode& crop_mode,
                        int capture_width,
                        int capture_height,
                        int target_width,
                        int target_height) {
    return crop_mode.strategy != CameraCropMode::Strategy::Off && capture_width >= target_width &&
           capture_height >= target_height &&
           (capture_width != target_width || capture_height != target_height);
}

unsigned int camera_device_capabilities(const v4l2_capability& caps) {
    if (caps.capabilities & V4L2_CAP_DEVICE_CAPS) {
        return caps.device_caps;
    }
    return caps.capabilities;
}

v4l2_buf_type select_capture_buffer_type(unsigned int capabilities) {
    if (capabilities & V4L2_CAP_VIDEO_CAPTURE_MPLANE) {
        return V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    }
    if (capabilities & V4L2_CAP_VIDEO_CAPTURE) {
        return V4L2_BUF_TYPE_VIDEO_CAPTURE;
    }
    return static_cast<v4l2_buf_type>(0);
}

void set_v4l2_format_request(v4l2_format* fmt,
                             v4l2_buf_type buf_type,
                             __u32 width,
                             __u32 height,
                             __u32 pixel_format) {
    std::memset(fmt, 0, sizeof(*fmt));
    fmt->type = buf_type;
    if (buf_type == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE) {
        fmt->fmt.pix_mp.width = width;
        fmt->fmt.pix_mp.height = height;
        fmt->fmt.pix_mp.pixelformat = pixel_format;
        fmt->fmt.pix_mp.field = V4L2_FIELD_NONE;
    } else {
        fmt->fmt.pix.width = width;
        fmt->fmt.pix.height = height;
        fmt->fmt.pix.pixelformat = pixel_format;
        fmt->fmt.pix.field = V4L2_FIELD_NONE;
    }
}

int v4l2_format_width(const v4l2_format& fmt, v4l2_buf_type buf_type) {
    return (buf_type == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE) ? static_cast<int>(fmt.fmt.pix_mp.width)
                                                            : static_cast<int>(fmt.fmt.pix.width);
}

int v4l2_format_height(const v4l2_format& fmt, v4l2_buf_type buf_type) {
    return (buf_type == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE) ? static_cast<int>(fmt.fmt.pix_mp.height)
                                                            : static_cast<int>(fmt.fmt.pix.height);
}

void update_resolution_caps(CameraResolutionCaps* caps,
                            int width,
                            int height,
                            const std::string& source_label) {
    if (!caps || width <= 0 || height <= 0) {
        return;
    }

    const long long candidate_area = static_cast<long long>(width) * height;
    const long long current_area = static_cast<long long>(caps->max_width) * caps->max_height;
    if (!caps->is_valid() || candidate_area > current_area ||
        (candidate_area == current_area && width > caps->max_width)) {
        caps->max_width = width;
        caps->max_height = height;
        caps->ratio_label = make_ratio_label_from_dimensions(width, height);
        caps->source = source_label;
    }
}

bool enumerate_frame_sizes_for_format(int fd,
                                      __u32 pixel_format,
                                      CameraResolutionCaps* caps,
                                      const std::string& source_label) {
    bool got_any = false;
    for (__u32 index = 0;; ++index) {
        v4l2_frmsizeenum frame_size;
        std::memset(&frame_size, 0, sizeof(frame_size));
        frame_size.index = index;
        frame_size.pixel_format = pixel_format;
        if (ioctl(fd, VIDIOC_ENUM_FRAMESIZES, &frame_size) < 0) {
            break;
        }

        got_any = true;
        if (frame_size.type == V4L2_FRMSIZE_TYPE_DISCRETE) {
            update_resolution_caps(caps,
                                   static_cast<int>(frame_size.discrete.width),
                                   static_cast<int>(frame_size.discrete.height),
                                   source_label);
            continue;
        }

        if (frame_size.type == V4L2_FRMSIZE_TYPE_STEPWISE ||
            frame_size.type == V4L2_FRMSIZE_TYPE_CONTINUOUS) {
            update_resolution_caps(caps,
                                   static_cast<int>(frame_size.stepwise.max_width),
                                   static_cast<int>(frame_size.stepwise.max_height),
                                   source_label);
            break;
        }
    }
    return got_any;
}

CameraResolutionCaps probe_camera_resolution_caps(const char* devpath) {
    CameraResolutionCaps caps;
    const int fd = ::open(devpath, O_RDWR | O_NONBLOCK, 0);
    if (fd < 0) {
        return caps;
    }

    v4l2_capability device_caps;
    std::memset(&device_caps, 0, sizeof(device_caps));
    if (ioctl(fd, VIDIOC_QUERYCAP, &device_caps) < 0) {
        ::close(fd);
        return caps;
    }

    const unsigned int capabilities = camera_device_capabilities(device_caps);
    const v4l2_buf_type buf_type = select_capture_buffer_type(capabilities);
    if (buf_type == static_cast<v4l2_buf_type>(0)) {
        ::close(fd);
        return caps;
    }

    bool enumerated = enumerate_frame_sizes_for_format(fd,
                                                       V4L2_PIX_FMT_NV12,
                                                       &caps,
                                                       "VIDIOC_ENUM_FRAMESIZES(NV12)");

    if (!enumerated || !caps.is_valid()) {
        for (__u32 index = 0;; ++index) {
            v4l2_fmtdesc format_desc;
            std::memset(&format_desc, 0, sizeof(format_desc));
            format_desc.index = index;
            format_desc.type = buf_type;
            if (ioctl(fd, VIDIOC_ENUM_FMT, &format_desc) < 0) {
                break;
            }
            enumerated |= enumerate_frame_sizes_for_format(
                fd,
                format_desc.pixelformat,
                &caps,
                std::string("VIDIOC_ENUM_FRAMESIZES(") +
                    reinterpret_cast<const char*>(format_desc.description) + ')');
        }
    }

    if (!caps.is_valid()) {
        v4l2_format current_format;
        std::memset(&current_format, 0, sizeof(current_format));
        current_format.type = buf_type;
        if (ioctl(fd, VIDIOC_G_FMT, &current_format) == 0) {
            update_resolution_caps(&caps,
                                   v4l2_format_width(current_format, buf_type),
                                   v4l2_format_height(current_format, buf_type),
                                   "VIDIOC_G_FMT");
        }
    }

    if (!caps.is_valid()) {
        v4l2_format probe_format;
        set_v4l2_format_request(&probe_format, buf_type, 8192, 8192, V4L2_PIX_FMT_NV12);
        if (ioctl(fd, VIDIOC_S_FMT, &probe_format) == 0) {
            update_resolution_caps(&caps,
                                   v4l2_format_width(probe_format, buf_type),
                                   v4l2_format_height(probe_format, buf_type),
                                   "VIDIOC_S_FMT probe");
        }
    }

    ::close(fd);
    return caps;
}

[[noreturn]] void throw_camera_geometry_limit_error(int requested_width,
                                                    int requested_height,
                                                    int aligned_width,
                                                    int aligned_height,
                                                    const CameraCropMode& crop_mode,
                                                    const CaptureGeometry& capture,
                                                    const CameraResolutionCaps& sensor_caps) {
    std::ostringstream oss;
    oss << "Requested camera size " << requested_width << 'x' << requested_height
        << " (aligned " << aligned_width << 'x' << aligned_height << ')';
    if (crop_mode.strategy == CameraCropMode::Strategy::Off) {
        oss << " exceeds the sensor limit " << sensor_caps.max_width << 'x' << sensor_caps.max_height;
    } else {
        oss << " with crop_mode " << crop_mode.label << " requires capture " << capture.width << 'x'
            << capture.height << ", but the sensor limit is " << sensor_caps.max_width << 'x'
            << sensor_caps.max_height;
    }
    oss << ". Use a smaller resolution that fits within the probed camera maximum.";
    throw std::invalid_argument(oss.str());
}

CameraGeometryPlan build_camera_geometry_plan(int user_width,
                                              int user_height,
                                              const CameraCropMode& crop_mode,
                                              const CameraResolutionCaps& sensor_caps) {
    CameraGeometryPlan plan;
    plan.target_width = std::max(16, align_up(user_width, 16));
    plan.target_height = std::max(2, align_even(user_height));

    CaptureGeometry capture = compute_capture_geometry(plan.target_width, plan.target_height, crop_mode);
    if (sensor_caps.is_valid() &&
        (plan.target_width > sensor_caps.max_width || plan.target_height > sensor_caps.max_height ||
         capture.width > sensor_caps.max_width || capture.height > sensor_caps.max_height)) {
        throw_camera_geometry_limit_error(user_width,
                                          user_height,
                                          plan.target_width,
                                          plan.target_height,
                                          crop_mode,
                                          capture,
                                          sensor_caps);
    }

    plan.capture_width = capture.width;
    plan.capture_height = capture.height;
    return plan;
}
} // namespace

namespace {
std::atomic<bool> g_sys_globally_initialized(false);
std::recursive_mutex g_sys_init_mutex;
int g_camera_instance_count = 0;
}  // namespace

bool visiong_init_sys_if_needed() {
    std::lock_guard<std::recursive_mutex> lock(g_sys_init_mutex);
    if (!g_sys_globally_initialized) {
        if (RK_MPI_SYS_Init() != RK_SUCCESS) {
            std::cerr << "VisionG Error: Global RK_MPI_SYS_Init failed" << std::endl;
            return false;
        }
        g_sys_globally_initialized = true;
    }
    return true;
}

class CaptureV4L2Impl {
  public:
    CaptureV4L2Impl();
    ~CaptureV4L2Impl();

    int open_and_prepare(int width, int height, float fps, const char* iq_file_dir,
                         rk_aiq_working_mode_t wdr_mode);
    int start_streaming();
    int read_frame_to_yuv_buffer(ImageBuffer& frame_out);
    int stop_streaming();
    int close();

    rk_aiq_sys_ctx_t* aiq_ctx;
    char devpath[32];
    int fd;
    v4l2_buf_type buf_type;

    __u32 cap_width;
    __u32 cap_height;

    std::shared_ptr<V4L2FdState> fd_state;
    uint64_t fd_generation = 0;

    struct buffer {
        void* start;
        size_t length;
    };
    std::vector<buffer> buffers;
    // Reused fallback CPU buffers for EXPBUF failure path. / 为 EXPBUF 失败路径复用的 CPU 回退缓冲区。
    std::array<std::shared_ptr<std::vector<unsigned char>>, 2> fallback_cpu_frames;
    size_t fallback_cpu_index = 0;
};

class CaptureV4L2 {
  public:
    CaptureV4L2() : d(new CaptureV4L2Impl) {}
    ~CaptureV4L2() { delete d; }
    int open_and_prepare(int width, int height, float fps, const char* iq_file_dir,
                         rk_aiq_working_mode_t wdr_mode) {
        return d->open_and_prepare(width, height, fps, iq_file_dir, wdr_mode);
    }
    int start_streaming() { return d->start_streaming(); }
    int read_frame_to_yuv_buffer(ImageBuffer& frame_out) { return d->read_frame_to_yuv_buffer(frame_out); }
    int stop_streaming() { return d->stop_streaming(); }
    int close() { return d->close(); }
    int get_width() const { return d->cap_width; }
    int get_height() const { return d->cap_height; }
    rk_aiq_sys_ctx_t* get_aiq_context() const { return d->aiq_ctx; }

  private:
    CaptureV4L2Impl* const d;
};

CaptureV4L2Impl::CaptureV4L2Impl() {
    aiq_ctx = nullptr;
    fd = -1;
    buf_type = (v4l2_buf_type)0;
    cap_width = 0;
    cap_height = 0;
    fd_state = std::make_shared<V4L2FdState>();
    fd_generation = 0;
    for (auto& frame : fallback_cpu_frames) {
        frame = std::make_shared<std::vector<unsigned char>>();
    }
    fallback_cpu_index = 0;
}

CaptureV4L2Impl::~CaptureV4L2Impl() {
    close();
}

int CaptureV4L2Impl::open_and_prepare(int width, int height, float fps, const char* iq_file_dir,
                                      rk_aiq_working_mode_t wdr_mode) {
    (void)fps;
    // Declare commonly reused locals early. / 提前声明会重复使用的局部变量。
    v4l2_capability caps;
    v4l2_format fmt;
    v4l2_requestbuffers req;
    const char* sns_entity_name;
    rk_aiq_static_info_t aiq_static_info;
    int ret_prepare;
    unsigned int device_caps = 0;

    // 1. AIQ Init / 1. 初始化 AIQ。
    rk_aiq_uapi2_sysctl_enumStaticMetas(0, &aiq_static_info);
    sns_entity_name = aiq_static_info.sensor_info.sensor_name;

    aiq_ctx = rk_aiq_uapi2_sysctl_init(sns_entity_name, iq_file_dir, nullptr, nullptr);
    if (!aiq_ctx) {
        // Keep detailed diagnostics on stderr for troubleshooting. / 在 stderr 保留详细诊断信息，便于排障。
        fprintf(stderr, "V4L2 Error: rk_aiq_uapi2_sysctl_init failed\n");
        return -1;
    }

    // 2. Open V4L2 Device / 2. 打开 V4L2 设备。
    strcpy(devpath, "/dev/video11");
    fd = ::open(devpath, O_RDWR | O_NONBLOCK, 0);
    if (fd < 0) {
        fprintf(stderr, "V4L2 Error: open %s failed %d %s\n", devpath, errno, strerror(errno));
        goto OUT_AIQ_DEINIT;
    }
    fd_generation = fd_state->generation.fetch_add(1, std::memory_order_acq_rel) + 1;
    fd_state->fd.store(fd, std::memory_order_release);

    // 3. Configure V4L2 Format and Buffers / 配置 V4L2 采集格式与缓冲区。
    memset(&caps, 0, sizeof(caps));
    if (ioctl(fd, VIDIOC_QUERYCAP, &caps)) {
        goto OUT;
    }

    device_caps = camera_device_capabilities(caps);
    buf_type = select_capture_buffer_type(device_caps);
    if (buf_type == static_cast<v4l2_buf_type>(0))
        goto OUT;
    if (!(device_caps & V4L2_CAP_STREAMING))
        goto OUT;

    set_v4l2_format_request(&fmt, buf_type, static_cast<__u32>(width), static_cast<__u32>(height), V4L2_PIX_FMT_NV12);
    if (ioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
        fprintf(stderr, "V4L2 Error: ioctl VIDIOC_S_FMT failed: %s\n", strerror(errno));
        goto OUT;
    }

    cap_width = static_cast<__u32>(v4l2_format_width(fmt, buf_type));
    cap_height = static_cast<__u32>(v4l2_format_height(fmt, buf_type));

    memset(&req, 0, sizeof(req));
    req.count = 4;
    req.type = buf_type;
    req.memory = V4L2_MEMORY_MMAP;
    if (ioctl(fd, VIDIOC_REQBUFS, &req)) {
        fprintf(stderr, "V4L2 Error: ioctl VIDIOC_REQBUFS failed: %s\n", strerror(errno));
        goto OUT;
    }

    // 4. AIQ Prepare / 4. 准备 AIQ。
    ret_prepare = rk_aiq_uapi2_sysctl_prepare(aiq_ctx, cap_width, cap_height, wdr_mode);

    // 4.1 Try to disable AF/Dehaze modules (best effort). / 4.1 尝试关闭 AF/Dehaze 模块（尽力而为）。
    if (aiq_ctx) {
        // Explicit cast to module enum IDs: AF=2, Dehaze=15. / 显式转换到模块枚举 ID：AF=2，Dehaze=15。
        rk_aiq_uapi2_sysctl_setModuleCtl(aiq_ctx, (rk_aiq_module_id_t)2, false);
        rk_aiq_uapi2_sysctl_setModuleCtl(aiq_ctx, (rk_aiq_module_id_t)15, false);
    }

    if (ret_prepare != XCAM_RETURN_NO_ERROR) {
        fprintf(stderr, "V4L2 Error: rk_aiq_uapi2_sysctl_prepare failed\n");
        goto OUT;
    }

    // 5. Map Buffers / 5. 映射缓冲区。
    buffers.resize(req.count);
    for (unsigned int i = 0; i < req.count; ++i) {
        v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = buf_type;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;

        v4l2_plane planes[2];
        if (buf_type == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE) {
            buf.m.planes = planes;
            buf.length = 2;
        }

        if (ioctl(fd, VIDIOC_QUERYBUF, &buf)) {
            fprintf(stderr, "V4L2 Error: ioctl VIDIOC_QUERYBUF failed for buffer %d\n", i);
            goto OUT;
        }

        size_t map_len =
            (buf_type == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE) ? buf.m.planes[0].length : buf.length;
        off_t map_offset =
            (buf_type == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE) ? buf.m.planes[0].m.mem_offset : buf.m.offset;
        buffers[i].length = map_len;
        buffers[i].start = mmap(NULL, map_len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, map_offset);

        if (buffers[i].start == MAP_FAILED) {
            fprintf(stderr, "V4L2 Error: mmap failed\n");
            buffers[i].start = nullptr;
            goto OUT;
        }
    }

    return 0;

OUT:
    close();
    return -1;
OUT_AIQ_DEINIT:
    if (aiq_ctx)
        rk_aiq_uapi2_sysctl_deinit(aiq_ctx);
    aiq_ctx = nullptr;
    return -1;
}

int CaptureV4L2Impl::start_streaming() {
    if (rk_aiq_uapi2_sysctl_start(aiq_ctx) != XCAM_RETURN_NO_ERROR) {
        fprintf(stderr, "V4L2 Error: rk_aiq_uapi2_sysctl_start failed\n");
        return -1;
    }

    for (unsigned int i = 0; i < buffers.size(); ++i) {
        v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = buf_type;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        v4l2_plane planes[2];
        if (buf_type == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE) {
            buf.m.planes = planes;
            buf.length = 2;
        }
        if (ioctl(fd, VIDIOC_QBUF, &buf) == -1) {
            fprintf(stderr, "V4L2 Error: ioctl VIDIOC_QBUF failed for buffer %d: %s\n", i, strerror(errno));
            return -1;
        }
    }

    if (ioctl(fd, VIDIOC_STREAMON, &buf_type) < 0) {
        fprintf(stderr, "V4L2 Error: ioctl VIDIOC_STREAMON failed: %s\n", strerror(errno));
        rk_aiq_uapi2_sysctl_stop(aiq_ctx, false);
        return -1;
    }

    return 0;
}

// Zero-copy callback path: recycle V4L2 buffer (QBUF) and close exported fd. / 零拷贝回调路径：回收 V4L2 缓冲区（QBUF）并关闭导出的 fd。
struct V4L2MbOpaque {
    int buf_index;
    int export_fd;
    int v4l2_fd;
    uint64_t fd_generation;
    std::weak_ptr<V4L2FdState> fd_state;
    __u32 v4l2_buf_type;
};
static RK_S32 v4l2_mb_free_cb(void* opaque) {
    V4L2MbOpaque* p = static_cast<V4L2MbOpaque*>(opaque);
    if (!p) {
        return 0;
    }
    v4l2_buffer b;
    memset(&b, 0, sizeof(b));
    b.type = static_cast<v4l2_buf_type>(p->v4l2_buf_type);
    b.memory = V4L2_MEMORY_MMAP;
    b.index = p->buf_index;
    v4l2_plane planes[2];
    if (b.type == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE) {
        b.m.planes = planes;
        b.length = 2;
    }

    bool can_qbuf = false;
    if (const auto state = p->fd_state.lock()) {
        const uint64_t live_generation = state->generation.load(std::memory_order_acquire);
        const int live_fd = state->fd.load(std::memory_order_acquire);
        can_qbuf = (live_generation == p->fd_generation && live_fd >= 0 && live_fd == p->v4l2_fd);
    }
    if (can_qbuf) {
        ioctl(p->v4l2_fd, VIDIOC_QBUF, &b);
    }

    if (p->export_fd >= 0)
        ::close(p->export_fd);
    delete p;
    return 0;
}

int CaptureV4L2Impl::read_frame_to_yuv_buffer(ImageBuffer& frame_out) {
    v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.type = buf_type;
    buf.memory = V4L2_MEMORY_MMAP;
    v4l2_plane planes[2];
    if (buf_type == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE) {
        buf.m.planes = planes;
        buf.length = 2;
    }

    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(fd, &fds);
    struct timeval tv = {2, 0};
    int r = select(fd + 1, &fds, NULL, NULL, &tv);
    if (r <= 0) {
        fprintf(stderr, "V4L2 Error: select timeout or error\n");
        return -1;
    }

    if (ioctl(fd, VIDIOC_DQBUF, &buf) < 0) {
        fprintf(stderr, "V4L2 Error: VIDIOC_DQBUF failed: %s\n", strerror(errno));
        return -1;
    }

    const size_t yuv_size = cap_width * cap_height * 3 / 2;

    // Zero-copy path: export DMA-BUF fd, wrap with CreateMB, and requeue in free callback. / 零拷贝路径：导出 DMA-BUF fd，封装为 CreateMB，并在释放回调中重新入队。
    struct v4l2_exportbuffer expbuf;
    memset(&expbuf, 0, sizeof(expbuf));
    expbuf.type = buf_type;
    expbuf.index = buf.index;
    expbuf.plane = 0;
    expbuf.flags = 0;
    expbuf.fd = -1;
    if (ioctl(fd, VIDIOC_EXPBUF, &expbuf) == 0 && expbuf.fd >= 0) {
        if (visiong_init_sys_if_needed()) {
            MB_EXT_CONFIG_S ext;
            memset(&ext, 0, sizeof(ext));
            ext.pu8VirAddr = static_cast<RK_U8*>(buffers[buf.index].start);
            ext.u64PhyAddr = 0;
            ext.s32Fd = expbuf.fd;
            ext.u64Size = yuv_size;
            ext.pFreeCB = v4l2_mb_free_cb;
            ext.pOpaque =
                new V4L2MbOpaque{static_cast<int>(buf.index), expbuf.fd, fd, fd_generation, fd_state, buf_type};

            MB_BLK mb_blk = MB_INVALID_HANDLE;
            if (RK_MPI_SYS_CreateMB(&mb_blk, &ext) == 0 && mb_blk != MB_INVALID_HANDLE) {
                frame_out = ImageBuffer(cap_width, cap_height, RK_FMT_YUV420SP, mb_blk);
                // Align strides to VENC-friendly boundaries. / 将步幅对齐到更适合 VENC 的边界。
                frame_out.w_stride = (cap_width + 15) & ~15;
                frame_out.h_stride = (cap_height + 1) & ~1;
                visiong::bufstate::mark_device_write(frame_out, visiong::bufstate::BufferOwner::Camera);
                return 0;
            }
            v4l2_mb_free_cb(ext.pOpaque);
        } else {
            ::close(expbuf.fd);
            if (ioctl(fd, VIDIOC_QBUF, &buf) < 0)
                fprintf(stderr, "V4L2 Error: VIDIOC_QBUF failed after EXPBUF: %s\n", strerror(errno));
        }
    }

    // Fallback: copy to CPU and then QBUF. / 回退路径：先拷贝到 CPU，再执行 QBUF。
    auto& frame_slot = fallback_cpu_frames[fallback_cpu_index];
    fallback_cpu_index = (fallback_cpu_index + 1) % fallback_cpu_frames.size();
    if (!frame_slot || frame_slot.use_count() > 1) {
        frame_slot = std::make_shared<std::vector<unsigned char>>();
    }
    frame_slot->resize(yuv_size);
    memcpy(frame_slot->data(), buffers[buf.index].start, yuv_size);
    if (ioctl(fd, VIDIOC_QBUF, &buf) < 0) {
        fprintf(stderr, "V4L2 Error: VIDIOC_QBUF failed: %s\n", strerror(errno));
        return -1;
    }
    frame_out = ImageBuffer(cap_width, cap_height, RK_FMT_YUV420SP, frame_slot->data(),
                            frame_slot->size(), std::static_pointer_cast<void>(frame_slot));
    return 0;
}

int CaptureV4L2Impl::stop_streaming() {
    if (fd >= 0) {
        ioctl(fd, VIDIOC_STREAMOFF, &buf_type);
    }
    if (aiq_ctx) {
        rk_aiq_uapi2_sysctl_stop(aiq_ctx, false);
    }
    return 0;
}

int CaptureV4L2Impl::close() {
    stop_streaming();
    if (aiq_ctx) {
        rk_aiq_uapi2_sysctl_deinit(aiq_ctx);
        aiq_ctx = nullptr;
    }
    for (auto& buf : buffers) {
        if (buf.start && buf.start != MAP_FAILED) {
            munmap(buf.start, buf.length);
        }
    }
    buffers.clear();
    if (fd >= 0) {
        fd_state->fd.store(-1, std::memory_order_release);
        fd_state->generation.fetch_add(1, std::memory_order_acq_rel);
        ::close(fd);
        fd = -1;
    }
    return 0;
}


struct CameraImpl {
    std::unique_ptr<CaptureV4L2> v4l2_capture;
    int target_width = 0;
    int target_height = 0;
    int actual_capture_width = 0;
    int actual_capture_height = 0;
    int sensor_max_width = 0;
    int sensor_max_height = 0;
    PIXEL_FORMAT_E output_format = RK_FMT_YUV420SP;
    CameraCropMode crop_mode;
    std::string requested_crop_mode = "auto";
    bool hdr_enabled = false;
    std::atomic<bool> initialized{false};
    const char* iq_files_dir = "/etc/iqfiles/";
    std::unique_ptr<ISPController> isp_controller;
    MB_POOL convert_pool = MB_INVALID_POOLID;
    MB_POOL crop_pool = MB_INVALID_POOLID;
    int crop_vir_width = 0;
    int crop_vir_height = 0;
    size_t crop_mb_size = 0;
};
Camera::Camera(int target_width,
               int target_height,
               const std::string& format,
               bool hdr_enabled,
               const std::string& crop_mode)
    : m_impl(std::make_unique<CameraImpl>()) {
    visiong::initialize_camera_log_filter();
    std::lock_guard<std::recursive_mutex> lock(g_sys_init_mutex);
    g_camera_instance_count++;
    if (!init(target_width, target_height, format, hdr_enabled, crop_mode)) {
        VISIONG_LOG_ERROR("Camera", "Initialization failed in constructor.");
    }
}

Camera::Camera()
    : m_impl(std::make_unique<CameraImpl>()) {
    visiong::initialize_camera_log_filter();
    std::lock_guard<std::recursive_mutex> lock(g_sys_init_mutex);
    g_camera_instance_count++;
}

Camera::~Camera() {
    release();
    std::lock_guard<std::recursive_mutex> lock(g_sys_init_mutex);
    g_camera_instance_count--;
    if (g_camera_instance_count == 0 && g_sys_globally_initialized) {
        RK_MPI_SYS_Exit();
        g_sys_globally_initialized = false;
        VISIONG_LOG_INFO("Camera", "Global system resources released.");
    }
}

bool Camera::init(int user_width,
                  int user_height,
                  const std::string& format,
                  bool hdr_enabled,
                  const std::string& crop_mode) {
    if (m_impl->initialized)
        return true;
    if (user_width <= 0 || user_height <= 0) {
        VISIONG_LOG_ERROR("Camera", "Invalid user dimensions provided to init().");
        return false;
    }

    m_impl->hdr_enabled = hdr_enabled;
    m_impl->requested_crop_mode = trim_copy(crop_mode.empty() ? std::string("auto") : crop_mode);
    const CameraCropMode requested_crop_mode = parse_camera_crop_mode(m_impl->requested_crop_mode);

    const CameraResolutionCaps sensor_caps = probe_camera_resolution_caps("/dev/video11");
    m_impl->sensor_max_width = sensor_caps.max_width;
    m_impl->sensor_max_height = sensor_caps.max_height;
    m_impl->crop_mode = resolve_camera_crop_mode(requested_crop_mode, sensor_caps);

    const std::string effective_format = normalize_camera_format_token(format);
    if (effective_format != format) {
        VISIONG_LOG_INFO("Camera", "Normalizing format token '" << format << "' -> '" << effective_format << "'.");
    }

    try {
        m_impl->output_format = visiong::parse_camera_pixel_format(effective_format);
    } catch (const std::exception& e) {
        VISIONG_LOG_ERROR("Camera", "Invalid format: " << e.what());
        throw;
    }

    const CameraGeometryPlan geometry_plan =
        build_camera_geometry_plan(user_width, user_height, m_impl->crop_mode, sensor_caps);

    m_impl->target_width = geometry_plan.target_width;
    m_impl->target_height = geometry_plan.target_height;
    m_impl->actual_capture_width = geometry_plan.capture_width;
    m_impl->actual_capture_height = geometry_plan.capture_height;

    if (sensor_caps.is_valid()) {
        VISIONG_LOG_INFO("Camera",
                         "Detected camera max capture size via " << sensor_caps.source << ": "
                                                                 << sensor_caps.max_width << "x"
                                                                 << sensor_caps.max_height << " ("
                                                                 << sensor_caps.ratio_label << ")");
    } else {
        VISIONG_LOG_WARN("Camera",
                         "Failed to probe the camera max capture size via V4L2. Auto crop falls back to 16:9.");
    }

    VISIONG_LOG_INFO("Camera",
                     "User request " << user_width << "x" << user_height << ", format " << format
                                     << " (effective " << effective_format << "), requested crop_mode "
                                     << m_impl->requested_crop_mode << ", effective crop_mode "
                                     << m_impl->crop_mode.label);
    VISIONG_LOG_INFO("Camera",
                     "Final target (aligned): " << m_impl->target_width << "x" << m_impl->target_height);

    if (m_impl->crop_mode.strategy == CameraCropMode::Strategy::Off) {
        VISIONG_LOG_INFO("Camera",
                         "Cropping disabled. V4L2 will request aligned target capture: "
                             << m_impl->actual_capture_width << "x" << m_impl->actual_capture_height);
    } else {
        VISIONG_LOG_INFO("Camera",
                         "V4L2 will request " << m_impl->crop_mode.label
                                               << " capture before center crop: "
                                               << m_impl->actual_capture_width << "x"
                                               << m_impl->actual_capture_height);
    }

    if (!visiong_init_sys_if_needed())
        return false;

    try {
        m_impl->v4l2_capture = std::make_unique<CaptureV4L2>();
        rk_aiq_working_mode_t wdr_mode =
            m_impl->hdr_enabled ? RK_AIQ_WORKING_MODE_ISP_HDR2 : RK_AIQ_WORKING_MODE_NORMAL;
        VISIONG_LOG_INFO("Camera", "Initializing V4L2 with WDR mode: " << (m_impl->hdr_enabled ? "HDR2" : "NORMAL"));

        if (m_impl->v4l2_capture->open_and_prepare(m_impl->actual_capture_width,
                                                   m_impl->actual_capture_height,
                                                   30.0f,
                                                   m_impl->iq_files_dir,
                                                   wdr_mode) != 0) {
            throw std::runtime_error("Failed to open and configure V4L2 capture device.");
        }
        m_impl->isp_controller = std::make_unique<ISPController>(m_impl->v4l2_capture->get_aiq_context());
        if (m_impl->v4l2_capture->start_streaming() != 0) {
            throw std::runtime_error("Failed to start V4L2 streaming.");
        }
    } catch (const std::exception& e) {
        VISIONG_LOG_ERROR("Camera", "Error during V4L2 initialization: " << e.what());
        release();
        return false;
    }
    m_impl->initialized = true;
    m_impl->actual_capture_width = m_impl->v4l2_capture->get_width();
    m_impl->actual_capture_height = m_impl->v4l2_capture->get_height();

    // Pre-allocate crop buffers only when the selected mode really needs center crop.
    // 仅在当前模式确实需要居中裁切时预分配裁切缓冲。
    if (should_center_crop(m_impl->crop_mode,
                           m_impl->actual_capture_width,
                           m_impl->actual_capture_height,
                           m_impl->target_width,
                           m_impl->target_height)) {
        m_impl->crop_vir_width = (m_impl->target_width + 15) & ~15;
        m_impl->crop_vir_height = (m_impl->target_height + 1) & ~1;
        m_impl->crop_mb_size = static_cast<size_t>(m_impl->crop_vir_width) * m_impl->crop_vir_height * 3 / 2;

        MB_POOL_CONFIG_S crop_pool_cfg;
        memset(&crop_pool_cfg, 0, sizeof(crop_pool_cfg));
        crop_pool_cfg.u64MBSize = static_cast<RK_U64>(m_impl->crop_mb_size);
        crop_pool_cfg.u32MBCnt = 3;
        crop_pool_cfg.enAllocType = MB_ALLOC_TYPE_DMA;
        crop_pool_cfg.bPreAlloc = RK_TRUE;
        m_impl->crop_pool = RK_MPI_MB_CreatePool(&crop_pool_cfg);
        if (m_impl->crop_pool == MB_INVALID_POOLID) {
            VISIONG_LOG_WARN("Camera",
                             "Failed to create crop pool; crop path will use fallback allocations.");
            m_impl->crop_vir_width = 0;
            m_impl->crop_vir_height = 0;
            m_impl->crop_mb_size = 0;
        } else {
            VISIONG_LOG_INFO("Camera",
                             "Crop pool ready: " << m_impl->crop_vir_width << "x" << m_impl->crop_vir_height
                                                 << " mb_size=" << m_impl->crop_mb_size);
        }
    } else {
        m_impl->crop_vir_width = 0;
        m_impl->crop_vir_height = 0;
        m_impl->crop_mb_size = 0;
    }

    // Zero-copy YUV->RGB/BGR: RGA writes directly into pooled DMA buffers. / 零拷贝 YUV->RGB/BGR：由 RGA 直接写入池化 DMA 缓冲区。
    if (m_impl->output_format == RK_FMT_RGB888 || m_impl->output_format == RK_FMT_BGR888) {
        MB_POOL_CONFIG_S pool_cfg;
        memset(&pool_cfg, 0, sizeof(pool_cfg));
        const int convert_pool_width = std::max(m_impl->actual_capture_width, m_impl->target_width);
        const int convert_pool_height = std::max(m_impl->actual_capture_height, m_impl->target_height);
        pool_cfg.u64MBSize = static_cast<RK_U64>(convert_pool_width) * convert_pool_height * 3;
        pool_cfg.u32MBCnt = 2;
        pool_cfg.enAllocType = MB_ALLOC_TYPE_DMA;
        pool_cfg.bPreAlloc = RK_TRUE;
        m_impl->convert_pool = RK_MPI_MB_CreatePool(&pool_cfg);
        if (m_impl->convert_pool == MB_INVALID_POOLID) {
            VISIONG_LOG_WARN("Camera",
                             "Failed to create convert pool for zero-copy RGB; will fall back to copy path.");
        }
    }

    VISIONG_LOG_INFO("Camera",
                     "Initialized via V4L2. Actual capture size: " << m_impl->actual_capture_width << "x"
                                                                     << m_impl->actual_capture_height);
    return true;
}

ImageBuffer Camera::snapshot() {
    if (!m_impl->initialized || !m_impl->v4l2_capture) {
        return ImageBuffer();
    }

    ImageBuffer raw_yuv_frame;
    if (m_impl->v4l2_capture->read_frame_to_yuv_buffer(raw_yuv_frame) != 0) {
        VISIONG_LOG_WARN("Camera", "Failed to read frame from V4L2 device.");
        return ImageBuffer();
    }

    ImageBuffer cropped_yuv_frame;
    const ImageBuffer* processing_buf = &raw_yuv_frame;

    if (m_impl->target_width != m_impl->actual_capture_width || m_impl->target_height != m_impl->actual_capture_height) {
        const bool can_center_crop = should_center_crop(m_impl->crop_mode,
                                                        m_impl->actual_capture_width,
                                                        m_impl->actual_capture_height,
                                                        m_impl->target_width,
                                                        m_impl->target_height);
        int crop_x = (m_impl->actual_capture_width - m_impl->target_width) / 2;
        int crop_y = (m_impl->actual_capture_height - m_impl->target_height) / 2;
        crop_x = (crop_x / 2) * 2;
        crop_y = (crop_y / 2) * 2;

        bool geometry_fixed = false;
        if (can_center_crop && raw_yuv_frame.is_zero_copy() && raw_yuv_frame.get_dma_fd() >= 0 &&
            m_impl->crop_pool != MB_INVALID_POOLID && m_impl->crop_vir_width > 0 && m_impl->crop_vir_height > 0 &&
            m_impl->crop_mb_size > 0) {
            MB_BLK crop_mb = RK_MPI_MB_GetMB(m_impl->crop_pool, static_cast<RK_U64>(m_impl->crop_mb_size), RK_TRUE);
            if (crop_mb != MB_INVALID_HANDLE) {
                void* crop_vir_addr = RK_MPI_MB_Handle2VirAddr(crop_mb);
                const int crop_fd = RK_MPI_MB_Handle2Fd(crop_mb);
                const size_t crop_mb_size = RK_MPI_MB_GetSize(crop_mb);
                if (crop_vir_addr && crop_fd >= 0 && crop_mb_size >= m_impl->crop_mb_size) {
                    try {
                        RgaDmaBuffer src_dma(raw_yuv_frame.get_dma_fd(),
                                             const_cast<void*>(raw_yuv_frame.get_data()),
                                             raw_yuv_frame.get_size(),
                                             raw_yuv_frame.width,
                                             raw_yuv_frame.height,
                                             static_cast<int>(raw_yuv_frame.format),
                                             raw_yuv_frame.w_stride,
                                             raw_yuv_frame.h_stride);
                        RgaDmaBuffer dst_dma(crop_fd,
                                             crop_vir_addr,
                                             crop_mb_size,
                                             m_impl->target_width,
                                             m_impl->target_height,
                                             static_cast<int>(raw_yuv_frame.format),
                                             m_impl->crop_vir_width,
                                             m_impl->crop_vir_height);
                        im_rect crop_rect = {crop_x, crop_y, m_impl->target_width, m_impl->target_height};
                        visiong::bufstate::prepare_device_read(src_dma, visiong::bufstate::BufferOwner::RGA);
                        visiong::bufstate::prepare_device_write(dst_dma,
                                                               visiong::bufstate::BufferOwner::RGA,
                                                               visiong::bufstate::AccessIntent::Overwrite);
                        if (imcrop(src_dma.get_buffer(), dst_dma.get_buffer(), crop_rect) == IM_STATUS_SUCCESS) {
                            cropped_yuv_frame =
                                ImageBuffer(m_impl->target_width, m_impl->target_height, raw_yuv_frame.format, crop_mb);
                            cropped_yuv_frame.w_stride = m_impl->crop_vir_width;
                            cropped_yuv_frame.h_stride = m_impl->crop_vir_height;
                            visiong::bufstate::mark_device_write(cropped_yuv_frame,
                                                                 visiong::bufstate::BufferOwner::RGA);
                            processing_buf = &cropped_yuv_frame;
                            geometry_fixed = true;
                        } else {
                            RK_MPI_MB_ReleaseMB(crop_mb);
                        }
                    } catch (...) {
                        RK_MPI_MB_ReleaseMB(crop_mb);
                    }
                } else {
                    RK_MPI_MB_ReleaseMB(crop_mb);
                }
            }
        }

        if (!geometry_fixed && can_center_crop) {
            try {
                cropped_yuv_frame =
                    raw_yuv_frame.crop(std::make_tuple(crop_x, crop_y, m_impl->target_width, m_impl->target_height));
                processing_buf = &cropped_yuv_frame;
                geometry_fixed = true;
            } catch (const std::exception& e) {
                VISIONG_LOG_WARN("Camera", "Center crop failed, will retry with resize fallback: " << e.what());
            }
        }

        if (!geometry_fixed) {
            try {
                // Fall back to resize when cropping is disabled or the driver returned an odd geometry.
                // 关闭裁切或驱动返回意外尺寸时，退回到缩放以保证输出尺寸稳定。
                cropped_yuv_frame = raw_yuv_frame.resize(m_impl->target_width, m_impl->target_height);
                processing_buf = &cropped_yuv_frame;
            } catch (const std::exception& e) {
                VISIONG_LOG_ERROR("Camera", "Failed to normalize snapshot geometry: " << e.what());
            }
        }
    }

    // If output already matches requested YUV format, return without extra copy. / 如果输出已经符合请求的 YUV 格式，则直接返回而不再额外拷贝。
    if (processing_buf->format == m_impl->output_format) {
        if (processing_buf == &raw_yuv_frame)
            return std::move(raw_yuv_frame);
        if (processing_buf == &cropped_yuv_frame)
            return std::move(cropped_yuv_frame);
        return processing_buf->copy();
    }

    // Zero-copy path: source is DMA YUV, destination is RGB/BGR with convert pool. / 零拷贝路径：源为 DMA YUV，目标为使用 convert 池的 RGB/BGR。
    const bool want_rgb_bgr = (m_impl->output_format == RK_FMT_RGB888 || m_impl->output_format == RK_FMT_BGR888);
    const bool src_is_yuv =
        (processing_buf->format == RK_FMT_YUV420SP || processing_buf->format == RK_FMT_YUV420SP_VU);
    if (want_rgb_bgr && src_is_yuv && processing_buf->is_zero_copy() && processing_buf->get_dma_fd() >= 0 &&
        m_impl->convert_pool != MB_INVALID_POOLID) {
        const int w = processing_buf->width;
        const int h = processing_buf->height;
        const size_t rgb_size = static_cast<size_t>(w) * h * 3;
        MB_BLK mb_blk = RK_MPI_MB_GetMB(m_impl->convert_pool, static_cast<RK_U64>(rgb_size), RK_TRUE);
        if (mb_blk != MB_INVALID_HANDLE) {
            void* dst_vir = RK_MPI_MB_Handle2VirAddr(mb_blk);
            const int fd = RK_MPI_MB_Handle2Fd(mb_blk);
            const size_t mb_size = RK_MPI_MB_GetSize(mb_blk);
            int w_stride = (w + 15) & ~15;
            RgaDmaBuffer src_dma(processing_buf->get_dma_fd(), const_cast<void*>(processing_buf->get_data()),
                                 processing_buf->get_size(), w, h, static_cast<int>(processing_buf->format),
                                 processing_buf->w_stride, processing_buf->h_stride);
            RgaDmaBuffer dst_dma(fd, dst_vir, mb_size, w, h, static_cast<int>(m_impl->output_format), w_stride, h);
            int src_rga = (processing_buf->format == RK_FMT_YUV420SP_VU) ? RK_FORMAT_YCrCb_420_SP
                                                                         : RK_FORMAT_YCbCr_420_SP;
            int dst_rga = (m_impl->output_format == RK_FMT_RGB888) ? RK_FORMAT_RGB_888 : RK_FORMAT_BGR_888;
            visiong::bufstate::prepare_device_read(src_dma, visiong::bufstate::BufferOwner::RGA);
            visiong::bufstate::prepare_device_write(dst_dma,
                                                   visiong::bufstate::BufferOwner::RGA,
                                                   visiong::bufstate::AccessIntent::Overwrite);
            if (imcvtcolor(src_dma.get_buffer(), dst_dma.get_buffer(), src_rga, dst_rga) ==
                IM_STATUS_SUCCESS) {
                ImageBuffer converted(w, h, m_impl->output_format, mb_blk);
                visiong::bufstate::mark_device_write(converted, visiong::bufstate::BufferOwner::RGA);
                return converted;
            }
            RK_MPI_MB_ReleaseMB(mb_blk);
        }
    }

    try {
        return processing_buf->to_format(m_impl->output_format);
    } catch (const std::exception& e) {
        VISIONG_LOG_ERROR("Camera", "Failed to convert frame to target format: " << e.what());
        return processing_buf->copy();
    }
}
void Camera::skip_frames(int num_frames) {
    if (!m_impl->initialized.load()) {
        throw std::runtime_error("skip_frames: Camera is not initialized. Please call init() first.");
    }
    if (num_frames <= 0) {
        return;
    }

    for (int i = 0; i < num_frames; ++i) {
        ImageBuffer discarded_frame = this->snapshot();
        if (!discarded_frame.is_valid()) {
            VISIONG_LOG_WARN("Camera", "Failed to capture a valid frame while skipping (at frame "
                                           << i + 1 << "/" << num_frames
                                           << "). Stopping the skip process.");
            break;
        }
    }
    VISIONG_LOG_INFO("Camera", "Frame skipping complete.");
}
void Camera::release() {
    if (!m_impl->initialized.load())
        return;
    m_impl->isp_controller.reset();
    if (m_impl->v4l2_capture) {
        m_impl->v4l2_capture->close();
        m_impl->v4l2_capture.reset();
    }
    if (m_impl->crop_pool != MB_INVALID_POOLID) {
        RK_MPI_MB_DestroyPool(m_impl->crop_pool);
        m_impl->crop_pool = MB_INVALID_POOLID;
    }
    m_impl->crop_vir_width = 0;
    m_impl->crop_vir_height = 0;
    m_impl->crop_mb_size = 0;
    if (m_impl->convert_pool != MB_INVALID_POOLID) {
        RK_MPI_MB_DestroyPool(m_impl->convert_pool);
        m_impl->convert_pool = MB_INVALID_POOLID;
    }
    m_impl->initialized = false;
    VISIONG_LOG_INFO("Camera", "Resources released.");
}

bool Camera::is_initialized() const {
    return m_impl->initialized.load();
}
int Camera::get_target_width() const {
    return m_impl->target_width;
}

int Camera::get_target_height() const {
    return m_impl->target_height;
}
int Camera::get_actual_capture_width() const {
    return m_impl->initialized && m_impl->v4l2_capture ? m_impl->v4l2_capture->get_width() : 0;
}
int Camera::get_actual_capture_height() const {
    return m_impl->initialized && m_impl->v4l2_capture ? m_impl->v4l2_capture->get_height() : 0;
}
std::string Camera::get_crop_mode() const {
    return m_impl->crop_mode.label;
}

// ISP Control Methods / ISP 控制方法。
void Camera::set_saturation(int value) {
    require_isp_controller(m_impl->isp_controller, "set_saturation").set_saturation(value);
}
void Camera::set_contrast(int value) {
    require_isp_controller(m_impl->isp_controller, "set_contrast").set_contrast(value);
}
void Camera::set_brightness(int value) {
    require_isp_controller(m_impl->isp_controller, "set_brightness").set_brightness(value);
}
void Camera::set_sharpness(int value) {
    require_isp_controller(m_impl->isp_controller, "set_sharpness").set_sharpness(value);
}
void Camera::set_hue(int value) {
    require_isp_controller(m_impl->isp_controller, "set_hue").set_hue(value);
}
void Camera::set_white_balance_mode(const std::string& mode) {
    require_isp_controller(m_impl->isp_controller, "set_white_balance_mode").set_white_balance_mode(mode);
}
void Camera::set_white_balance_temperature(int temp) {
    require_isp_controller(m_impl->isp_controller, "set_white_balance_temperature")
        .set_white_balance_temperature(temp);
}
void Camera::set_exposure_mode(const std::string& mode) {
    require_isp_controller(m_impl->isp_controller, "set_exposure_mode").set_exposure_mode(mode);
}
void Camera::set_exposure_time(float time_s) {
    require_isp_controller(m_impl->isp_controller, "set_exposure_time").set_exposure_time(time_s);
}
void Camera::set_exposure_gain(int gain) {
    require_isp_controller(m_impl->isp_controller, "set_exposure_gain").set_exposure_gain(gain);
}
void Camera::set_frame_rate(int fps) {
    require_isp_controller(m_impl->isp_controller, "set_frame_rate").set_frame_rate(fps);
}
void Camera::set_power_line_frequency(const std::string& mode) {
    require_isp_controller(m_impl->isp_controller, "set_power_line_frequency").set_power_line_frequency(mode);
}
void Camera::set_flip(bool flip, bool mirror) {
    require_isp_controller(m_impl->isp_controller, "set_flip").set_flip(flip, mirror);
}
void Camera::set_spatial_denoise_level(int level) {
    require_isp_controller(m_impl->isp_controller, "set_spatial_denoise_level").set_spatial_denoise_level(level);
}
void Camera::set_temporal_denoise_level(int level) {
    require_isp_controller(m_impl->isp_controller, "set_temporal_denoise_level").set_temporal_denoise_level(level);
}

int Camera::get_saturation() {
    return m_impl->isp_controller ? m_impl->isp_controller->get_saturation() : -1;
}
int Camera::get_contrast() {
    return m_impl->isp_controller ? m_impl->isp_controller->get_contrast() : -1;
}
int Camera::get_brightness() {
    return m_impl->isp_controller ? m_impl->isp_controller->get_brightness() : -1;
}
int Camera::get_sharpness() {
    return m_impl->isp_controller ? m_impl->isp_controller->get_sharpness() : -1;
}
int Camera::get_hue() {
    return m_impl->isp_controller ? m_impl->isp_controller->get_hue() : -1;
}
std::string Camera::get_white_balance_mode() {
    return m_impl->isp_controller ? m_impl->isp_controller->get_white_balance_mode() : "error";
}
int Camera::get_white_balance_temperature() {
    return m_impl->isp_controller ? m_impl->isp_controller->get_white_balance_temperature() : -1;
}
std::string Camera::get_exposure_mode() {
    return m_impl->isp_controller ? m_impl->isp_controller->get_exposure_mode() : "error";
}
float Camera::get_exposure_time() {
    return m_impl->isp_controller ? m_impl->isp_controller->get_exposure_time() : -1.0f;
}
int Camera::get_exposure_gain() {
    return m_impl->isp_controller ? m_impl->isp_controller->get_exposure_gain() : -1;
}

void Camera::lock_focus() {
    require_isp_controller(m_impl->isp_controller, "lock_focus").lock_focus();
}

void Camera::unlock_focus() {
    require_isp_controller(m_impl->isp_controller, "unlock_focus").unlock_focus();
}

void Camera::trigger_focus() {
    require_isp_controller(m_impl->isp_controller, "trigger_focus").trigger_focus();
}
void Camera::set_focus_mode(const std::string& mode) {
    auto& controller = require_isp_controller(m_impl->isp_controller, "set_focus_mode");

    if (mode == "continuous") {
        controller.set_focus_mode("continuous-picture");
    } else if (mode == "manual") {
        controller.set_focus_mode("manual");
    } else {
        throw std::invalid_argument("Invalid focus mode '" + mode + "'. Use 'continuous' or 'manual'.");
    }
}

void Camera::set_manual_focus(int position) {
    require_isp_controller(m_impl->isp_controller, "set_manual_focus").set_manual_focus_position(position);
}

int Camera::get_focus_position() {
    if (!m_impl->isp_controller) {
        VISIONG_LOG_WARN("Camera", "Cannot get focus position, ISP controller is null.");
        return -1;
    }
    return m_impl->isp_controller->get_focus_position();
}
