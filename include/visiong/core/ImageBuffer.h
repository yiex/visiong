// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_CORE_IMAGEBUFFER_H
#define VISIONG_CORE_IMAGEBUFFER_H

#include <cstddef>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "visiong/common/vision_types.h"
#include "rk_comm_video.h"
#include "rk_comm_mb.h"

namespace cv {
class Mat;
}

class RgaDmaBuffer;


class ImageBuffer {
private:
    std::shared_ptr<void> m_mb_blk_handle_sptr;
    std::shared_ptr<RgaDmaBuffer> m_dma_buf_sptr;
    void* m_vir_addr;
    size_t m_size;
    std::vector<unsigned char> m_user_data;
    bool m_is_zero_copy;
    std::shared_ptr<void> m_external_keep_alive;
    MB_BLK m_mb_blk;
    int m_dma_fd;
    mutable std::unique_ptr<ImageBuffer> m_cached_bgr;
    mutable std::unique_ptr<ImageBuffer> m_cached_gray;

public:
    const ImageBuffer& get_bgr_version() const;
    const ImageBuffer& get_gray_version() const;

    void save_hsv_bin(const std::string& filepath) const;
    void save_venc_jpg(const std::string& filepath, int quality = 75) const;
    void save_venc_h264(
        const std::string& filepath,
        int quality = 75,
        const std::string& rc_mode = "cbr",
        int fps = 30,
        bool append = true,
        const std::string& container = "auto",
        bool mp4_faststart = true) const;
    void save_venc_h265(
        const std::string& filepath,
        int quality = 75,
        const std::string& rc_mode = "cbr",
        int fps = 30,
        bool append = true,
        const std::string& container = "auto",
        bool mp4_faststart = true) const;

public:
    int width;
    int height;
    int w_stride;
    int h_stride;
    PIXEL_FORMAT_E format;

    ImageBuffer();
    ImageBuffer(int w, int h, PIXEL_FORMAT_E fmt, std::vector<unsigned char>&& data);
    ImageBuffer(int w, int h, PIXEL_FORMAT_E fmt, void* ptr, size_t size, std::shared_ptr<void> keep_alive);
    ImageBuffer(int w, int h, PIXEL_FORMAT_E fmt, MB_BLK mb_blk);
    ImageBuffer(int w, int h, PIXEL_FORMAT_E fmt, std::shared_ptr<RgaDmaBuffer> dma_buf);
    ~ImageBuffer();

    ImageBuffer(const ImageBuffer& other);
    ImageBuffer& operator=(const ImageBuffer& other);
    ImageBuffer(ImageBuffer&& other) noexcept;
    ImageBuffer& operator=(ImageBuffer&& other) noexcept;

    static ImageBuffer load(const std::string& filepath);
    void save(const std::string& filepath, int quality = 75) const;
    static ImageBuffer from_cv_mat(const cv::Mat& mat, PIXEL_FORMAT_E format);
    ImageBuffer& copy_from_dma(const RgaDmaBuffer& dma_buf);

    const void* get_data() const;
    void* get_data();
    size_t get_size() const;
    bool is_valid() const;
    bool is_zero_copy() const { return m_is_zero_copy; }
    MB_BLK get_mb_blk() const { return m_mb_blk; }
    int get_dma_fd() const { return m_dma_fd; }

    std::vector<Blob> find_blobs(
        const std::vector<std::tuple<int, int, int, int, int, int>>& thresholds,
        bool invert = false,
        const std::tuple<int, int, int, int>& roi = std::make_tuple(0, 0, 0, 0),
        int x_stride = 2,
        int y_stride = 2,
        int area_threshold = 10,
        int pixels_threshold = 10,
        bool merge = true,
        int margin = 10,
        int mode = 0,
        int erode_size = 0,
        int dilate_size = 0) const;
    std::vector<Blob> find_blobs(
        const std::vector<std::tuple<int, int>>& gray_thresholds,
        bool invert = false,
        const std::tuple<int, int, int, int>& roi = std::make_tuple(0, 0, 0, 0),
        int x_stride = 2,
        int y_stride = 2,
        int area_threshold = 10,
        int pixels_threshold = 10,
        bool merge = true,
        int margin = 10,
        int mode = 0,
        int erode_size = 0,
        int dilate_size = 0) const;
    std::vector<Line> find_lines(
        const std::tuple<int, int, int, int>& roi_tuple,
        int x_stride,
        int y_stride,
        int threshold,
        double rho_resolution_px,
        double theta_resolution_deg,
        int canny_low_thresh,
        int canny_high_thresh) const;
    std::vector<Circle> find_circles(
        const std::tuple<int, int, int, int>& roi_tuple,
        int x_stride,
        int y_stride,
        int threshold,
        int r_min_param,
        int r_max_param,
        int r_step,
        int canny_low_thresh,
        int canny_high_thresh) const;
    std::vector<Polygon> find_polygons(
        const std::tuple<int, int, int, int>& roi_tuple,
        int min_area,
        int max_area,
        int min_sides,
        int max_sides,
        int accuracy) const;
    std::vector<QRCode> find_qrcodes() const;

    using Square = Polygon;
    std::vector<Square> find_squares(
        const std::tuple<int, int, int, int>& roi_tuple,
        int threshold,
        int min_area,
        double corner_scale,
        int corner_sample_radius,
        double corner_ratio_threshold,
        int edge_offset,
        int edge_sample_num,
        double edge_binary_threshold,
        int inside_sample_num,
        double inside_white_ratio_threshold,
        double overlap_ratio_threshold) const;

    ImageBuffer binarize(
        const std::string& method,
        const std::tuple<int, int>& threshold,
        bool invert,
        int x_stride,
        int y_stride,
        int roi_x,
        int roi_y) const;
    ImageBuffer copy() const;
    ImageBuffer to_format(PIXEL_FORMAT_E new_format) const;
    ImageBuffer to_grayscale() const;
    ImageBuffer resize(int new_width, int new_height) const;
    ImageBuffer crop(const std::tuple<int, int, int, int>& rect_tuple) const;
    ImageBuffer letterbox(int target_width, int target_height, std::tuple<unsigned char, unsigned char, unsigned char> color) const;

    static ImageBuffer create(
        int w,
        int h,
        PIXEL_FORMAT_E fmt,
        std::tuple<unsigned char, unsigned char, unsigned char> color_rgb);
    static ImageBuffer create(
        int w,
        int h,
        PIXEL_FORMAT_E fmt,
        std::tuple<unsigned char, unsigned char, unsigned char, unsigned char> color_rgba);

    ImageBuffer warp_perspective(const Polygon& quad, int out_width, int out_height) const;

    ImageBuffer& draw_line(
        int x0,
        int y0,
        int x1,
        int y1,
        std::tuple<unsigned char, unsigned char, unsigned char> color_rgb,
        int thickness);
    ImageBuffer& draw_rectangle(
        int x,
        int y,
        int w,
        int h,
        std::tuple<unsigned char, unsigned char, unsigned char> color_rgb,
        int thickness,
        bool fill);
    ImageBuffer& draw_rectangle(
        const std::tuple<int, int, int, int>& rect_tuple,
        std::tuple<unsigned char, unsigned char, unsigned char> color_rgb,
        int thickness,
        bool fill);
    ImageBuffer& draw_string(
        int x,
        int y,
        const std::string& text,
        std::tuple<unsigned char, unsigned char, unsigned char> color_rgb,
        double scale,
        int thickness);

    static void set_text_font(
        const std::string& font_path = "",
        const std::string& pre_chars = "",
        size_t glyph_budget = 6623);
    static void clear_text_font();
    ImageBuffer& draw_cross(
        int cx,
        int cy,
        std::tuple<unsigned char, unsigned char, unsigned char> color_rgb,
        int size,
        int thickness);
    ImageBuffer& draw_circle(
        int cx,
        int cy,
        int radius,
        std::tuple<unsigned char, unsigned char, unsigned char> color_rgb,
        int thickness,
        bool fill);
    ImageBuffer& paste(const ImageBuffer& img_to_paste, int x, int y);
    ImageBuffer& blend(const ImageBuffer& img_to_blend, int x = 0, int y = 0);

    ImageBuffer rotate(int angle_degrees) const;
    ImageBuffer flip(bool horizontal, bool vertical) const;
};

#endif  // VISIONG_CORE_IMAGEBUFFER_H

