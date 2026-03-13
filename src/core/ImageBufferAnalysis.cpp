// SPDX-License-Identifier: LGPL-3.0-or-later
#include "visiong/core/ImageBuffer.h"
#include "visiong/core/BufferStateMachine.h"
#include "visiong/core/RgaHelper.h"
#include "core/internal/rga_utils.h"
#include "visiong/common/build_config.h"
#include "visiong/common/pixel_format.h"
#if VISIONG_WITH_IVE
#include "visiong/modules/IVE.h"
#endif
#include "quirc.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#ifndef CV_PI
#define CV_PI 3.1415926535897932384626433832795
#endif
static cv::Mat image_buffer_to_gray_mat_view(const ImageBuffer& img_buf) {
    if (!img_buf.is_valid()) return cv::Mat();
    const ImageBuffer& gray_version = img_buf.get_gray_version();
    visiong::bufstate::prepare_cpu_read(gray_version);

    cv::Mat view(gray_version.height, gray_version.w_stride, CV_8UC1, const_cast<void*>(gray_version.get_data()));
    
    if (gray_version.w_stride != gray_version.width) {
        return view(cv::Rect(0, 0, gray_version.width, gray_version.height));
    }
    return view;
}

static void threshold_hsv_pipeline(uint8_t* mask, const uint8_t* hsv_data, int count,
    uint8_t h_min, uint8_t h_max, 
    uint8_t s_min, uint8_t s_max,
    uint8_t v_min, uint8_t v_max,
    bool invert) {

    bool h_wrap = (h_min > h_max);

#if defined(__ARM_NEON)
    // ------------------------------------------------------------------------
    uint8x16_t v_h_min = vdupq_n_u8(h_min);
    uint8x16_t v_h_max = vdupq_n_u8(h_max);
    uint8x16_t v_s_min = vdupq_n_u8(s_min);
    uint8x16_t v_s_max = vdupq_n_u8(s_max);
    uint8x16_t v_v_min = vdupq_n_u8(v_min);
    uint8x16_t v_v_max = vdupq_n_u8(v_max);

    int i = 0;
    for (; i <= count - 16; i += 16) {
        uint8x16x3_t hsv = vld3q_u8(hsv_data + i * 3);
        // hsv.val[0] = H, hsv.val[1] = S, hsv.val[2] = V
        // hsv.val[0] 表示 H，hsv.val[1] 表示 S，hsv.val[2] 表示 V。

        uint8x16_t h_mask;
        if (h_wrap) {
            uint8x16_t h_ge = vcgeq_u8(hsv.val[0], v_h_min);
            uint8x16_t h_le = vcleq_u8(hsv.val[0], v_h_max);
            h_mask = vorrq_u8(h_ge, h_le); // OR / 或
        } else {
            uint8x16_t h_ge = vcgeq_u8(hsv.val[0], v_h_min);
            uint8x16_t h_le = vcleq_u8(hsv.val[0], v_h_max);
            h_mask = vandq_u8(h_ge, h_le); // AND / 与
        }

        uint8x16_t s_mask = vandq_u8(vcgeq_u8(hsv.val[1], v_s_min), vcleq_u8(hsv.val[1], v_s_max));
        uint8x16_t v_mask = vandq_u8(vcgeq_u8(hsv.val[2], v_v_min), vcleq_u8(hsv.val[2], v_v_max));

        uint8x16_t res = vandq_u8(vandq_u8(h_mask, s_mask), v_mask);

        if (invert) {
            res = vmvnq_u8(res);
        }

        vst1q_u8(mask + i, res);
    }
#else
    int i = 0;
#endif

    // ------------------------------------------------------------------------
    // ------------------------------------------------------------------------
    for (; i < count; i++) {
        uint8_t h = hsv_data[i * 3 + 0];
        uint8_t s = hsv_data[i * 3 + 1];
        uint8_t v = hsv_data[i * 3 + 2];

        bool h_match;
        if (h_wrap) {
            h_match = (h >= h_min || h <= h_max);
        } else {
            h_match = (h >= h_min && h <= h_max);
        }

        bool match = h_match && (s >= s_min && s <= s_max) && (v >= v_min && v <= v_max);
        
        if (invert) match = !match;

        mask[i] = match ? 255 : 0;
    }
}

// ============================================================================
static void threshold_grayscale_in_range(uint8_t* mask, const uint8_t* gray, int count,
    uint8_t low, uint8_t high, bool invert) {
    int i = 0;
#if defined(__ARM_NEON)
    uint8x16_t v_low = vdupq_n_u8(low);
    uint8x16_t v_high = vdupq_n_u8(high);
    for (; i <= count - 16; i += 16) {
        uint8x16_t v = vld1q_u8(gray + i);
        uint8x16_t ge = vcgeq_u8(v, v_low);
        uint8x16_t le = vcleq_u8(v, v_high);
        uint8x16_t res = vandq_u8(ge, le);
        if (invert) res = vmvnq_u8(res);
        vst1q_u8(mask + i, res);
    }
#endif
    for (; i < count; i++) {
        uint8_t v = gray[i];
        bool in_range = (v >= low && v <= high);
        if (invert) in_range = !in_range;
        mask[i] = in_range ? 255 : 0;
    }
}

std::vector<Blob> ImageBuffer::find_blobs(
    const std::vector<std::tuple<int, int, int, int, int, int>>& thresholds,
    bool invert,
    const std::tuple<int, int, int, int>& roi_tuple,
    int x_stride, int y_stride,
    int area_threshold_val, int pixels_threshold_val,
    bool merge_blobs, int margin_val,
    int mode,
    int erode_size,
    int dilate_size
) const {
    (void)x_stride;
    (void)y_stride;
    
    std::vector<Blob> found_blobs;
    if (!is_valid()) return found_blobs;

    ImageBuffer roi_img_buf_owner;
    const ImageBuffer* processing_buf = this;
    int roi_x_offset = 0, roi_y_offset = 0;

    bool use_roi_flag = (std::get<2>(roi_tuple) > 0 && std::get<3>(roi_tuple) > 0);
    if (use_roi_flag) {
        try {
            roi_img_buf_owner = this->crop(roi_tuple);
            processing_buf = &roi_img_buf_owner;
            roi_x_offset = std::get<0>(roi_tuple);
            roi_y_offset = std::get<1>(roi_tuple);
        } catch (const std::exception& e) {
            throw std::runtime_error("find_blobs: ROI crop failed: " + std::string(e.what()));
        }
    }

    int w = processing_buf->width;
    int h = processing_buf->height;
    std::vector<uint8_t> mask_data(w * h);

    constexpr PIXEL_FORMAT_E GRAY8 = visiong::kGray8Format;
    if (processing_buf->format == GRAY8) {
        if (thresholds.size() != 1) return found_blobs;
        int g_lo = std::get<0>(thresholds[0]), g_hi = std::get<1>(thresholds[0]);
        uint8_t gray_low  = static_cast<uint8_t>(std::max(0, std::min(255, g_lo)));
        uint8_t gray_high = static_cast<uint8_t>(std::max(0, std::min(255, g_hi)));
        if (gray_low > gray_high) std::swap(gray_low, gray_high);

        threshold_grayscale_in_range(mask_data.data(), (const uint8_t*)processing_buf->get_data(), w * h, gray_low, gray_high, false);
        if (invert)
            for (size_t i = 0; i < (size_t)(w * h); i++) mask_data[i] = 255u - mask_data[i];

        cv::Mat final_mask(h, w, CV_8UC1, mask_data.data());
        if (erode_size > 0 || dilate_size > 0) {
            if (erode_size == 3 || erode_size == 5) {
                cv::Mat ke = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(erode_size, erode_size));
                cv::erode(final_mask, final_mask, ke);
            }
            if (dilate_size == 3 || dilate_size == 5) {
                cv::Mat kd = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(dilate_size, dilate_size));
                cv::dilate(final_mask, final_mask, kd);
            }
        }

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(final_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for (const auto& contour : contours) {
            double pixels = cv::contourArea(contour);
            cv::Rect rect = cv::boundingRect(contour);
            if (pixels < pixels_threshold_val || rect.area() < area_threshold_val) continue;
            cv::Moments m = cv::moments(contour);
            int cx = (m.m00 == 0) ? (rect.x + rect.width / 2) : static_cast<int>(m.m10 / m.m00);
            int cy = (m.m00 == 0) ? (rect.y + rect.height / 2) : static_cast<int>(m.m01 / m.m00);
            found_blobs.emplace_back(rect.x + roi_x_offset, rect.y + roi_y_offset, rect.width, rect.height,
                                    cx + roi_x_offset, cy + roi_y_offset, static_cast<int>(pixels), 1u);
        }
        return found_blobs;
    }

    ImageBuffer color_space_buf;
    cv::Mat lab_mat;
    bool use_custom_pipeline = (mode == 0);

    if (mode == 0) {
#if VISIONG_WITH_IVE
        if (processing_buf->format == RK_FMT_YUV420SP || processing_buf->format == RK_FMT_YUV420SP_VU) {
            auto& ive = IVE::get_instance();
            color_space_buf = ive.yuv_to_hsv(*processing_buf, true);
        } else
#endif
        {
            // Fall back to OpenCV HSV conversion when IVE is unavailable.
            // 当 IVE 不可用时，回退到 OpenCV 的 HSV 转换路径。
            const ImageBuffer& bgr = processing_buf->get_bgr_version();
            visiong::bufstate::prepare_cpu_read(bgr);
            cv::Mat bgr_mat(
                bgr.height,
                bgr.width,
                CV_8UC3,
                const_cast<void*>(bgr.get_data()),
                static_cast<size_t>(bgr.w_stride) * 3);
            cv::Mat hsv_mat;
            cv::cvtColor(bgr_mat, hsv_mat, cv::COLOR_BGR2HSV_FULL);
            const size_t size = hsv_mat.total() * hsv_mat.elemSize();
            std::vector<unsigned char> data(hsv_mat.data, hsv_mat.data + size);
            color_space_buf = ImageBuffer(hsv_mat.cols, hsv_mat.rows, RK_FMT_RGB888, std::move(data));
        }
    } else {
        const ImageBuffer& bgr = processing_buf->get_bgr_version();
        visiong::bufstate::prepare_cpu_read(bgr);
        cv::Mat bgr_mat(
            bgr.height,
            bgr.width,
            CV_8UC3,
            const_cast<void*>(bgr.get_data()),
            static_cast<size_t>(bgr.w_stride) * 3);

        cv::cvtColor(bgr_mat, lab_mat, cv::COLOR_BGR2Lab);
    }

    unsigned int current_threshold_bit_code = 1;

    for (const auto& th : thresholds) {
        if (use_custom_pipeline) {
            threshold_hsv_pipeline(mask_data.data(), (const uint8_t*)color_space_buf.get_data(), 
                                   w * h,
                                   (uint8_t)std::get<0>(th), (uint8_t)std::get<1>(th), // H
                                   (uint8_t)std::get<2>(th), (uint8_t)std::get<3>(th), // S
                                   (uint8_t)std::get<4>(th), (uint8_t)std::get<5>(th), // V
                                   invert);
        } else {
            cv::Scalar lower_b(std::get<0>(th), std::get<2>(th), std::get<4>(th));
            cv::Scalar upper_b(std::get<1>(th), std::get<3>(th), std::get<5>(th));

            cv::Mat mask_mat(h, w, CV_8UC1, mask_data.data());
            
            cv::inRange(lab_mat, lower_b, upper_b, mask_mat);
            if (invert) cv::bitwise_not(mask_mat, mask_mat);
        }

        if (erode_size > 0 || dilate_size > 0) {
#if VISIONG_WITH_IVE
            auto& ive = IVE::get_instance();
            ImageBuffer mask_buf(
                w,
                h,
                visiong::kGray8Format,
                mask_data.data(),
                mask_data.size(),
                nullptr);

            if (erode_size > 0) mask_buf = ive.erode(mask_buf, erode_size);
            if (dilate_size > 0) mask_buf = ive.dilate(mask_buf, dilate_size);

            if (mask_buf.get_data() != mask_data.data()) {
                memcpy(mask_data.data(), mask_buf.get_data(), static_cast<size_t>(w * h));
            }
#else
            // Keep blob morphology available even in slim builds without IVE.
            // 即便在不含 IVE 的精简构建中，也保留 blob 形态学处理能力。
            cv::Mat final_mask(h, w, CV_8UC1, mask_data.data());
            if (erode_size > 0) {
                cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(erode_size, erode_size));
                cv::erode(final_mask, final_mask, kernel);
            }
            if (dilate_size > 0) {
                cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(dilate_size, dilate_size));
                cv::dilate(final_mask, final_mask, kernel);
            }
#endif
        }

        cv::Mat final_mask(h, w, CV_8UC1, mask_data.data());
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(final_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        for (const auto& contour : contours) {
            double pixels = cv::contourArea(contour);
            cv::Rect rect = cv::boundingRect(contour);
            
            if (pixels < pixels_threshold_val || rect.area() < area_threshold_val) continue;
            
            cv::Moments m = cv::moments(contour);
            int cx = (m.m00 == 0) ? (rect.x + rect.width / 2) : static_cast<int>(m.m10 / m.m00);
            int cy = (m.m00 == 0) ? (rect.y + rect.height / 2) : static_cast<int>(m.m01 / m.m00);
            
            found_blobs.emplace_back(rect.x + roi_x_offset, rect.y + roi_y_offset, rect.width, rect.height,
                                     cx + roi_x_offset, cy + roi_y_offset, static_cast<int>(pixels), current_threshold_bit_code);
        }
        current_threshold_bit_code <<= 1;
    }

    if (merge_blobs && found_blobs.size() > 1) {
        bool merged_in_pass = true;
        while (merged_in_pass) {
            merged_in_pass = false;
            std::vector<bool> to_be_removed(found_blobs.size(), false);
            for (size_t i = 0; i < found_blobs.size(); ++i) {
                if (to_be_removed[i]) continue;
                for (size_t j = i + 1; j < found_blobs.size(); ++j) {
                    if (to_be_removed[j]) continue;
                    
                    cv::Rect r1(found_blobs[i].x - margin_val, found_blobs[i].y - margin_val,
                                found_blobs[i].w + 2 * margin_val, found_blobs[i].h + 2 * margin_val);
                    cv::Rect r2(found_blobs[j].x, found_blobs[j].y, found_blobs[j].w, found_blobs[j].h);
                    
                    if ((r1 & r2).area() > 0) {
                        cv::Rect original_r1(found_blobs[i].x, found_blobs[i].y, found_blobs[i].w, found_blobs[i].h);
                        cv::Rect union_rect = original_r1 | r2;
                        
                        double total_pixels = found_blobs[i].pixels + found_blobs[j].pixels;
                        found_blobs[i].cx = (found_blobs[i].cx * found_blobs[i].pixels + found_blobs[j].cx * found_blobs[j].pixels) / total_pixels;
                        found_blobs[i].cy = (found_blobs[i].cy * found_blobs[i].pixels + found_blobs[j].cy * found_blobs[j].pixels) / total_pixels;
                        found_blobs[i].pixels = total_pixels;
                        found_blobs[i].x = union_rect.x;
                        found_blobs[i].y = union_rect.y;
                        found_blobs[i].w = union_rect.width;
                        found_blobs[i].h = union_rect.height;
                        found_blobs[i].code |= found_blobs[j].code;
                        
                        to_be_removed[j] = true;
                        merged_in_pass = true;
                    }
                }
            }
            if (merged_in_pass) {
                std::vector<Blob> temp_blobs;
                temp_blobs.reserve(found_blobs.size());
                for(size_t i = 0; i < found_blobs.size(); ++i) {
                    if (!to_be_removed[i]) temp_blobs.push_back(found_blobs[i]);
                }
                found_blobs = std::move(temp_blobs);
            }
        }
    }
    
    return found_blobs;
}

std::vector<Blob> ImageBuffer::find_blobs(
    const std::vector<std::tuple<int, int>>& gray_thresholds,
    bool invert,
    const std::tuple<int, int, int, int>& roi_tuple,
    int x_stride, int y_stride,
    int area_threshold_val, int pixels_threshold_val,
    bool merge_blobs, int margin_val,
    int mode,
    int erode_size,
    int dilate_size
) const {
    constexpr PIXEL_FORMAT_E GRAY8 = visiong::kGray8Format;
    if (!is_valid())
        return std::vector<Blob>();
    if (format != GRAY8)
        throw std::invalid_argument("find_blobs([(gray_min, gray_max)]) is only for grayscale images; use img.to_grayscale() first, or use 6-element color thresholds.");
    if (gray_thresholds.size() != 1)
        throw std::invalid_argument("Grayscale find_blobs allows only one threshold tuple: [(gray_min, gray_max)].");
    std::vector<std::tuple<int, int, int, int, int, int>> th;
    th.push_back(std::make_tuple(std::get<0>(gray_thresholds[0]), std::get<1>(gray_thresholds[0]), 0, 0, 0, 0));
    return find_blobs(th, invert, roi_tuple, x_stride, y_stride, area_threshold_val, pixels_threshold_val, merge_blobs, margin_val, mode, erode_size, dilate_size);
}

std::vector<Polygon> ImageBuffer::find_polygons(const std::tuple<int, int, int, int>& roi_tuple, int min_area, int max_area, int min_sides, int max_sides, int accuracy) const {
    std::vector<Polygon> found_polygons_list;
    if (!is_valid()) return found_polygons_list;

    ImageBuffer roi_img_buf_owner;
    const ImageBuffer* processing_buf = this;
    int roi_x_offset = 0, roi_y_offset = 0;

    bool use_roi_flag = (std::get<2>(roi_tuple) > 0 && std::get<3>(roi_tuple) > 0);
    if (use_roi_flag) {
        try {
            roi_img_buf_owner = this->crop(roi_tuple);
            processing_buf = &roi_img_buf_owner;
            roi_x_offset = std::get<0>(roi_tuple);
            roi_y_offset = std::get<1>(roi_tuple);
        } catch (const std::exception& e) {
            throw std::runtime_error("find_polygons: ROI crop failed: " + std::string(e.what()));
        }
    }
    
    cv::Mat gray_mat = image_buffer_to_gray_mat_view(*processing_buf);
    if (gray_mat.empty()) return found_polygons_list;

    // Expect a user-provided binarized GRAY8 image; do not threshold here.
    // 这里要求调用方传入已经二值化的 GRAY8 图像，不在此处再次阈值化。
    cv::Mat binary_mat;
    gray_mat.copyTo(binary_mat);
    std::vector<std::vector<cv::Point>> contours;
    int contour_mode = cv::RETR_LIST;
    int contour_chain = cv::CHAIN_APPROX_SIMPLE;
    double epsilon_factor = 0.04; // default / 默认值
    if (accuracy <= 0) { // fast / 快速模式
        contour_mode = cv::RETR_EXTERNAL;
        contour_chain = cv::CHAIN_APPROX_SIMPLE;
        epsilon_factor = 0.06;
    } else if (accuracy >= 2) { // accurate / 高精度模式
        contour_mode = cv::RETR_LIST;
        contour_chain = cv::CHAIN_APPROX_NONE;
        epsilon_factor = 0.02;
    }
    cv::findContours(binary_mat, contours, contour_mode, contour_chain);
    
    for (const auto& cnt : contours) {
        double area = cv::contourArea(cnt);
        if (area < min_area || area > max_area) continue;
        
        std::vector<cv::Point> approx_cnt;
        double epsilon = epsilon_factor * cv::arcLength(cnt, true);
        cv::approxPolyDP(cnt, approx_cnt, epsilon, true);
        
        if (approx_cnt.size() >= static_cast<size_t>(min_sides) && approx_cnt.size() <= static_cast<size_t>(max_sides)) {
            Polygon current_poly;
            for(const auto& pt : approx_cnt) {
                current_poly.emplace_back(pt.x + roi_x_offset, pt.y + roi_y_offset);
            }
            found_polygons_list.push_back(current_poly);
        }
    }
    return found_polygons_list;
}

std::vector<Line> ImageBuffer::find_lines(const std::tuple<int, int, int, int>& roi_tuple, int x_stride, int y_stride, int threshold, double rho_resolution_px, double theta_resolution_deg, int canny_low_thresh, int canny_high_thresh) const {
    (void)x_stride;
    (void)y_stride;
    std::vector<Line> final_lines_list;
    if (!is_valid()) return final_lines_list;

    ImageBuffer roi_img_buf_owner;
    const ImageBuffer* processing_buf = this;
    int roi_x_offset = 0, roi_y_offset = 0;

    bool use_roi_flag = (std::get<2>(roi_tuple) > 0 && std::get<3>(roi_tuple) > 0);
    if (use_roi_flag) {
        try {
            roi_img_buf_owner = this->crop(roi_tuple);
            processing_buf = &roi_img_buf_owner;
            roi_x_offset = std::get<0>(roi_tuple);
            roi_y_offset = std::get<1>(roi_tuple);
        } catch (const std::exception& e) {
            throw std::runtime_error("find_lines: ROI crop failed: " + std::string(e.what()));
        }
    }

    cv::Mat gray_mat = image_buffer_to_gray_mat_view(*processing_buf);
    if (gray_mat.empty()) return final_lines_list;

    cv::Mat edges;
    cv::Canny(gray_mat, edges, canny_low_thresh, canny_high_thresh);
    
    std::vector<cv::Vec4i> hough_lines;
    cv::HoughLinesP(edges, hough_lines, rho_resolution_px, theta_resolution_deg * CV_PI / 180.0, threshold, 30, 10);
    
    if (hough_lines.empty()) {
        return final_lines_list;
    }

    std::map<int, std::vector<cv::Vec4i>> clustered_lines;
    int angle_granularity = 10;
    for(const auto& l : hough_lines) {
        int x1 = l[0], y1 = l[1], x2 = l[2], y2 = l[3];
        if (x1 == x2) continue; 
        double angle = atan2(y2 - y1, x2 - x1) * 180.0 / CV_PI;
        int angle_key = static_cast<int>(round(angle / angle_granularity)) * angle_granularity;
        clustered_lines[angle_key].push_back(cv::Vec4i(x1, y1, x2, y2));
    }

    for (const auto& [angle_key, lines_in_cluster] : clustered_lines) {
        if (lines_in_cluster.empty()) continue;

        float min_y = gray_mat.rows, max_y = 0;
        int representative_x1 = 0, representative_x2 = 0;

        for (const auto& line : lines_in_cluster) {
            min_y = std::min({min_y, (float)line[1], (float)line[3]});
            max_y = std::max({max_y, (float)line[1], (float)line[3]});
        }

        std::vector<cv::Point> all_points;
        for(const auto& line : lines_in_cluster) {
            all_points.push_back(cv::Point(line[0], line[1]));
            all_points.push_back(cv::Point(line[2], line[3]));
        }

        cv::Vec4f fit_line;
        cv::fitLine(all_points, fit_line, cv::DIST_L2, 0, 0.01, 0.01);

        float vx = fit_line[0], vy = fit_line[1], x0 = fit_line[2], y0 = fit_line[3];

        if (std::abs(vx) < 0.1) {
            representative_x1 = representative_x2 = static_cast<int>(x0);
        } else {
            float m_inv = vx / vy;
            representative_x1 = static_cast<int>((min_y - y0) * m_inv + x0);
            representative_x2 = static_cast<int>((max_y - y0) * m_inv + x0);
        }

        final_lines_list.emplace_back(representative_x1 + roi_x_offset, min_y + roi_y_offset,
                                      representative_x2 + roi_x_offset, max_y + roi_y_offset,
                                      lines_in_cluster.size());
    }

    return final_lines_list;
}

std::vector<Circle> ImageBuffer::find_circles(const std::tuple<int, int, int, int>& roi_tuple, int x_stride, int y_stride, int threshold, int r_min_param, int r_max_param, int r_step, int canny_low_thresh, int canny_high_thresh) const {
    (void)x_stride;
    (void)y_stride;
    (void)r_step;
    (void)canny_low_thresh;
    std::vector<Circle> found_circles_list;
    if (!is_valid()) return found_circles_list;

    ImageBuffer roi_img_buf_owner;
    const ImageBuffer* processing_buf = this;
    int roi_x_offset = 0, roi_y_offset = 0;

    bool use_roi_flag = (std::get<2>(roi_tuple) > 0 && std::get<3>(roi_tuple) > 0);
    if (use_roi_flag) {
        try {
            roi_img_buf_owner = this->crop(roi_tuple);
            processing_buf = &roi_img_buf_owner;
            roi_x_offset = std::get<0>(roi_tuple);
            roi_y_offset = std::get<1>(roi_tuple);
        } catch (const std::exception& e) {
            throw std::runtime_error("find_circles: ROI crop failed: " + std::string(e.what()));
        }
    }

    cv::Mat gray_mat = image_buffer_to_gray_mat_view(*processing_buf);
    if (gray_mat.empty()) return found_circles_list;

    cv::GaussianBlur(gray_mat, gray_mat, cv::Size(5, 5), 2, 2);
    
    std::vector<cv::Vec3f> hough_circles;
    cv::HoughCircles(gray_mat, hough_circles, cv::HOUGH_GRADIENT,
                     1,
                     gray_mat.rows / 8,
                     canny_high_thresh,
                     threshold,
                     r_min_param,
                     r_max_param == 0 ? gray_mat.rows / 2 : r_max_param);
    for (const auto& h_circle : hough_circles) {
        found_circles_list.emplace_back(cvRound(h_circle[0]) + roi_x_offset,
                                        cvRound(h_circle[1]) + roi_y_offset,
                                        cvRound(h_circle[2]));
    }
    return found_circles_list;
}

std::vector<QRCode> ImageBuffer::find_qrcodes() const {
    std::vector<QRCode> results;
    if (!is_valid()) {
        return results;
    }

    const ImageBuffer& gray_img = this->get_gray_version();
    if (!gray_img.is_valid()) {
        std::cerr << "find_qrcodes: Failed to get grayscale version of the image." << std::endl;
        return results;
    }

    quirc* qr_recognizer = quirc_new();
    if (!qr_recognizer) {
        std::cerr << "find_qrcodes: Failed to create quirc object." << std::endl;
        return results;
    }

    if (quirc_resize(qr_recognizer, gray_img.width, gray_img.height) < 0) {
        std::cerr << "find_qrcodes: Failed to resize quirc buffer." << std::endl;
        quirc_destroy(qr_recognizer);
        return results;
    }

    int w, h;
    unsigned char* qr_buffer = quirc_begin(qr_recognizer, &w, &h);

    // If source rows are tightly packed, use one memcpy for the whole image.
    // 如果源图像行数据紧密排列，则对整张图执行一次 memcpy。
    if (gray_img.width == gray_img.w_stride && w == gray_img.width) {
        size_t data_size = static_cast<size_t>(w) * h;
        if (gray_img.get_data() != nullptr) {
             memcpy(qr_buffer, gray_img.get_data(), data_size);
        } else {
             std::cerr << "find_qrcodes: gray_img data pointer is null despite being compact." << std::endl;
        }
    } else {
        copy_data_from_stride(qr_buffer, gray_img.get_data(), w, h, gray_img.w_stride);
    }

    quirc_end(qr_recognizer);

    int count = quirc_count(qr_recognizer);
    for (int i = 0; i < count; ++i) {
        struct quirc_code code;
        struct quirc_data data;
        quirc_decode_error_t err;

        quirc_extract(qr_recognizer, i, &code);
        err = quirc_decode(&code, &data);

        if (err == QUIRC_SUCCESS) {
            Polygon corners;
            for(int j = 0; j < 4; ++j) {
                corners.emplace_back(code.corners[j].x, code.corners[j].y);
            }
            results.emplace_back(corners, std::string(reinterpret_cast<char*>(data.payload), data.payload_len));
        }
    }

    quirc_destroy(qr_recognizer);
    return results;
}

// find_squares implementation.
// find_squares 的实现。


enum class CornerStatus {
    VALID,
    FAIL_SHORT_EDGE,
    FAIL_RATIO,
    FAIL_INTERNAL
};

enum class EdgeType {
    EXPOSED,
    INTERNAL,
    FAIL_SHORT,
    FAIL_OOB,
    FAIL_MIXED_SIDE,
    FAIL_SAME_COLOR,
    FAIL_UNKNOWN
};

struct EdgeCheckResult {
    EdgeType type;
    cv::Point2f inward_vector;
};

static CornerStatus get_corner_status(const cv::Point& p_curr, const cv::Point& p_prev, const cv::Point& p_next,
                                      const cv::Mat& binary_image, int sample_radius, double ratio_threshold) {
    int h = binary_image.rows;
    int w = binary_image.cols;

    cv::Point2f v1 = p_prev - p_curr;
    cv::Point2f v2 = p_next - p_curr;
    double len1 = cv::norm(v1);
    double len2 = cv::norm(v2);

    if (len1 < 5.0 || len2 < 5.0) {
        return CornerStatus::FAIL_SHORT_EDGE;
    }

    int white_count = 0, black_count = 0;
    const int num_circle_samples = 36;
    for (int i = 0; i < num_circle_samples; ++i) {
        double angle_rad = (2.0 * CV_PI * i) / num_circle_samples;
        int sample_x = static_cast<int>(p_curr.x + sample_radius * std::cos(angle_rad));
        int sample_y = static_cast<int>(p_curr.y + sample_radius * std::sin(angle_rad));

        if (sample_y >= 0 && sample_y < h && sample_x >= 0 && sample_x < w) {
            if (binary_image.at<uchar>(sample_y, sample_x) == 255) {
                white_count++;
            } else {
                black_count++;
            }
        }
    }

    if (white_count <= 1) {
        return CornerStatus::FAIL_INTERNAL;
    }

    double ratio = (white_count > 0) ? static_cast<double>(black_count) / white_count : 1e9;
    return (ratio >= ratio_threshold) ? CornerStatus::VALID : CornerStatus::FAIL_RATIO;
}

static EdgeCheckResult check_edge_type(const cv::Point& c1, const cv::Point& c2, const cv::Mat& image, int offset) {
    int h = image.rows, w = image.cols;
    cv::Point2f v = c2 - c1;
    double dist = cv::norm(v);
    if (dist < 10) return {EdgeType::FAIL_SHORT, {0, 0}};

    cv::Point2f perp_v = {-v.y, v.x};
    cv::Point2f perp_v_norm = perp_v / (dist + 1e-6);

    std::vector<std::string> side_colors;
    const int num_samples = 10;
    const double threshold = 0.9;

    for (int direction : {1, -1}) {
        int color_sum = 0;
        int valid_samples = 0;

        for (int i = 0; i < num_samples; ++i) {
            double alpha = 0.1 + i * (0.8 / (num_samples - 1));
            cv::Point2f p_on_line_f = cv::Point2f(c1) * (1.0 - alpha) + cv::Point2f(c2) * alpha;
            cv::Point2f p_shifted_f = p_on_line_f + direction * offset * perp_v_norm;
            cv::Point p_shifted(cvRound(p_shifted_f.x), cvRound(p_shifted_f.y));
            if (p_shifted.y >= 0 && p_shifted.y < h && p_shifted.x >= 0 && p_shifted.x < w) {
                color_sum += image.at<uchar>(p_shifted.y, p_shifted.x);
                valid_samples++;
            }
        }
        
        if (valid_samples == 0) {
            side_colors.push_back("FAIL_OOB");
            continue;
        }

        double white_ratio = (color_sum / 255.0) / valid_samples;
        if (white_ratio >= threshold) side_colors.push_back("WHITE");
        else if (white_ratio <= (1.0 - threshold)) side_colors.push_back("BLACK");
        else side_colors.push_back("MIXED");
    }

    if (side_colors[0] == "BLACK" && side_colors[1] == "WHITE") return {EdgeType::EXPOSED, -perp_v_norm};
    if (side_colors[0] == "WHITE" && side_colors[1] == "BLACK") return {EdgeType::EXPOSED, perp_v_norm};
    if (side_colors[0] == "BLACK" && side_colors[1] == "BLACK") return {EdgeType::INTERNAL, {0,0}};
    if (side_colors[0] == "MIXED" || side_colors[1] == "MIXED") return {EdgeType::FAIL_MIXED_SIDE, {0,0}};
    if (side_colors[0] == side_colors[1]) return {EdgeType::FAIL_SAME_COLOR, {0,0}};
    return {EdgeType::FAIL_UNKNOWN, {0,0}};
}

static bool validate_square_by_area_sampling(
    const std::vector<cv::Point>& square, const cv::Mat& image, 
    int sample_points, double white_threshold, 
    int morph_close_kernel_size) {

    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8U);
    cv::fillConvexPoly(mask, square, 255);

    cv::Mat processed_image;
    if (morph_close_kernel_size >= 2) {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,
                                                   cv::Size(morph_close_kernel_size, morph_close_kernel_size));
        cv::Mat temp_image = cv::Mat::zeros(image.size(), CV_8U);
        image.copyTo(temp_image, mask);
        cv::morphologyEx(temp_image, processed_image, cv::MORPH_CLOSE, kernel);
    } else {
        processed_image = image;
    }
    // -----------------------------------------

    cv::Mat inner_mask_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::erode(mask, mask, inner_mask_kernel);
    
    std::vector<cv::Point> locations;
    cv::findNonZero(mask, locations);
    if (locations.size() < static_cast<size_t>(sample_points)) {
        return false;
    }

    unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::shuffle(locations.begin(), locations.end(), std::mt19937(seed));
    
    int white_count = 0;
    for (int i = 0; i < sample_points; ++i) {
        if (processed_image.at<uchar>(locations[i]) == 255) {
            white_count++;
        }
    }
    
    return (static_cast<double>(white_count) / sample_points) >= white_threshold;
}

static bool are_squares_similar(const std::vector<cv::Point>& sq1, const std::vector<cv::Point>& sq2, double center_thresh, double area_thresh) {
    cv::Moments m1 = cv::moments(sq1, true);
    cv::Point2f center1(m1.m10 / m1.m00, m1.m01 / m1.m00);
    cv::Moments m2 = cv::moments(sq2, true);
    cv::Point2f center2(m2.m10 / m2.m00, m2.m01 / m2.m00);

    if (cv::norm(center1 - center2) > center_thresh) return false;

    double area1 = cv::contourArea(sq1);
    double area2 = cv::contourArea(sq2);

    if (std::max(area1, area2) > 1 && std::abs(area1 - area2) / std::max(area1, area2) > area_thresh) return false;

    return true;
}


std::vector<ImageBuffer::Square> ImageBuffer::find_squares(
    const std::tuple<int, int, int, int>& roi_tuple, int threshold_val, int min_area, double approx_epsilon,
    int corner_sample_radius, double corner_ratio_thresh, int edge_check_offset,
    int area_sample_points, double area_white_thresh,
    int area_morph_close_kernel_size,
    double duplicate_center_thresh, double duplicate_area_thresh) const {

    std::vector<Square> final_squares_list;
    if (!is_valid()) return final_squares_list;

    ImageBuffer roi_img_buf_owner;
    const ImageBuffer* processing_buf = this;
    int roi_x_offset = 0, roi_y_offset = 0;

    bool use_roi_flag = (std::get<2>(roi_tuple) > 0 && std::get<3>(roi_tuple) > 0);
    if (use_roi_flag) {
        try {
            roi_img_buf_owner = this->crop(roi_tuple);
            processing_buf = &roi_img_buf_owner;
            roi_x_offset = std::get<0>(roi_tuple);
            roi_y_offset = std::get<1>(roi_tuple);
        } catch (const std::exception& e) {
            throw std::runtime_error("find_squares: ROI crop failed: " + std::string(e.what()));
        }
    }

    cv::Mat gray_mat = image_buffer_to_gray_mat_view(*processing_buf);
    if (gray_mat.empty()) return final_squares_list;

    cv::Mat binary_mat;
    cv::threshold(gray_mat, binary_mat, threshold_val, 255, cv::THRESH_BINARY_INV);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary_mat, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<std::vector<cv::Point>> found_squares_cv;

    for (const auto& contour : contours) {
        if (cv::contourArea(contour) < min_area) continue;

        std::vector<cv::Point> approx_vertices;
        cv::approxPolyDP(contour, approx_vertices, cv::arcLength(contour, true) * approx_epsilon, true);

        if (approx_vertices.size() < 3) continue;

        std::vector<cv::Point> valid_corners;
        for (size_t i = 0; i < approx_vertices.size(); ++i) {
            const auto& p_curr = approx_vertices[i];
            const auto& p_prev = approx_vertices[(i + approx_vertices.size() - 1) % approx_vertices.size()];
            const auto& p_next = approx_vertices[(i + 1) % approx_vertices.size()];

            if (get_corner_status(p_curr, p_prev, p_next, binary_mat, corner_sample_radius, corner_ratio_thresh) == CornerStatus::VALID) {
                valid_corners.push_back(p_curr);
            }
        }

        if (valid_corners.size() < 2) continue;

        for (size_t i = 0; i < valid_corners.size(); ++i) {
            for (size_t j = i + 1; j < valid_corners.size(); ++j) {
                const auto& c1 = valid_corners[i];
                const auto& c2 = valid_corners[j];

                EdgeCheckResult edge_res = check_edge_type(c1, c2, binary_mat, edge_check_offset);

                if (edge_res.type == EdgeType::EXPOSED) {
                    cv::Point2f edge_v = c2 - c1;
                    cv::Point2f side_v = edge_res.inward_vector * cv::norm(edge_v);

                    std::vector<cv::Point> new_square = {
                        c1, c2,
                        cv::Point(c2.x + side_v.x, c2.y + side_v.y),
                        cv::Point(c1.x + side_v.x, c1.y + side_v.y)
                    };

                    if (validate_square_by_area_sampling(new_square, binary_mat, area_sample_points, area_white_thresh, area_morph_close_kernel_size)) {
                        bool is_duplicate = false;
                        for (const auto& existing_sq : found_squares_cv) {
                            if (are_squares_similar(new_square, existing_sq, duplicate_center_thresh, duplicate_area_thresh)) {
                                is_duplicate = true;
                                break;
                            }
                        }
                        if (!is_duplicate) {
                            found_squares_cv.push_back(new_square);
                        }
                    }
                }
            }
        }
    }

    for (const auto& sq_cv : found_squares_cv) {
        Square final_square;
        for (const auto& pt : sq_cv) {
            final_square.emplace_back(pt.x + roi_x_offset, pt.y + roi_y_offset);
        }
        final_squares_list.push_back(final_square);
    }

    return final_squares_list;
}

