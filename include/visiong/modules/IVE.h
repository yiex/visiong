// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_MODULES_IVE_H
#define VISIONG_MODULES_IVE_H

#include "visiong/core/ImageBuffer.h"
#include "rk_mpi_ive.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>


// Persistent memory block for stateful IVE algorithms such as GMM/GMM2. / 供 GMM/GMM2 等有状态 IVE 算法使用的持久内存块。
class IVEModel {
public:
    IVEModel(int width, int height, int model_size = 0);
    ~IVEModel();

    IVE_MEM_INFO_S mem_info;

private:
    IVEModel(const IVEModel&) = delete;
    IVEModel& operator=(const IVEModel&) = delete;
};

// Raw LK optical flow output for one feature point. / 单个特征点的原始 LK 光流输出。
struct IVEMotionVector {
    int status = -1;   // 0: valid, negative: invalid/failure
    short mv_x = 0;    // S9.7 fixed-point x displacement
    short mv_y = 0;    // S9.7 fixed-point y displacement
};

// Singleton wrapper for Rockchip IVE hardware operators. / Rockchip IVE 硬件算子的单例封装。
class IVE {
public:
    static IVE& get_instance();
    ~IVE();

    static void set_log_enabled(bool enabled);
    static bool is_log_enabled();

    ImageBuffer filter(const ImageBuffer& src, const std::vector<int8_t>& mask);
    std::tuple<ImageBuffer, ImageBuffer> sobel(
        const ImageBuffer& src,
        IVE_SOBEL_OUT_CTRL_E out_ctrl,
        IVE_IMAGE_TYPE_E out_format = IVE_IMAGE_TYPE_S16C1);
    ImageBuffer canny(const ImageBuffer& src, uint16_t high_thresh, uint16_t low_thresh);
    ImageBuffer mag_and_ang(const ImageBuffer& src, uint16_t threshold = 0, bool return_magnitude = true);
    ImageBuffer dilate(const ImageBuffer& src, int kernel_size = 3);
    ImageBuffer erode(const ImageBuffer& src, int kernel_size = 3);
    ImageBuffer ordered_stat_filter(const ImageBuffer& src, IVE_ORD_STAT_FILTER_MODE_E mode);
    ImageBuffer add(const ImageBuffer& src1, const ImageBuffer& src2);
    ImageBuffer sub(const ImageBuffer& src1, const ImageBuffer& src2, IVE_SUB_MODE_E mode = IVE_SUB_MODE_ABS);
    ImageBuffer logic_op(const ImageBuffer& src1, const ImageBuffer& src2, IVE_LOGICOP_MODE_E op);
    ImageBuffer threshold(
        const ImageBuffer& src,
        uint8_t low_thresh,
        uint8_t high_thresh = 255,
        IVE_THRESH_MODE_E mode = IVE_THRESH_MODE_BINARY);
    ImageBuffer cast_16bit_to_8bit(const ImageBuffer& src, IVE_16BIT_TO_8BIT_MODE_E mode);
    std::vector<uint32_t> hist(const ImageBuffer& src);
    ImageBuffer equalize_hist(const ImageBuffer& src);
    ImageBuffer integral(const ImageBuffer& src, IVE_INTEG_OUT_CTRL_E mode = IVE_INTEG_OUT_CTRL_COMBINE);
    std::vector<Blob> ccl(const ImageBuffer& src, int min_area = 100);
    double ncc(const ImageBuffer& src1, const ImageBuffer& src2);
    ImageBuffer csc(const ImageBuffer& src, IVE_CSC_MODE_E mode);

    // Color-space conversion helpers. / 色彩空间转换辅助函数。
    ImageBuffer yuv_to_rgb(const ImageBuffer& src, bool full_range = true);
    ImageBuffer yuv_to_hsv(const ImageBuffer& src, bool full_range = true);
    ImageBuffer rgb_to_yuv(const ImageBuffer& src, bool full_range = true);
    ImageBuffer rgb_to_hsv(const ImageBuffer& src, bool full_range = true);

    // Extended operators. / 扩展算子。
    ImageBuffer dma(const ImageBuffer& src, IVE_DMA_MODE_E mode = IVE_DMA_MODE_DIRECT_COPY);
    ImageBuffer cast_8bit_to_8bit(const ImageBuffer& src, int8_t bias, uint8_t numerator, uint8_t denominator);
    ImageBuffer map(const ImageBuffer& src, const std::vector<uint8_t>& lut);
    std::tuple<ImageBuffer, ImageBuffer> gmm(const ImageBuffer& src, IVEModel& model, bool first_frame = false);
    std::tuple<ImageBuffer, ImageBuffer> gmm2(
        const ImageBuffer& src,
        const ImageBuffer& factor,
        IVEModel& model,
        bool first_frame = false);
    ImageBuffer lbp(const ImageBuffer& src, bool abs_mode = false, int8_t threshold = 0);
    std::tuple<ImageBuffer, ImageBuffer> norm_grad(const ImageBuffer& src);
    std::vector<IVEMotionVector> lk_optical_flow(
        const ImageBuffer& prev_img,
        const ImageBuffer& next_img,
        const std::vector<std::tuple<int, int>>& points);
    std::vector<std::tuple<int, int>> st_corner(
        const ImageBuffer& src,
        uint16_t max_corners = 200,
        uint16_t min_dist = 10,
        uint8_t quality_level = 25);
    ImageBuffer match_bg_model(const ImageBuffer& current_img, IVEModel& bg_model, uint32_t frame_num);
    ImageBuffer update_bg_model(
        const ImageBuffer& current_img,
        const ImageBuffer& fg_flag,
        IVEModel& bg_model,
        uint32_t frame_num);
    std::tuple<ImageBuffer, ImageBuffer> sad(
        const ImageBuffer& src1,
        const ImageBuffer& src2,
        IVE_SAD_MODE_E mode,
        uint16_t threshold,
        uint8_t min_val = 0,
        uint8_t max_val = 255);
    std::vector<ImageBuffer> create_pyramid(const ImageBuffer& src, int levels);

private:
    IVE();
    IVE(const IVE&) = delete;
    IVE& operator=(const IVE&) = delete;

    void ensure_buffers(int width, int height);
    void release_buffers();
    void create_and_check_buffer(
        std::unique_ptr<IVE_IMAGE_S>& img,
        IVE_IMAGE_TYPE_E type,
        int width,
        int height,
        const std::string& name);
    void copy_to_ive_buffer(const ImageBuffer& src, IVE_IMAGE_S* dst);
    ImageBuffer copy_from_ive_buffer(const IVE_IMAGE_S* src, int expected_w = 0, int expected_h = 0);
    ImageBuffer morph_op(const ImageBuffer& src, bool is_dilate, int kernel_size);

    std::mutex m_mutex;
    int m_width = 0;
    int m_height = 0;

    // Reused image buffers. / 复用的图像缓冲区。
    std::unique_ptr<IVE_IMAGE_S> m_img_u8c1_in1;
    std::unique_ptr<IVE_IMAGE_S> m_img_u8c1_in2;
    std::unique_ptr<IVE_IMAGE_S> m_img_u8c1_out1;
    std::unique_ptr<IVE_IMAGE_S> m_img_u8c1_out2;
    std::unique_ptr<IVE_IMAGE_S> m_img_s16c1_out1;
    std::unique_ptr<IVE_IMAGE_S> m_img_s16c1_out2;
    std::unique_ptr<IVE_IMAGE_S> m_img_u16c1_in1;
    std::unique_ptr<IVE_IMAGE_S> m_img_u64c1_out1;

    // Reused workspace / metadata buffers. / 复用的工作区 / 元数据缓冲区。
    std::unique_ptr<IVE_MEM_INFO_S> m_mem_info;
    std::unique_ptr<IVE_MEM_INFO_S> m_lk_pts_mem;
    std::unique_ptr<IVE_MEM_INFO_S> m_lk_mv_mem;
    std::unique_ptr<IVE_MEM_INFO_S> m_st_candi_corner_mem;
    std::unique_ptr<IVE_MEM_INFO_S> m_st_corner_info_mem;
    std::unique_ptr<IVE_MEM_INFO_S> m_st_ctrl_mem;
};

#endif  // VISIONG_MODULES_IVE_H

