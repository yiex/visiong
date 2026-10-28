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

    int width() const noexcept { return m_width; }
    int height() const noexcept { return m_height; }
    void reset();

    IVE_MEM_INFO_S mem_info{};

private:
    friend class IVE;
    enum class Algorithm { Uninitialized, Gmm, Gmm2, Codebook };
    void validate_dimensions(int width, int height) const;
    void clear_locked();
    void prepare_gmm_locked(bool separate_output);

    const int m_width;
    const int m_height;
    Algorithm m_algorithm = Algorithm::Uninitialized;
    int m_gmm_models = 0;
    IVE_IMAGE_S m_gmm_matched{};
    IVE_MEM_INFO_S m_gmm_output{};
    std::mutex m_mutex;

    IVEModel(const IVEModel&) = delete;
    IVEModel& operator=(const IVEModel&) = delete;
};

// Raw LK optical flow output for one feature point. / 单个特征点的原始 LK 光流输出。
struct IVEMotionVector {
    int status = -1;   // 0: valid, -1: failed. / 0 表示有效，-1 表示跟踪失败。
    short mv_x = 0;    // S9.7 x displacement; divide by 128. / S9.7 横向位移，除以 128 得到像素。
    short mv_y = 0;    // S9.7 y displacement; divide by 128. / S9.7 纵向位移，除以 128 得到像素。
};

// Singleton wrapper for Rockchip IVE hardware operators. / Rockchip IVE 硬件算子的单例封装。
class IVE {
public:
    static IVE& get_instance();
    ~IVE();

    static void set_log_enabled(bool enabled);
    static bool is_log_enabled();

    ImageBuffer filter(const ImageBuffer& src, const std::vector<int8_t>& mask);
    ImageBuffer filter_ex(const ImageBuffer& src, const std::vector<int8_t>& mask,
                          int norm = 0, IVE_IMAGE_TYPE_E out_format = IVE_IMAGE_TYPE_U8C1);
    std::tuple<ImageBuffer, ImageBuffer> sobel_ex(
        const ImageBuffer& src, const std::vector<int8_t>& mask, int norm = 0,
        IVE_SOBEL_OUT_CTRL_E out_ctrl = IVE_SOBEL_OUT_CTRL_BOTH,
        IVE_IMAGE_TYPE_E out_format = IVE_IMAGE_TYPE_S16C1);
    std::tuple<ImageBuffer, ImageBuffer> sobel(
        const ImageBuffer& src,
        IVE_SOBEL_OUT_CTRL_E out_ctrl,
        IVE_IMAGE_TYPE_E out_format = IVE_IMAGE_TYPE_S16C1);
    ImageBuffer canny(const ImageBuffer& src, uint16_t high_thresh, uint16_t low_thresh);
    ImageBuffer canny_ex(const ImageBuffer& src, uint16_t high_thresh, uint16_t low_thresh,
                          const std::vector<int8_t>& mask = {});
    ImageBuffer mag_and_ang(const ImageBuffer& src, uint16_t threshold = 0, bool return_magnitude = true);
    std::tuple<ImageBuffer, ImageBuffer> mag_and_ang_ex(
        const ImageBuffer& src, uint16_t threshold, const std::vector<int8_t>& mask);
    ImageBuffer morphology(const ImageBuffer& src, const std::vector<uint8_t>& mask, bool dilate_op);
    ImageBuffer dilate(const ImageBuffer& src, int kernel_size = 3);
    ImageBuffer erode(const ImageBuffer& src, int kernel_size = 3);
    ImageBuffer ordered_stat_filter(const ImageBuffer& src, IVE_ORD_STAT_FILTER_MODE_E mode);
    ImageBuffer add(const ImageBuffer& src1, const ImageBuffer& src2);
    ImageBuffer add_weighted(const ImageBuffer& src1, const ImageBuffer& src2,
                             int weight1 = 32768, int weight2 = 32768);
    ImageBuffer sub(const ImageBuffer& src1, const ImageBuffer& src2, IVE_SUB_MODE_E mode = IVE_SUB_MODE_ABS);
    ImageBuffer logic_op(const ImageBuffer& src1, const ImageBuffer& src2, IVE_LOGICOP_MODE_E op);
    ImageBuffer threshold(
        const ImageBuffer& src,
        uint8_t low_thresh,
        uint8_t high_thresh = 255,
        IVE_THRESH_MODE_E mode = IVE_THRESH_MODE_BINARY);
    // Complete threshold controls; output constants are checked before narrowing. / 完整阈值控制；窄化前检查输出常量。
    ImageBuffer threshold_u8(
        const ImageBuffer& src,
        int low_thresh,
        int high_thresh = 255,
        IVE_THRESH_MODE_E mode = IVE_THRESH_MODE_BINARY,
        int min_val = 0,
        int mid_val = 0,
        int max_val = 255);
    ImageBuffer threshold_s16(
        const ImageBuffer& src,
        int low_thresh,
        int high_thresh,
        IVE_THRESH_S16_MODE_E mode = IVE_THRESH_S16_MODE_S16_TO_U8_MIN_MID_MAX,
        int min_val = 0,
        int mid_val = 0,
        int max_val = 255);
    ImageBuffer threshold_u16(
        const ImageBuffer& src,
        int low_thresh,
        int high_thresh,
        IVE_THRESH_U16_MODE_E mode = IVE_THRESH_U16_MODE_U16_TO_U8_MIN_MID_MAX,
        int min_val = 0,
        int mid_val = 0,
        int max_val = 255);
    ImageBuffer cast_16bit_to_8bit(const ImageBuffer& src, IVE_16BIT_TO_8BIT_MODE_E mode);
    std::vector<uint32_t> hist(const ImageBuffer& src);
    ImageBuffer equalize_hist(const ImageBuffer& src);
    std::tuple<ImageBuffer, std::vector<uint32_t>> equalize_hist_ex(
        const ImageBuffer& src, IVE_EQUALIZE_MODE_E mode = IVE_EQUALIZE_MODE_EQHIST,
        const std::vector<uint32_t>& histogram = {});
    ImageBuffer integral(const ImageBuffer& src, IVE_INTEG_OUT_CTRL_E mode = IVE_INTEG_OUT_CTRL_COMBINE);
    std::vector<Blob> ccl(const ImageBuffer& src, int min_area = 100);
    std::vector<Blob> ccl_config(const ImageBuffer& src, int min_area, IVE_CCL_MODE_E mode,
                                 uint16_t init_area_threshold, uint16_t step);
    double ncc(const ImageBuffer& src1, const ImageBuffer& src2);
    std::tuple<uint64_t, uint64_t, uint64_t> ncc_raw(const ImageBuffer& src1, const ImageBuffer& src2);
    ImageBuffer csc(const ImageBuffer& src, IVE_CSC_MODE_E mode, bool input_full_range = false);
    // Compact byte layouts; planar and 422 layouts use CPU layout bridges around one hardware CSC.
    // 紧凑字节布局；平面与 422 布局通过 CPU 布局桥接配合一次硬件 CSC。
    std::vector<uint8_t> csc_bytes(
        std::vector<uint8_t> data, int width, int height, IVE_IMAGE_TYPE_E src_type,
        IVE_IMAGE_TYPE_E dst_type, IVE_CSC_MODE_E mode, bool input_full_range = false,
        bool bgr_order = false, bool vu_order = false);

    // Color-space conversion helpers. / 色彩空间转换辅助函数。
    ImageBuffer yuv_to_rgb(const ImageBuffer& src, bool full_range = true, bool input_full_range = false);
    ImageBuffer yuv_to_hsv(const ImageBuffer& src, bool full_range = true, bool input_full_range = false);
    ImageBuffer rgb_to_yuv(const ImageBuffer& src, bool full_range = true);
    ImageBuffer rgb_to_hsv(const ImageBuffer& src, bool full_range = true);

    // Extended operators. / 扩展算子。
    ImageBuffer dma(const ImageBuffer& src, IVE_DMA_MODE_E mode = IVE_DMA_MODE_DIRECT_COPY);
    // DMA widths and strides are byte counts; this interface accepts GRAY8 planes. / DMA 宽度与步长按字节计，此接口接收 GRAY8 平面。
    ImageBuffer dma_transfer(
        const ImageBuffer& src,
        IVE_DMA_MODE_E mode = IVE_DMA_MODE_DIRECT_COPY,
        uint64_t value = 0,
        int hor_seg_size = 4,
        int elem_size = 1,
        int ver_seg_rows = 1,
        int output_width = 0,
        int output_height = 0);
    ImageBuffer cast_8bit_to_8bit(const ImageBuffer& src, int8_t bias, uint8_t numerator, uint8_t denominator);
    ImageBuffer map(const ImageBuffer& src, const std::vector<uint8_t>& lut);
    ImageBuffer map_typed(const ImageBuffer& src, const std::vector<int>& lut,
                          IVE_MAP_MODE_E mode = IVE_MAP_MODE_U8);
    ImageBuffer cast_8bit_ex(const ImageBuffer& src, IVE_8BIT_TO_8BIT_MODE_E mode,
                            int numerator = 1, int denominator = 1, int bias = 0);
    ImageBuffer cast_16bit_ex(const ImageBuffer& src, IVE_16BIT_TO_8BIT_MODE_E mode,
                             int numerator = 1, int denominator = 1, int bias = 0);
    // Single-model hardware GMM; initialize each sequence with first_frame=true. / 单模型硬件 GMM；每段序列首帧需设为 true。
    std::tuple<ImageBuffer, ImageBuffer> gmm(const ImageBuffer& src, IVEModel& model, bool first_frame = false);
    std::tuple<ImageBuffer, ImageBuffer> gmm2(
        const ImageBuffer& src,
        const ImageBuffer& factor,
        IVEModel& model,
        bool first_frame = false);
    // Persistent multi-model state; returns foreground, background and matched records.
    // 独立持久的多模型状态；返回前景、背景和匹配记录图。
    std::tuple<ImageBuffer, ImageBuffer, ImageBuffer> gmm_config(
        const ImageBuffer& src, IVEModel& model, int max_models = 3, bool first_frame = false);
    std::tuple<ImageBuffer, ImageBuffer, ImageBuffer> gmm2_config(
        const ImageBuffer& src, const ImageBuffer& factor, IVEModel& model,
        int max_models = 3, bool first_frame = false);
    ImageBuffer lbp(const ImageBuffer& src, bool abs_mode = false, int threshold = 8);
    std::tuple<ImageBuffer, ImageBuffer> norm_grad(const ImageBuffer& src);
    std::tuple<ImageBuffer, ImageBuffer, ImageBuffer> norm_grad_config(
        const ImageBuffer& src, IVE_NORM_GRAD_OUT_CTRL_E out_ctrl = IVE_NORM_GRAD_OUT_CTRL_HOR_AND_VER,
        uint8_t norm = 7, const std::vector<int8_t>& mask = {});
    std::vector<IVEMotionVector> lk_optical_flow(
        const ImageBuffer& prev_img,
        const ImageBuffer& next_img,
        const std::vector<std::tuple<int, int>>& points);
    // Additional hardware controls are validated independently from the default helpers. / 扩展硬件控制项独立于默认辅助接口进行验证。
    std::vector<IVEMotionVector> lk_optical_flow_single(
        const ImageBuffer& prev_img, const ImageBuffer& next_img, const std::vector<std::tuple<int, int>>& points,
        uint8_t iterations = 20, uint8_t min_eigenvalue = 100, uint16_t epsilon = 32);
    std::tuple<std::vector<IVEMotionVector>, std::vector<uint16_t>> lk_optical_flow_config(
        const ImageBuffer& prev_img, const ImageBuffer& next_img, const std::vector<std::tuple<int, int>>& points,
        int max_level = 3, int iterations = 20, int min_eigenvalue = 100, int epsilon = 32,
        const std::vector<std::tuple<int, int>>& initial_points = {});
    std::vector<std::tuple<int, int>> st_corner(
        const ImageBuffer& src,
        uint16_t max_corners = 200,
        uint16_t min_dist = 10,
        uint8_t quality_level = 25);
    std::vector<std::tuple<int, int, int>> st_corner_candidates(const ImageBuffer& src, uint8_t quality_level = 25);
    std::vector<std::tuple<int, int, int>> st_corner_sorted(
        const ImageBuffer& src, uint16_t max_corners = 200, uint16_t min_dist = 10, uint8_t quality_level = 25);
    void match_bg_model(const ImageBuffer& current_img, IVEModel& bg_model, uint32_t frame_num);
    std::tuple<ImageBuffer, ImageBuffer> update_bg_model(
        const ImageBuffer& current_img,
        IVEModel& bg_model,
        uint32_t frame_num);
    std::tuple<ImageBuffer, ImageBuffer> sad(
        const ImageBuffer& src1,
        const ImageBuffer& src2,
        IVE_SAD_MODE_E mode,
        uint16_t threshold,
        uint8_t min_val = 0,
        uint8_t max_val = 255);
    std::tuple<ImageBuffer, ImageBuffer> sad_config(
        const ImageBuffer& src1, const ImageBuffer& src2, IVE_SAD_MODE_E mode, uint16_t threshold,
        uint8_t min_val = 0, uint8_t max_val = 255, IVE_SAD_OUT_MODE_E out_mode = IVE_SAD_OUT_MODE_BOTH,
        IVE_SAD_OUT_BITS_E out_bits = IVE_SAD_OUT_16_BITS);
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
    std::tuple<ImageBuffer, ImageBuffer, ImageBuffer> gmm_execute(
        const ImageBuffer& src, const ImageBuffer* factor, IVEModel& model,
        int max_models, bool first_frame, bool return_matched);

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
    std::unique_ptr<IVE_MEM_INFO_S> m_lk_next_pts_mem;
    std::unique_ptr<IVE_MEM_INFO_S> m_lk_status_mem;
    std::unique_ptr<IVE_MEM_INFO_S> m_lk_error_mem;
    std::unique_ptr<IVE_MEM_INFO_S> m_lk_pyramid_mem;
    std::unique_ptr<IVE_MEM_INFO_S> m_st_candi_corner_mem;
    std::unique_ptr<IVE_MEM_INFO_S> m_st_corner_info_mem;
    std::unique_ptr<IVE_MEM_INFO_S> m_st_ctrl_mem;
};

#endif  // VISIONG_MODULES_IVE_H

