// SPDX-License-Identifier: LGPL-3.0-or-later
#include "visiong/modules/IVE.h"
#include "visiong/common/pixel_format.h"

#include "modules/internal/ive_memory.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace {
using visiong::ive_internal::calc_stride;
using visiong::ive_internal::create_image;
using visiong::ive_internal::create_mem_info;
using visiong::ive_internal::free_mmz;
using visiong::ive_internal::kImageAlign;
using visiong::ive_internal::mmz_alloc;
using visiong::ive_internal::mmz_flush_end;
using visiong::ive_internal::mmz_flush_start;
using visiong::ive_internal::mmz_free;

#ifndef RK_FMT_U16C1
#define RK_FMT_U16C1 static_cast<PIXEL_FORMAT_E>(0x1002)
#endif

struct IveMmzGuard {
    RK_U64 phy = 0;
    RK_U64 vir = 0;

    ~IveMmzGuard() {
        if (vir) free_mmz(phy, vir);
    }

    void set(RK_U64 p, RK_U64 v) {
        phy = p;
        vir = v;
    }
};

constexpr int kLkInputPointShift = 2;

void flush_mmz_cache_by_vir_addr(RK_U64 vir_addr, bool cpu_to_device) {
    MB_BLK blk = RK_MPI_MB_VirAddr2Handle(reinterpret_cast<void*>(vir_addr));
    if (blk != MB_INVALID_HANDLE) {
        RK_MPI_SYS_MmzFlushCache(blk, cpu_to_device ? RK_TRUE : RK_FALSE);
    }
}

const ImageBuffer& ensure_gray_input(const ImageBuffer& src, std::unique_ptr<ImageBuffer>& holder) {
    if (src.format == visiong::kGray8Format) {
        return src;
    }
    holder = std::make_unique<ImageBuffer>(src.to_grayscale());
    return *holder;
}

} // namespace

std::tuple<ImageBuffer, ImageBuffer> IVE::gmm(const ImageBuffer& src, IVEModel& model, bool first_frame) {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_buffers(src.width, src.height);
    if (src.format != visiong::kGray8Format) throw std::invalid_argument("GMM requires a GRAY8 source image.");
    copy_to_ive_buffer(src, m_img_u8c1_in1.get());
    IVE_GMM_CTRL_S stCtrl; memset(&stCtrl, 0, sizeof(stCtrl));
    stCtrl.u8FirstFrameFlag = first_frame ? 1 : 0;
    stCtrl.u8MaxModelNum = 3; stCtrl.u10q0InitVar = 300; stCtrl.u10q0MinVar = 50; stCtrl.u8q2WeightInitVal = 5; stCtrl.u8VarThreshGen = 4;
    IVE_HANDLE handle;
    RK_S32 s32Ret = RK_MPI_IVE_GMM(&handle, m_img_u8c1_in1.get(), m_img_u8c1_out1.get(), m_img_u8c1_out2.get(), nullptr, &model.mem_info, &stCtrl, RK_TRUE);
    if (s32Ret != RK_SUCCESS) throw std::runtime_error("RK_MPI_IVE_GMM failed with error code: " + std::to_string(s32Ret));
    return std::make_tuple(copy_from_ive_buffer(m_img_u8c1_out1.get(), m_width, m_height), copy_from_ive_buffer(m_img_u8c1_out2.get(), m_width, m_height));
}

std::tuple<ImageBuffer, ImageBuffer> IVE::gmm2(const ImageBuffer& src, const ImageBuffer& factor, IVEModel& model, bool first_frame) {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_buffers(src.width, src.height);
    if (src.format != visiong::kGray8Format || factor.format != RK_FMT_U16C1) throw std::invalid_argument("GMM2 requires GRAY8 source and U16C1 factor images.");
    copy_to_ive_buffer(src, m_img_u8c1_in1.get());
    copy_to_ive_buffer(factor, m_img_u16c1_in1.get());
    IVE_GMM2_CTRL_S stCtrl; memset(&stCtrl, 0, sizeof(stCtrl));
    stCtrl.u8FirstFrameFlag = first_frame ? 1 : 0;
    stCtrl.u8MaxModelNum = 3; stCtrl.u10q0InitVar = 300; stCtrl.u10q0MinVar = 50; stCtrl.u8q2WeightInitVal = 5; stCtrl.u8VarThreshGen = 4;
    IVE_HANDLE handle;
    RK_S32 s32Ret = RK_MPI_IVE_GMM2(&handle, m_img_u8c1_in1.get(), (IVE_SRC_IMAGE_S*)m_img_u16c1_in1.get(), m_img_u8c1_out1.get(), m_img_u8c1_out2.get(), nullptr, &model.mem_info, &stCtrl, RK_TRUE);
    if (s32Ret != RK_SUCCESS) throw std::runtime_error("RK_MPI_IVE_GMM2 failed with error code: " + std::to_string(s32Ret));
    return std::make_tuple(copy_from_ive_buffer(m_img_u8c1_out1.get(), m_width, m_height), copy_from_ive_buffer(m_img_u8c1_out2.get(), m_width, m_height));
}

ImageBuffer IVE::lbp(const ImageBuffer& src, bool abs_mode, int8_t threshold) {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_buffers(src.width, src.height);
    if (src.format != visiong::kGray8Format) throw std::invalid_argument("LBP requires a GRAY8 source image.");
    copy_to_ive_buffer(src, m_img_u8c1_in1.get());
    
    IVE_LBP_CTRL_S stCtrl; 
    memset(&stCtrl, 0, sizeof(stCtrl)); 
    stCtrl.enMode = abs_mode ? IVE_LBP_CMP_MODE_ABS : IVE_LBP_CMP_MODE_NORMAL;
    
    // Follow vendor recommendation: default threshold = 8 when caller passes 0. / 遵循厂商建议：当调用方传入 0 时，默认阈值设为 8。
    // NORMAL mode checks signed difference; ABS mode checks absolute difference.
    int8_t effective_threshold = (threshold == 0) ? 8 : threshold;
    
    if (abs_mode) {
        stCtrl.un8BitThr.u8Val = static_cast<uint8_t>(effective_threshold);
    } else {
        stCtrl.un8BitThr.s8Val = effective_threshold;
    }
    
    IVE_HANDLE handle;
    if (RK_MPI_IVE_LBP(&handle, m_img_u8c1_in1.get(), m_img_u8c1_out1.get(), &stCtrl, RK_TRUE) != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_LBP failed.");
    }
    return copy_from_ive_buffer(m_img_u8c1_out1.get(), m_width, m_height);
}

std::tuple<ImageBuffer, ImageBuffer> IVE::norm_grad(const ImageBuffer& src) {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_buffers(src.width, src.height);
    if (src.format != visiong::kGray8Format) throw std::invalid_argument("NormGrad requires a GRAY8 source image.");
    copy_to_ive_buffer(src, m_img_u8c1_in1.get());
    IVE_NORM_GRAD_CTRL_S stCtrl;
    memset(&stCtrl, 0, sizeof(stCtrl)); 
    stCtrl.enOutCtrl = IVE_NORM_GRAD_OUT_CTRL_HOR_AND_VER; stCtrl.u8Norm = 7;
    const RK_S8 mask[25] = { -1, -2, 0, 2, 1, -4, -8, 0, 8, 4, -6, -12, 0, 12, 6, -4, -8, 0, 8, 4, -1, -2, 0, 2, 1 };
    memcpy(stCtrl.as8Mask, mask, sizeof(mask));
    IVE_HANDLE handle;
    if (RK_MPI_IVE_NormGrad(&handle, m_img_u8c1_in1.get(), (IVE_DST_IMAGE_S*)m_img_s16c1_out1.get(), (IVE_DST_IMAGE_S*)m_img_s16c1_out2.get(), nullptr, &stCtrl, RK_TRUE) != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_NormGrad failed.");
    }
    return std::make_tuple(copy_from_ive_buffer(m_img_s16c1_out1.get(), m_width, m_height), copy_from_ive_buffer(m_img_s16c1_out2.get(), m_width, m_height));
}

std::vector<IVEMotionVector> IVE::lk_optical_flow(const ImageBuffer& prev_img, const ImageBuffer& next_img, const std::vector<std::tuple<int, int>>& points) {
    std::lock_guard<std::mutex> lock(m_mutex);

    if (!prev_img.is_valid() || !next_img.is_valid()) return {};
    if (prev_img.width != next_img.width || prev_img.height != next_img.height) {
        throw std::invalid_argument("LKOpticalFlow: Image dimension mismatch.");
    }
    if (points.empty()) return {};
    if (points.size() > LK_OPTICAL_FLOW_MAX_POINT_NUM) {
        throw std::invalid_argument("LKOpticalFlow: Too many points (Max " + std::to_string(LK_OPTICAL_FLOW_MAX_POINT_NUM) + ").");
    }

    ensure_buffers(prev_img.width, prev_img.height);

    std::unique_ptr<ImageBuffer> prev_holder;
    std::unique_ptr<ImageBuffer> next_holder;
    const ImageBuffer& prev_gray = ensure_gray_input(prev_img, prev_holder);
    const ImageBuffer& next_gray = ensure_gray_input(next_img, next_holder);

    copy_to_ive_buffer(prev_gray, m_img_u8c1_in1.get());
    copy_to_ive_buffer(next_gray, m_img_u8c1_in2.get());

    // IVE expects U14.2 fixed-point coordinates. / IVE 期望 U14.2 定点坐标。
    IVE_POINT_U16_S* prev_pts = reinterpret_cast<IVE_POINT_U16_S*>(m_lk_pts_mem->u64VirAddr);
    for (size_t i = 0; i < points.size(); ++i) {
        prev_pts[i].u16X = static_cast<RK_U16>(std::get<0>(points[i]) << kLkInputPointShift);
        prev_pts[i].u16Y = static_cast<RK_U16>(std::get<1>(points[i]) << kLkInputPointShift);
    }

    flush_mmz_cache_by_vir_addr(m_lk_pts_mem->u64VirAddr, true);
    std::memset(reinterpret_cast<void*>(m_lk_mv_mem->u64VirAddr), 0, m_lk_mv_mem->u32Size);

    IVE_LK_OPTICAL_FLOW_CTRL_S stCtrl;
    std::memset(&stCtrl, 0, sizeof(stCtrl));
    stCtrl.u16PtsNum = static_cast<RK_U16>(points.size());
    stCtrl.u0q8MinEigThr = 100;
    stCtrl.u8IterCnt = 20;
    stCtrl.u0q11Eps = 32;

    IVE_HANDLE handle;
    const RK_S32 ret = RK_MPI_IVE_LKOpticalFlow(&handle, m_img_u8c1_in1.get(), m_img_u8c1_in2.get(),
                                          m_lk_pts_mem.get(), m_lk_mv_mem.get(), &stCtrl, RK_TRUE);

    if (ret != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_LKOpticalFlow failed: " + std::to_string(ret));
    }

    flush_mmz_cache_by_vir_addr(m_lk_mv_mem->u64VirAddr, false);

    std::vector<IVEMotionVector> results;
    results.reserve(points.size());

    // IVE_MV_S16_S matches LK output memory layout. / IVE_MV_S16_S 与 LK 输出内存布局一致。
    IVE_MV_S16_S* motion = reinterpret_cast<IVE_MV_S16_S*>(m_lk_mv_mem->u64VirAddr);
    for (size_t i = 0; i < points.size(); ++i) {
        results.push_back({
            static_cast<int>(motion[i].s32Statys),
            static_cast<short>(motion[i].s16X),
            static_cast<short>(motion[i].s16Y)
        });
    }

    return results;
}

std::vector<std::tuple<int, int>> IVE::st_corner(const ImageBuffer& src, uint16_t max_corners, uint16_t min_dist, uint8_t quality_level) {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_buffers(src.width, src.height);

    std::unique_ptr<ImageBuffer> src_holder;
    const ImageBuffer& gray = ensure_gray_input(src, src_holder);
    copy_to_ive_buffer(gray, m_img_u8c1_in1.get());
    std::memset(reinterpret_cast<void*>(m_st_candi_corner_mem->u64VirAddr), 0, m_st_candi_corner_mem->u32Size);
    std::memset(reinterpret_cast<void*>(m_st_corner_info_mem->u64VirAddr), 0, m_st_corner_info_mem->u32Size);

    IVE_HANDLE handle;
    RK_S32 ret = RK_SUCCESS;
    std::vector<std::tuple<int, int>> result_points;

    IVE_ST_CANDI_CORNER_CTRL_S stCandiCtrl;
    std::memset(&stCandiCtrl, 0, sizeof(stCandiCtrl));
    stCandiCtrl.u0q8QualityLevel = quality_level;

    ret = RK_MPI_IVE_STCandiCorner(&handle, m_img_u8c1_in1.get(), m_st_candi_corner_mem.get(), &stCandiCtrl, RK_TRUE);
    if (ret != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_STCandiCorner failed: " + std::to_string(ret));
    }

    // Ensure candidate corners are visible to the following hardware stage. / 确保候选角点对后续硬件阶段可见。
    flush_mmz_cache_by_vir_addr(m_st_candi_corner_mem->u64VirAddr, true);

    IVE_ST_CORNER_CTRL_S stCornerCtrl;
    std::memset(&stCornerCtrl, 0, sizeof(stCornerCtrl));
    stCornerCtrl.stMem = *m_st_ctrl_mem.get();
    stCornerCtrl.u16MaxCornerNum = std::min<uint16_t>(IVE_ST_MAX_CORNER_NUM, max_corners);
    stCornerCtrl.u16MinDist = min_dist;

    ret = RK_MPI_IVE_STCorner(&handle, m_img_u8c1_in1.get(), m_st_candi_corner_mem.get(), m_st_corner_info_mem.get(), &stCornerCtrl, RK_TRUE);
    if (ret != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_STCorner failed: " + std::to_string(ret));
    }

    flush_mmz_cache_by_vir_addr(m_st_corner_info_mem->u64VirAddr, false);

    const IVE_ST_CORNER_INFO_S* corner_info =
        reinterpret_cast<const IVE_ST_CORNER_INFO_S*>(m_st_corner_info_mem->u64VirAddr);
    const uint16_t valid_corners = std::min<uint16_t>(corner_info->u16CornerNum, IVE_ST_MAX_CORNER_NUM);

    result_points.reserve(valid_corners);
    for (uint16_t i = 0; i < valid_corners; ++i) {
        result_points.emplace_back(corner_info->astCorner[i].u16X, corner_info->astCorner[i].u16Y);
    }

    return result_points;
}

ImageBuffer IVE::match_bg_model(const ImageBuffer& current_img, IVEModel& bg_model, uint32_t frame_num) {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_buffers(current_img.width, current_img.height);
    copy_to_ive_buffer(current_img, m_img_u8c1_in1.get());
    IVE_DATA_S stBgModel;
    memset(&stBgModel, 0, sizeof(stBgModel));
    stBgModel.u64PhyAddr = bg_model.mem_info.u64PhyAddr;
    stBgModel.u64VirAddr = bg_model.mem_info.u64VirAddr;
    stBgModel.u32Stride = static_cast<RK_U32>(current_img.width);
    stBgModel.u32Width = static_cast<RK_U32>(current_img.width);
    stBgModel.u32Height = static_cast<RK_U32>(current_img.height);
    IVE_MATCH_BG_MODEL_CTRL_S stCtrl;
    memset(&stCtrl, 0, sizeof(stCtrl));
    stCtrl.u32CurFrmNum = frame_num; 
    stCtrl.u8DiffMaxThr = 10; 
    stCtrl.u8DiffMinThr = 10; 
    stCtrl.u8TrainTimeThr = 20;

    IVE_HANDLE handle;
    if (RK_MPI_IVE_MatchBgModel(&handle, m_img_u8c1_in1.get(), &stBgModel, &stCtrl, RK_TRUE) != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_MatchBgModel failed.");
    }
    return ImageBuffer();
}

ImageBuffer IVE::update_bg_model(const ImageBuffer& current_img, const ImageBuffer& fg_flag, IVEModel& bg_model, uint32_t frame_num) {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_buffers(current_img.width, current_img.height);
    copy_to_ive_buffer(current_img, m_img_u8c1_in1.get());
    copy_to_ive_buffer(fg_flag, m_img_u8c1_in2.get());
    IVE_DATA_S stBgModel;
    memset(&stBgModel, 0, sizeof(stBgModel));
    stBgModel.u64PhyAddr = bg_model.mem_info.u64PhyAddr;
    stBgModel.u64VirAddr = bg_model.mem_info.u64VirAddr;
    stBgModel.u32Stride = static_cast<RK_U32>(current_img.width);
    stBgModel.u32Width = static_cast<RK_U32>(current_img.width);
    stBgModel.u32Height = static_cast<RK_U32>(current_img.height);
    IVE_UPDATE_BG_MODEL_CTRL_S stCtrl; memset(&stCtrl, 0, sizeof(stCtrl));
    stCtrl.u32CurFrmNum = frame_num; stCtrl.u8TimeThr = 20; stCtrl.u8DiffMaxThr = 10; stCtrl.u8DiffMinThr = 10; stCtrl.u8FastLearnRate = 4;
    IVE_HANDLE handle;
    if (RK_MPI_IVE_UpdateBgModel(&handle, m_img_u8c1_in1.get(), &stBgModel, m_img_u8c1_in2.get(), m_img_u8c1_out1.get(), &stCtrl, RK_TRUE) != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_UpdateBgModel failed.");
    }
    return copy_from_ive_buffer(m_img_u8c1_out1.get(), m_width, m_height);
}

std::tuple<ImageBuffer, ImageBuffer> IVE::sad(const ImageBuffer& src1, const ImageBuffer& src2, IVE_SAD_MODE_E mode, uint16_t threshold, uint8_t min_val, uint8_t max_val) {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_buffers(src1.width, src1.height);
    if (src1.width != src2.width || src1.height != src2.height) throw std::invalid_argument("SAD images must be same size.");
    int out_w = src1.width, out_h = src1.height;
    if (mode == IVE_SAD_MODE_MB_4X4) { out_w /= 4; out_h /= 4; }
    else if (mode == IVE_SAD_MODE_MB_8X8) { out_w /= 8; out_h /= 8; }
    else if (mode == IVE_SAD_MODE_MB_16X16) { out_w /= 16; out_h /= 16; }

    IVE_DST_IMAGE_S stSad, stThr;
    if (create_image(&stSad, IVE_IMAGE_TYPE_U16C1, out_w, out_h) != RK_SUCCESS ||
        create_image(&stThr, IVE_IMAGE_TYPE_U8C1, out_w, out_h) != RK_SUCCESS)
        throw std::runtime_error("IVE SAD: failed to create output images.");
    IveMmzGuard sad_guard, thr_guard;
    sad_guard.set(stSad.au64PhyAddr[0], stSad.au64VirAddr[0]);
    thr_guard.set(stThr.au64PhyAddr[0], stThr.au64VirAddr[0]);

    copy_to_ive_buffer(src1, m_img_u8c1_in1.get());
    copy_to_ive_buffer(src2, m_img_u8c1_in2.get());
    IVE_SAD_CTRL_S stCtrl;
    memset(&stCtrl, 0, sizeof(stCtrl));
    stCtrl.enMode = mode; stCtrl.enOutMode = IVE_SAD_OUT_MODE_BOTH; stCtrl.enOutBits = IVE_SAD_OUT_16_BITS;
    stCtrl.u16Thr = threshold; stCtrl.u8MinVal = min_val; stCtrl.u8MaxVal = max_val;
    IVE_HANDLE handle;
    if (RK_MPI_IVE_SAD(&handle, m_img_u8c1_in1.get(), m_img_u8c1_in2.get(), &stSad, &stThr, &stCtrl, RK_TRUE) != RK_SUCCESS)
        throw std::runtime_error("RK_MPI_IVE_SAD failed.");
    return std::make_tuple(copy_from_ive_buffer(&stSad), copy_from_ive_buffer(&stThr));
}

std::vector<ImageBuffer> IVE::create_pyramid(const ImageBuffer& src, int levels) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (src.format != visiong::kGray8Format) throw std::invalid_argument("Pyramid creation requires a GRAY8 source image.");
    if (levels <= 1 || levels > 4) throw std::invalid_argument("Pyramid levels must be 2-4.");

    // Build pyramid manually with Filter + DMA downsampling. / 使用 Filter + DMA 下采样手工构建金字塔。
    // This path is more stable than RK_MPI_IVE_Pyramid_Create on this platform.
    std::vector<ImageBuffer> pyramid_imgs;
    pyramid_imgs.reserve(levels);
    
    // Level 0 is the original input. / 第 0 层即原始输入。
    pyramid_imgs.push_back(src);
    
    std::vector<std::unique_ptr<IVE_IMAGE_S>> pyr_images(levels);
    std::vector<IveMmzGuard> guards(levels);
    
    // Allocate per-level images. / 为每一层分配图像。
    int cur_w = src.width;
    int cur_h = src.height;
    for (int i = 0; i < levels; ++i) {
        pyr_images[i] = std::make_unique<IVE_IMAGE_S>();
        memset(pyr_images[i].get(), 0, sizeof(IVE_IMAGE_S));
        if (create_image(pyr_images[i].get(), IVE_IMAGE_TYPE_U8C1, cur_w, cur_h) != RK_SUCCESS) {
            throw std::runtime_error("IVE Pyramid: failed to create pyramid level " + std::to_string(i));
        }
        guards[i].set(pyr_images[i]->au64PhyAddr[0], pyr_images[i]->au64VirAddr[0]);
        cur_w /= 2;
        cur_h /= 2;
        if (cur_w < 8 || cur_h < 8) break; // stop when the next level is too small
    }
    
    // Copy level 0 image. / 复制第 0 层图像。
    copy_to_ive_buffer(src, pyr_images[0].get());
    
    // 5x5 Gaussian filter used before decimation. / 在降采样前使用 5x5 高斯滤波。
    IVE_FILTER_CTRL_S filter_ctrl;
    memset(&filter_ctrl, 0, sizeof(filter_ctrl));
    filter_ctrl.u8Norm = 8;     // 2^8 = 256, Gaussian kernel sum normalization
    filter_ctrl.u8OutMode = 0;  // U8 output
    RK_S8 gaussian_mask[25] = {
        1,  4,  6,  4, 1,
        4, 16, 24, 16, 4,
        6, 24, 36, 24, 6,
        4, 16, 24, 16, 4,
        1,  4,  6,  4, 1
    };
    memcpy(filter_ctrl.as8Mask, gaussian_mask, sizeof(gaussian_mask));
    
    // DMA interval-copy config for 2x downsampling. / 用于 2 倍下采样的 DMA interval-copy 配置。
    IVE_DMA_CTRL_S dma_ctrl;
    memset(&dma_ctrl, 0, sizeof(dma_ctrl));
    dma_ctrl.enMode = IVE_DMA_MODE_INTERVAL_COPY;
    dma_ctrl.u8HorSegSize = 2;
    dma_ctrl.u8ElemSize = 1;
    dma_ctrl.u8VerSegRows = 2;
    
    // Temporary image for filtered intermediate output. / 用于滤波中间结果的临时图像。
    IVE_IMAGE_S stFilterOut;
    memset(&stFilterOut, 0, sizeof(stFilterOut));
    if (create_image(&stFilterOut, IVE_IMAGE_TYPE_U8C1, src.width, src.height) != RK_SUCCESS) {
        throw std::runtime_error("IVE Pyramid: failed to create filter output image.");
    }
    IveMmzGuard filter_guard;
    filter_guard.set(stFilterOut.au64PhyAddr[0], stFilterOut.au64VirAddr[0]);
    
    IVE_HANDLE handle;
    
    // Build pyramid level by level. / 逐层构建金字塔。
    for (int k = 0; k < levels - 1; ++k) {
        if (!pyr_images[k] || !pyr_images[k + 1]) break;
        
        // Match temporary output shape with current level. / 让临时输出形状与当前层保持一致。
        stFilterOut.u32Width = pyr_images[k]->u32Width;
        stFilterOut.u32Height = pyr_images[k]->u32Height;
        stFilterOut.au32Stride[0] = pyr_images[k]->au32Stride[0];
        
        // 1) Blur current level. / 1）对当前层做模糊。
        RK_S32 s32Ret = RK_MPI_IVE_Filter(&handle, pyr_images[k].get(), &stFilterOut, &filter_ctrl, RK_TRUE);
        if (s32Ret != RK_SUCCESS) {
            throw std::runtime_error("IVE Pyramid: filter failed at level " + std::to_string(k));
        }
        
        // 2) Downsample into next level via DMA interval copy. / 2）通过 DMA interval copy 下采样到下一层。
        IVE_DATA_S inData, outData;
        RK_MPI_IVE_CvtImageToData(&stFilterOut, &inData);
        RK_MPI_IVE_CvtImageToData(pyr_images[k + 1].get(), &outData);
        
        s32Ret = RK_MPI_IVE_DMA(&handle, &inData, &outData, &dma_ctrl, RK_TRUE);
        if (s32Ret != RK_SUCCESS) {
            throw std::runtime_error("IVE Pyramid: DMA failed at level " + std::to_string(k));
        }
        
        // 3) Export next level. / 3）导出下一层。
        pyramid_imgs.push_back(copy_from_ive_buffer(pyr_images[k + 1].get()));
    }
    
    return pyramid_imgs;
}

