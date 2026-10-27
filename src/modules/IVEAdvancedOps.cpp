// SPDX-License-Identifier: LGPL-3.0-or-later
#include "visiong/modules/IVE.h"
#include "visiong/common/pixel_format.h"
#include "visiong/core/BufferStateMachine.h"

#include "modules/internal/ive_memory.h"

#include <algorithm>
#include <cstring>
#include <limits>
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

constexpr int kLkCoordinateScale = 128;

RK_U32 lk_pyramid_workspace_size(RK_U32 width, RK_U32 height) {
    // Match the per-level strides used by Pyramid_Create. / 按 Pyramid_Create 的逐层步长计算空间。
    RK_U64 bytes = static_cast<RK_U64>(calc_stride(width)) * height;
    for (unsigned level = 0; level <= 3; ++level) {
        const RK_U32 w = level ? (width + 1) >> level : width;
        const RK_U32 h = level ? (height + 1) >> level : height;
        bytes += static_cast<RK_U64>(calc_stride(w)) * h;
    }
    bytes = (bytes + kImageAlign - 1) & ~static_cast<RK_U64>(kImageAlign - 1);
    if (bytes > std::numeric_limits<RK_U32>::max() / 2) {
        throw std::invalid_argument("LKOpticalFlow: Pyramid workspace is too large.");
    }
    return static_cast<RK_U32>(bytes);
}

class IveCpuAccess {
public:
    explicit IveCpuAccess(RK_U64 address) : address_(reinterpret_cast<void*>(address)) {
        if (mmz_flush_start(address_) != RK_SUCCESS) {
            throw std::runtime_error("IVE: Failed to begin memory buffer access.");
        }
    }
    ~IveCpuAccess() { mmz_flush_end(address_); }
    IveCpuAccess(const IveCpuAccess&) = delete;
    IveCpuAccess& operator=(const IveCpuAccess&) = delete;
private:
    void* address_;
};

template <typename Control>
void initialize_gmm_control(Control& control, bool first_frame) {
    // The public API updates model memory in place; use fixed-size single-model records.
    // 公开接口原地更新模型内存；使用固定长度的单模型记录。
    control.u8FirstFrameFlag = first_frame ? 1 : 0;
    control.u8EnBgOut = 1;
    control.u8MaxModelNum = 1;
    control.u8UpdateVar = 1;
    control.u8q2WeightInitVal = 16;
    control.u8q2WeightAddFactor = 4;
    control.u8q2WeightReduFactor = 1016;
    control.u8q2WeightThr = 4;
    control.u8VarThreshGen = 9;
    control.u8q2BgRatio = 712;
    control.u10q0InitVar = 225;
    control.u10q0MinVar = 200;
    control.u10q0MaxVar = 512;
    control.u8VarThr = 16;
}

void validate_gmm_model(const IVEModel& model, int width, int height) {
    const RK_U64 required = static_cast<RK_U64>(calc_stride(width)) * height * 4 + sizeof(IVE_GMM_NUM_INFO_S);
    if (model.mem_info.u32Size < required) throw std::invalid_argument("GMM model buffer is too small.");
}

IVE_DATA_S background_model_data(const IVEModel& model, int width, int height) {
    const RK_U32 row_bytes = width * sizeof(IVE_BG_MODEL_PIX_S);
    const RK_U32 stride = calc_stride(row_bytes);
    if (model.mem_info.u32Size < static_cast<RK_U64>(stride) * height) {
        throw std::invalid_argument("Background model buffer is too small.");
    }
    IVE_DATA_S data{};
    data.u64PhyAddr = model.mem_info.u64PhyAddr;
    data.u64VirAddr = model.mem_info.u64VirAddr;
    data.u32Width = row_bytes;
    data.u32Height = height;
    data.u32Stride = stride;
    return data;
}

const ImageBuffer& background_frame(const ImageBuffer& source, std::unique_ptr<ImageBuffer>& holder) {
    const int width = calc_stride(source.width);
    const int height = calc_stride(source.height);
    if (width == source.width && height == source.height) return source;
    visiong::bufstate::prepare_cpu_read(source);
    std::vector<unsigned char> bytes(static_cast<size_t>(width) * height, 0);
    for (int y = 0; y < source.height; ++y) {
        std::memcpy(bytes.data() + static_cast<size_t>(y) * width,
                static_cast<const unsigned char*>(source.get_data()) + static_cast<size_t>(y) * source.w_stride, source.width);
    }
    holder = std::make_unique<ImageBuffer>(width, height, visiong::kGray8Format, std::move(bytes));
    return *holder;
}

ImageBuffer widen_gradient(const IVE_IMAGE_S& image) {
    std::vector<unsigned char> bytes(static_cast<size_t>(image.u32Width) * image.u32Height * sizeof(int16_t));
    auto* target = reinterpret_cast<int16_t*>(bytes.data());
    IveCpuAccess access(image.au64VirAddr[0]);
    for (RK_U32 y = 0; y < image.u32Height; ++y) {
        const auto* row = reinterpret_cast<const int8_t*>(image.au64VirAddr[0]) + static_cast<size_t>(y) * image.au32Stride[0];
        for (RK_U32 x = 0; x < image.u32Width; ++x) target[static_cast<size_t>(y) * image.u32Width + x] = row[x];
    }
    return ImageBuffer(image.u32Width, image.u32Height, visiong::kS16C1Format, std::move(bytes));
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
    validate_gmm_model(model, src.width, src.height);
    IVE_GMM_CTRL_S stCtrl{};
    initialize_gmm_control(stCtrl, first_frame);
    IVE_HANDLE handle;
    RK_S32 s32Ret = RK_MPI_IVE_GMM(&handle, m_img_u8c1_in1.get(), m_img_u8c1_out1.get(), m_img_u8c1_out2.get(), m_img_u8c1_in2.get(), &model.mem_info, &stCtrl, RK_TRUE);
    if (s32Ret != RK_SUCCESS) throw std::runtime_error("RK_MPI_IVE_GMM failed with error code: " + std::to_string(s32Ret));
    return std::make_tuple(copy_from_ive_buffer(m_img_u8c1_out1.get(), m_width, m_height), copy_from_ive_buffer(m_img_u8c1_out2.get(), m_width, m_height));
}

std::tuple<ImageBuffer, ImageBuffer> IVE::gmm2(const ImageBuffer& src, const ImageBuffer& factor, IVEModel& model, bool first_frame) {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_buffers(src.width, src.height);
    if (src.format != visiong::kGray8Format || factor.format != visiong::kU16C1Format) throw std::invalid_argument("GMM2 requires GRAY8 source and U16C1 factor images.");
    copy_to_ive_buffer(src, m_img_u8c1_in1.get());
    copy_to_ive_buffer(factor, m_img_u16c1_in1.get());
    validate_gmm_model(model, src.width, src.height);
    IVE_GMM2_CTRL_S stCtrl{};
    initialize_gmm_control(stCtrl, first_frame);
    stCtrl.u8UseVarFactor = 1;
    IVE_HANDLE handle;
    RK_S32 s32Ret = RK_MPI_IVE_GMM2(&handle, m_img_u8c1_in1.get(), (IVE_SRC_IMAGE_S*)m_img_u16c1_in1.get(), m_img_u8c1_out1.get(), m_img_u8c1_out2.get(), m_img_u8c1_in2.get(), &model.mem_info, &stCtrl, RK_TRUE);
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
    IVE_IMAGE_S horizontal = *m_img_u8c1_out1;
    IVE_IMAGE_S vertical = *m_img_u8c1_out2;
    horizontal.enType = IVE_IMAGE_TYPE_S8C1;
    vertical.enType = IVE_IMAGE_TYPE_S8C1;
    IVE_HANDLE handle;
    if (RK_MPI_IVE_NormGrad(&handle, m_img_u8c1_in1.get(), &horizontal, &vertical, nullptr, &stCtrl, RK_TRUE) != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_NormGrad failed.");
    }
    return std::make_tuple(widen_gradient(horizontal), widen_gradient(vertical));
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
    for (const auto& point : points) {
        const int x = std::get<0>(point);
        const int y = std::get<1>(point);
        if (x < 0 || y < 0 || x >= prev_img.width || y >= prev_img.height) {
            throw std::invalid_argument("LKOpticalFlow: Points must lie inside the source image.");
        }
    }

    ensure_buffers(prev_img.width, prev_img.height);

    if (prev_img.width < 64 || prev_img.height < 64) {
        throw std::invalid_argument("LKOpticalFlow: Four pyramid levels require images of at least 64x64.");
    }
    const RK_U32 pyramid_size = lk_pyramid_workspace_size(prev_img.width, prev_img.height);
    if (!m_lk_pyramid_mem) {
        auto workspace = std::make_unique<IVE_MEM_INFO_S>();
        if (create_mem_info(workspace.get(), pyramid_size * 2) != RK_SUCCESS) {
            throw std::runtime_error("LKOpticalFlow: Failed to allocate pyramid workspace.");
        }
        m_lk_pyramid_mem = std::move(workspace);
    }

    std::unique_ptr<ImageBuffer> prev_holder;
    std::unique_ptr<ImageBuffer> next_holder;
    const ImageBuffer& prev_gray = ensure_gray_input(prev_img, prev_holder);
    const ImageBuffer& next_gray = ensure_gray_input(next_img, next_holder);

    copy_to_ive_buffer(prev_gray, m_img_u8c1_in1.get());
    copy_to_ive_buffer(next_gray, m_img_u8c1_in2.get());

    // End CPU access before submitting DMA, including freshly initialized outputs.
    // 提交 DMA 前结束 CPU 访问，新初始化的输出缓冲区也需同步。
    {
        IveCpuAccess input_access(m_lk_pts_mem->u64VirAddr);
        auto* coordinates = reinterpret_cast<IVE_POINT_S25Q7_S*>(m_lk_pts_mem->u64VirAddr);
        for (size_t i = 0; i < points.size(); ++i) {
            coordinates[i].s25q7X = static_cast<IVE_S25Q7>(std::get<0>(points[i])) * kLkCoordinateScale;
            coordinates[i].s25q7Y = static_cast<IVE_S25Q7>(std::get<1>(points[i])) * kLkCoordinateScale;
        }
    }
    for (const auto* memory : {m_lk_next_pts_mem.get(), m_lk_status_mem.get(), m_lk_error_mem.get()}) {
        IveCpuAccess output_access(memory->u64VirAddr);
        std::memset(reinterpret_cast<void*>(memory->u64VirAddr), 0, memory->u32Size);
    }

    IVE_IMAGE_S previous_pyramid[4]{};
    IVE_IMAGE_S current_pyramid[4]{};
    IVE_PYRAMID_CTRL_S pyramid_ctrl{};
    pyramid_ctrl.level = 3;
    pyramid_ctrl.stPyramidMem = *m_lk_pyramid_mem;
    pyramid_ctrl.stPyramidMem.u32Size = pyramid_size;
    IVE_HANDLE handle{};
    RK_S32 ret = RK_MPI_IVE_Pyramid_Create(&handle, m_img_u8c1_in1.get(), previous_pyramid, &pyramid_ctrl, true);
    if (ret != RK_SUCCESS) {
        throw std::runtime_error("LKOpticalFlow: Failed to build the previous pyramid: " + std::to_string(ret));
    }
    pyramid_ctrl.stPyramidMem.u64PhyAddr += pyramid_size;
    pyramid_ctrl.stPyramidMem.u64VirAddr += pyramid_size;
    ret = RK_MPI_IVE_Pyramid_Create(&handle, m_img_u8c1_in2.get(), current_pyramid, &pyramid_ctrl, true);
    if (ret != RK_SUCCESS) {
        throw std::runtime_error("LKOpticalFlow: Failed to build the current pyramid: " + std::to_string(ret));
    }

    IVE_LK_OPTICAL_FLOW_PYR_CTRL_S control{};
    control.enOutMode = IVE_LK_OPTICAL_FLOW_PYR_OUT_MODE_BOTH;
    control.u16PtsNum = static_cast<RK_U16>(points.size());
    control.u8MaxLevel = 3;
    control.u0q8MinEigThr = 100;
    control.u8IterCnt = 20;
    control.u0q11Eps = 32;
    ret = RK_MPI_IVE_LKOpticalFlowPyr(&handle, previous_pyramid, current_pyramid,
            m_lk_pts_mem.get(), m_lk_next_pts_mem.get(), m_lk_status_mem.get(), m_lk_error_mem.get(), &control, true);
    if (ret != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_LKOpticalFlowPyr failed: " + std::to_string(ret));
    }

    std::vector<IVEMotionVector> results;
    results.reserve(points.size());
    IveCpuAccess next_access(m_lk_next_pts_mem->u64VirAddr);
    IveCpuAccess status_access(m_lk_status_mem->u64VirAddr);
    const auto* tracked = reinterpret_cast<const IVE_POINT_S25Q7_S*>(m_lk_next_pts_mem->u64VirAddr);
    const auto* status = reinterpret_cast<const RK_U8*>(m_lk_status_mem->u64VirAddr);
    for (size_t i = 0; i < points.size(); ++i) {
        const int64_t dx = static_cast<int64_t>(tracked[i].s25q7X) - std::get<0>(points[i]) * kLkCoordinateScale;
        const int64_t dy = static_cast<int64_t>(tracked[i].s25q7Y) - std::get<1>(points[i]) * kLkCoordinateScale;
        const bool valid = status[i] == 1 &&
                dx >= std::numeric_limits<short>::min() && dx <= std::numeric_limits<short>::max() &&
                dy >= std::numeric_limits<short>::min() && dy <= std::numeric_limits<short>::max();
        results.push_back({valid ? 0 : -1, valid ? static_cast<short>(dx) : short{0},
                          valid ? static_cast<short>(dy) : short{0}});
    }

    return results;
}

std::vector<std::tuple<int, int>> IVE::st_corner(const ImageBuffer& src, uint16_t max_corners, uint16_t min_dist, uint8_t quality_level) {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_buffers(src.width, src.height);

    std::unique_ptr<ImageBuffer> src_holder;
    const ImageBuffer& gray = ensure_gray_input(src, src_holder);
    copy_to_ive_buffer(gray, m_img_u8c1_in1.get());
    for (const auto* memory : {m_st_candi_corner_mem.get(), m_st_corner_info_mem.get()}) {
        IveCpuAccess access(memory->u64VirAddr);
        std::memset(reinterpret_cast<void*>(memory->u64VirAddr), 0, memory->u32Size);
    }

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

    IVE_ST_CORNER_CTRL_S stCornerCtrl;
    std::memset(&stCornerCtrl, 0, sizeof(stCornerCtrl));
    stCornerCtrl.stMem = *m_st_ctrl_mem.get();
    stCornerCtrl.u16MaxCornerNum = std::min<uint16_t>(IVE_ST_MAX_CORNER_NUM, max_corners);
    stCornerCtrl.u16MinDist = min_dist;

    ret = RK_MPI_IVE_STCorner(&handle, m_img_u8c1_in1.get(), m_st_candi_corner_mem.get(), m_st_corner_info_mem.get(), &stCornerCtrl, RK_TRUE);
    if (ret != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_STCorner failed: " + std::to_string(ret));
    }

    IveCpuAccess corner_access(m_st_corner_info_mem->u64VirAddr);

    const IVE_ST_CORNER_INFO_S* corner_info =
        reinterpret_cast<const IVE_ST_CORNER_INFO_S*>(m_st_corner_info_mem->u64VirAddr);
    const uint16_t valid_corners = std::min<uint16_t>(corner_info->u16CornerNum, IVE_ST_MAX_CORNER_NUM);

    result_points.reserve(valid_corners);
    for (uint16_t i = 0; i < valid_corners; ++i) {
        result_points.emplace_back(corner_info->astCorner[i].u16X, corner_info->astCorner[i].u16Y);
    }

    return result_points;
}

void IVE::match_bg_model(const ImageBuffer& current_img, IVEModel& bg_model, uint32_t frame_num) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (current_img.format != visiong::kGray8Format) throw std::invalid_argument("Background modeling requires GRAY8 input.");
    std::unique_ptr<ImageBuffer> holder;
    const auto& frame = background_frame(current_img, holder);
    ensure_buffers(frame.width, frame.height);
    copy_to_ive_buffer(frame, m_img_u8c1_in1.get());
    IVE_DATA_S stBgModel = background_model_data(bg_model, frame.width, frame.height);
    IVE_MATCH_BG_MODEL_CTRL_S stCtrl;
    memset(&stCtrl, 0, sizeof(stCtrl));
    stCtrl.u32CurFrmNum = frame_num; 
    stCtrl.u8CodeWordNum = 3;
    stCtrl.u8DiffMaxThr = 10; 
    stCtrl.u8DiffMinThr = 10; 
    stCtrl.u8TrainTimeThr = 20;

    IVE_HANDLE handle;
    if (RK_MPI_IVE_MatchBgModel(&handle, m_img_u8c1_in1.get(), &stBgModel, &stCtrl, RK_TRUE) != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_MatchBgModel failed.");
    }
}

std::tuple<ImageBuffer, ImageBuffer> IVE::update_bg_model(const ImageBuffer& current_img, IVEModel& bg_model, uint32_t frame_num) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (current_img.format != visiong::kGray8Format) throw std::invalid_argument("Background modeling requires GRAY8 input.");
    std::unique_ptr<ImageBuffer> holder;
    const auto& frame = background_frame(current_img, holder);
    ensure_buffers(frame.width, frame.height);
    copy_to_ive_buffer(frame, m_img_u8c1_in1.get());
    IVE_DATA_S stBgModel = background_model_data(bg_model, frame.width, frame.height);
    IVE_UPDATE_BG_MODEL_CTRL_S stCtrl; memset(&stCtrl, 0, sizeof(stCtrl));
    stCtrl.u32CurFrmNum = frame_num; stCtrl.u8CodeWordNum = 3; stCtrl.u8TimeThr = 50; stCtrl.u8DiffMaxThr = 10; stCtrl.u8DiffMinThr = 10; stCtrl.u8FastLearnRate = 4; stCtrl.u8Alpha = 4;
    IVE_HANDLE handle;
    if (RK_MPI_IVE_UpdateBgModel(&handle, m_img_u8c1_in1.get(), &stBgModel, m_img_u8c1_out1.get(), m_img_u8c1_out2.get(), &stCtrl, RK_TRUE) != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_UpdateBgModel failed.");
    }
    return std::make_tuple(copy_from_ive_buffer(m_img_u8c1_out1.get(), current_img.width, current_img.height),
                           copy_from_ive_buffer(m_img_u8c1_out2.get(), current_img.width, current_img.height));
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

