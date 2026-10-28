// SPDX-License-Identifier: LGPL-3.0-or-later
#include "visiong/modules/IVE.h"
#include "visiong/common/pixel_format.h"
#include "visiong/core/BufferStateMachine.h"

#include "modules/internal/ive_memory.h"
#include "modules/internal/ive_runtime_compat.h"

#include <algorithm>
#include <cstring>
#include <dlfcn.h>
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

bool has_known_lk_stride_defect() {
#if defined(__arm__) && !defined(__aarch64__)
    static const bool defective = [] {
        const auto* function = static_cast<const unsigned char*>(
            dlsym(RTLD_DEFAULT, "rve_hal_lk_params_to_regs"));
        if (!function || (reinterpret_cast<uintptr_t>(function) & 1u)) return false;
        uint32_t instructions[12]{};
        std::memcpy(instructions, function, sizeof(instructions));
        // This runtime uses max_level+3 for a current-image array fixed at index six.
        // 此运行库用 max_level+3 索引固定从第六项开始的当前图像数组。
        return instructions[0] == 0xe52db004u && instructions[10] == 0xe5d33006u &&
               instructions[11] == 0xe2833003u;
    }();
    return defective;
#else
    return false;
#endif
}

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

void validate_gmm_model(const IVEModel& model, int width, int height, int models) {
    const RK_U64 required = static_cast<RK_U64>(calc_stride(width)) * height * models * 4 + sizeof(IVE_GMM_NUM_INFO_S);
    if (model.mem_info.u32Size < required || model.mem_info.u32Size % 4) {
        throw std::invalid_argument("GMM model buffer must be large enough and four-byte aligned.");
    }
}

void zero_gmm_memory(const IVE_MEM_INFO_S& memory) {
    void* address = reinterpret_cast<void*>(memory.u64VirAddr);
    if (mmz_flush_start(address) != RK_SUCCESS) throw std::runtime_error("Failed to begin GMM output access.");
    std::memset(address, 0, memory.u32Size);
    if (mmz_flush_end(address) != RK_SUCCESS) throw std::runtime_error("Failed to end GMM output access.");
}

RK_U32 gmm_record_count(const IVE_MEM_INFO_S& memory) {
    void* address = reinterpret_cast<void*>(memory.u64VirAddr);
    if (mmz_flush_start(address) != RK_SUCCESS) throw std::runtime_error("Failed to begin GMM state access.");
    RK_U32 result;
    std::memcpy(&result, static_cast<const unsigned char*>(address) + memory.u32Size - sizeof(IVE_GMM_NUM_INFO_S), sizeof(result));
    if (mmz_flush_end(address) != RK_SUCCESS) throw std::runtime_error("Failed to end GMM state access.");
    return result;
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
    if (!source.is_valid() || source.width <= 0 || source.height <= 0 || source.width > 4096 || source.height > 4096 ||
        source.w_stride < source.width || source.h_stride < source.height ||
        static_cast<uint64_t>(source.get_size()) < static_cast<uint64_t>(source.height - 1) * source.w_stride + source.width) {
        throw std::invalid_argument("Background modeling requires valid image dimensions, strides and storage.");
    }
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

const ImageBuffer& ensure_gray_input(const ImageBuffer& src, std::unique_ptr<ImageBuffer>& holder) {
    if (src.format == visiong::kGray8Format) {
        return src;
    }
    holder = std::make_unique<ImageBuffer>(src.to_grayscale());
    return *holder;
}

} // namespace

std::tuple<ImageBuffer, ImageBuffer> IVE::gmm(const ImageBuffer& src, IVEModel& model, bool first_frame) {
    auto output = gmm_execute(src, nullptr, model, 1, first_frame, false);
    return std::make_tuple(std::move(std::get<0>(output)), std::move(std::get<1>(output)));
}

std::tuple<ImageBuffer, ImageBuffer> IVE::gmm2(const ImageBuffer& src, const ImageBuffer& factor, IVEModel& model, bool first_frame) {
    auto output = gmm_execute(src, &factor, model, 1, first_frame, false);
    return std::make_tuple(std::move(std::get<0>(output)), std::move(std::get<1>(output)));
}

std::tuple<ImageBuffer, ImageBuffer, ImageBuffer> IVE::gmm_config(
    const ImageBuffer& src, IVEModel& model, int max_models, bool first_frame) {
    return gmm_execute(src, nullptr, model, max_models, first_frame, true);
}

std::tuple<ImageBuffer, ImageBuffer, ImageBuffer> IVE::gmm2_config(
    const ImageBuffer& src, const ImageBuffer& factor, IVEModel& model, int max_models, bool first_frame) {
    return gmm_execute(src, &factor, model, max_models, first_frame, true);
}

std::tuple<ImageBuffer, ImageBuffer, ImageBuffer> IVE::gmm_execute(
    const ImageBuffer& src, const ImageBuffer* factor, IVEModel& model,
    int max_models, bool first_frame, bool return_matched) {
    std::lock_guard<std::mutex> lock(m_mutex);
    std::lock_guard<std::mutex> model_lock(model.m_mutex);
    if (max_models < 1 || max_models > 5) throw std::invalid_argument("GMM max_models must be in [1,5].");
    model.validate_dimensions(src.width, src.height);
    if (src.format != visiong::kGray8Format) throw std::invalid_argument("GMM requires a GRAY8 source image.");
    if (factor && (factor->format != visiong::kU16C1Format || factor->width != src.width || factor->height != src.height)) {
        throw std::invalid_argument("GMM2 requires a U16C1 factor image with matching dimensions.");
    }
    const auto algorithm = factor ? IVEModel::Algorithm::Gmm2 : IVEModel::Algorithm::Gmm;
    if (!first_frame && (model.m_algorithm != algorithm || model.m_gmm_models != max_models)) {
        throw std::invalid_argument("GMM requires first_frame=True when initializing or changing algorithm or model count.");
    }
    validate_gmm_model(model, src.width, src.height, max_models);
    if (max_models > 1) visiong::ive_internal::require_gmm_separate_output_runtime();
    ensure_buffers(src.width, src.height);
    copy_to_ive_buffer(src, m_img_u8c1_in1.get());
    if (factor) copy_to_ive_buffer(*factor, m_img_u16c1_in1.get());
    model.prepare_gmm_locked(max_models > 1);
    try {
        if (first_frame) model.clear_locked();
        RK_S32 result;
        if (max_models > 1) {
            const auto count = gmm_record_count(model.mem_info);
            if (count > static_cast<RK_U64>(calc_stride(src.width)) * src.height * max_models) {
                throw std::runtime_error("GMM model record count is invalid; reinitialize the sequence.");
            }
            zero_gmm_memory(model.m_gmm_output);
            result = visiong::ive_internal::gmm_separate_output(
                *m_img_u8c1_in1, factor ? m_img_u16c1_in1.get() : nullptr,
                *m_img_u8c1_out1, *m_img_u8c1_out2, model.m_gmm_matched,
                model.mem_info, model.m_gmm_output, max_models, first_frame, count);
            if (result == RK_SUCCESS) std::swap(model.mem_info, model.m_gmm_output);
        } else {
            IVE_HANDLE handle = 0;
            if (factor) {
                IVE_GMM2_CTRL_S control{};
                initialize_gmm_control(control, first_frame);
                control.u8UseVarFactor = 1;
                result = RK_MPI_IVE_GMM2(&handle, m_img_u8c1_in1.get(), m_img_u16c1_in1.get(),
                    m_img_u8c1_out1.get(), m_img_u8c1_out2.get(), &model.m_gmm_matched,
                    &model.mem_info, &control, RK_TRUE);
            } else {
                IVE_GMM_CTRL_S control{};
                initialize_gmm_control(control, first_frame);
                result = RK_MPI_IVE_GMM(&handle, m_img_u8c1_in1.get(), m_img_u8c1_out1.get(),
                    m_img_u8c1_out2.get(), &model.m_gmm_matched, &model.mem_info, &control, RK_TRUE);
            }
        }
        if (result != RK_SUCCESS) throw std::runtime_error("Hardware GMM failed with error code: " + std::to_string(result));
        auto foreground = copy_from_ive_buffer(m_img_u8c1_out1.get(), m_width, m_height);
        auto background = copy_from_ive_buffer(m_img_u8c1_out2.get(), m_width, m_height);
        auto matched = return_matched ? copy_from_ive_buffer(&model.m_gmm_matched, m_width, m_height) : ImageBuffer();
        model.m_algorithm = algorithm;
        model.m_gmm_models = max_models;
        return std::make_tuple(std::move(foreground), std::move(background), std::move(matched));
    } catch (...) {
        // Matched records may already have changed even when the model output was not committed.
        // 即使模型输出尚未提交，匹配记录也可能已改变。
        model.m_algorithm = IVEModel::Algorithm::Uninitialized;
        model.m_gmm_models = 0;
        throw;
    }
}

ImageBuffer IVE::lbp(const ImageBuffer& src, bool abs_mode, int threshold) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (src.format != visiong::kGray8Format) throw std::invalid_argument("LBP requires a GRAY8 source image.");
    if ((abs_mode && (threshold < 0 || threshold > 255)) ||
        (!abs_mode && (threshold < -128 || threshold > 127))) {
        throw std::invalid_argument("LBP threshold must be [-128,127] in NORMAL mode or [0,255] in ABS mode.");
    }
    ensure_buffers(src.width, src.height);
    copy_to_ive_buffer(src, m_img_u8c1_in1.get());
    
    IVE_LBP_CTRL_S stCtrl; 
    memset(&stCtrl, 0, sizeof(stCtrl)); 
    stCtrl.enMode = abs_mode ? IVE_LBP_CMP_MODE_ABS : IVE_LBP_CMP_MODE_NORMAL;
    
    if (abs_mode) {
        stCtrl.un8BitThr.u8Val = static_cast<uint8_t>(threshold);
    } else {
        stCtrl.un8BitThr.s8Val = static_cast<int8_t>(threshold);
    }
    
    IVE_HANDLE handle;
    if (RK_MPI_IVE_LBP(&handle, m_img_u8c1_in1.get(), m_img_u8c1_out1.get(), &stCtrl, RK_TRUE) != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_LBP failed.");
    }
    return copy_from_ive_buffer(m_img_u8c1_out1.get(), m_width, m_height);
}

std::tuple<ImageBuffer, ImageBuffer> IVE::norm_grad(const ImageBuffer& src) {
    auto result = norm_grad_config(src, IVE_NORM_GRAD_OUT_CTRL_HOR_AND_VER, 7, {});
    const auto widen = [](const ImageBuffer& image) {
        visiong::bufstate::prepare_cpu_read(image);
        std::vector<unsigned char> bytes(static_cast<size_t>(image.width) * image.height * sizeof(int16_t));
        auto* output = reinterpret_cast<int16_t*>(bytes.data());
        for (int y = 0; y < image.height; ++y) {
            const auto* row = reinterpret_cast<const int8_t*>(image.get_data()) + static_cast<size_t>(y) * image.w_stride;
            for (int x = 0; x < image.width; ++x) output[static_cast<size_t>(y) * image.width + x] = row[x];
        }
        return ImageBuffer(image.width, image.height, visiong::kS16C1Format, std::move(bytes));
    };
    return std::make_tuple(widen(std::get<0>(result)), widen(std::get<1>(result)));
}

std::tuple<ImageBuffer, ImageBuffer, ImageBuffer> IVE::norm_grad_config(
        const ImageBuffer& src, IVE_NORM_GRAD_OUT_CTRL_E out_ctrl, uint8_t norm,
        const std::vector<int8_t>& mask) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (src.format != visiong::kGray8Format) throw std::invalid_argument("NormGrad requires a GRAY8 source image.");
    if (out_ctrl < IVE_NORM_GRAD_OUT_CTRL_HOR_AND_VER || out_ctrl >= IVE_NORM_GRAD_OUT_CTRL_BUTT) {
        throw std::invalid_argument("Invalid normalized gradient output mode.");
    }
    if (norm > 15) throw std::invalid_argument("Normalized gradient shift must be in [0,15].");
    if (!mask.empty() && mask.size() != 25) throw std::invalid_argument("Normalized gradient mask must contain 25 coefficients.");
    ensure_buffers(src.width, src.height);
    copy_to_ive_buffer(src, m_img_u8c1_in1.get());
    IVE_NORM_GRAD_CTRL_S stCtrl{};
    // Compute both hardware planes before selecting or packing public outputs. / 先计算硬件双分量，再选择或打包对外输出。
    stCtrl.enOutCtrl = IVE_NORM_GRAD_OUT_CTRL_HOR_AND_VER;
    stCtrl.u8Norm = norm;
    const RK_S8 default_mask[25] = { -1, -2, 0, 2, 1, -4, -8, 0, 8, 4, -6, -12, 0, 12, 6, -4, -8, 0, 8, 4, -1, -2, 0, 2, 1 };
    std::memcpy(stCtrl.as8Mask, mask.empty() ? default_mask : mask.data(), sizeof(stCtrl.as8Mask));
    IVE_IMAGE_S horizontal = *m_img_u8c1_out1;
    IVE_IMAGE_S vertical = *m_img_u8c1_out2;
    horizontal.enType = IVE_IMAGE_TYPE_S8C1;
    vertical.enType = IVE_IMAGE_TYPE_S8C1;
    IVE_HANDLE handle;
    if (RK_MPI_IVE_NormGrad(&handle, m_img_u8c1_in1.get(), &horizontal, &vertical, nullptr, &stCtrl, RK_TRUE) != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_NormGrad failed.");
    }
    ImageBuffer result_h;
    ImageBuffer result_v;
    ImageBuffer result_hv;
    if (out_ctrl == IVE_NORM_GRAD_OUT_CTRL_COMBINE) {
        // Store signed H/V bytes in the low/high byte of one U16 pixel. / 将有符号 H/V 字节分别存入 U16 像素的低/高字节。
        std::vector<unsigned char> bytes(static_cast<size_t>(src.width) * src.height * sizeof(uint16_t));
        IveCpuAccess horizontal_access(horizontal.au64VirAddr[0]);
        IveCpuAccess vertical_access(vertical.au64VirAddr[0]);
        for (int y = 0; y < src.height; ++y) {
            const auto* row_h = reinterpret_cast<const uint8_t*>(horizontal.au64VirAddr[0]) + static_cast<size_t>(y) * horizontal.au32Stride[0];
            const auto* row_v = reinterpret_cast<const uint8_t*>(vertical.au64VirAddr[0]) + static_cast<size_t>(y) * vertical.au32Stride[0];
            for (int x = 0; x < src.width; ++x) {
                const size_t offset = (static_cast<size_t>(y) * src.width + x) * 2;
                bytes[offset] = row_h[x];
                bytes[offset + 1] = row_v[x];
            }
        }
        result_hv = ImageBuffer(src.width, src.height, visiong::kU16C1Format, std::move(bytes));
    } else {
        if (out_ctrl != IVE_NORM_GRAD_OUT_CTRL_VER) result_h = copy_from_ive_buffer(&horizontal, m_width, m_height);
        if (out_ctrl != IVE_NORM_GRAD_OUT_CTRL_HOR) result_v = copy_from_ive_buffer(&vertical, m_width, m_height);
    }
    return std::make_tuple(std::move(result_h), std::move(result_v), std::move(result_hv));
}

std::vector<IVEMotionVector> IVE::lk_optical_flow(const ImageBuffer& prev_img, const ImageBuffer& next_img, const std::vector<std::tuple<int, int>>& points) {
    auto result = lk_optical_flow_config(prev_img, next_img, points, 3, 20, 100, 32, {});
    return std::move(std::get<0>(result));
}

std::tuple<std::vector<IVEMotionVector>, std::vector<uint16_t>> IVE::lk_optical_flow_config(
        const ImageBuffer& prev_img, const ImageBuffer& next_img,
        const std::vector<std::tuple<int, int>>& points, int max_level, int iterations,
        int min_eigenvalue, int epsilon, const std::vector<std::tuple<int, int>>& initial_points) {
    std::lock_guard<std::mutex> lock(m_mutex);

    if (!prev_img.is_valid() || !next_img.is_valid()) return std::make_tuple(std::vector<IVEMotionVector>{}, std::vector<uint16_t>{});
    if (prev_img.width != next_img.width || prev_img.height != next_img.height) {
        throw std::invalid_argument("LKOpticalFlow: Image dimension mismatch.");
    }
    if (max_level < 0 || max_level > 3 || iterations < 1 || iterations > 20 ||
        min_eigenvalue < 0 || min_eigenvalue > 255 || epsilon < 0 || epsilon > 65535) {
        throw std::invalid_argument("LKOpticalFlow requires level [0,3], iterations [1,20], eigenvalue [0,255], and epsilon [0,65535].");
    }
    if (!initial_points.empty() && initial_points.size() != points.size()) {
        throw std::invalid_argument("LKOpticalFlow initial points must match the number of source points.");
    }
    if (points.empty()) return std::make_tuple(std::vector<IVEMotionVector>{}, std::vector<uint16_t>{});
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
    for (const auto& point : initial_points) {
        const int x = std::get<0>(point);
        const int y = std::get<1>(point);
        if (x < 0 || y < 0 || x >= next_img.width || y >= next_img.height) {
            throw std::invalid_argument("LKOpticalFlow: Initial points must lie inside the next image.");
        }
    }

    if (max_level < 3 && has_known_lk_stride_defect()) {
        throw std::runtime_error(
            "LKOpticalFlow levels 0-2 require a corrected IVE runtime; "
            "the loaded runtime has the known current-image stride defect. "
            "See docs/IVE_LK_RUNTIME.md.");
    }

    ensure_buffers(prev_img.width, prev_img.height);

    if (prev_img.width < 64 || prev_img.height < 64) {
        throw std::invalid_argument("LKOpticalFlow requires images of at least 64x64.");
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
    if (!initial_points.empty()) {
        IveCpuAccess initial_access(m_lk_next_pts_mem->u64VirAddr);
        auto* coordinates = reinterpret_cast<IVE_POINT_S25Q7_S*>(m_lk_next_pts_mem->u64VirAddr);
        for (size_t i = 0; i < initial_points.size(); ++i) {
            coordinates[i].s25q7X = static_cast<IVE_S25Q7>(std::get<0>(initial_points[i])) * kLkCoordinateScale;
            coordinates[i].s25q7Y = static_cast<IVE_S25Q7>(std::get<1>(initial_points[i])) * kLkCoordinateScale;
        }
    }

    IVE_IMAGE_S previous_pyramid[4]{};
    IVE_IMAGE_S current_pyramid[4]{};
    IVE_PYRAMID_CTRL_S pyramid_ctrl{};
    pyramid_ctrl.level = static_cast<RK_U8>(max_level);
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
    control.bUseInitFlow = initial_points.empty() ? RK_FALSE : RK_TRUE;
    control.u16PtsNum = static_cast<RK_U16>(points.size());
    control.u8MaxLevel = static_cast<RK_U8>(max_level);
    control.u0q8MinEigThr = static_cast<IVE_U0Q8>(min_eigenvalue);
    control.u8IterCnt = static_cast<RK_U8>(iterations);
    control.u0q11Eps = static_cast<IVE_U0Q11>(epsilon);
    ret = RK_MPI_IVE_LKOpticalFlowPyr(&handle, previous_pyramid, current_pyramid,
            m_lk_pts_mem.get(), m_lk_next_pts_mem.get(), m_lk_status_mem.get(), m_lk_error_mem.get(), &control, true);
    if (ret != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_LKOpticalFlowPyr failed: " + std::to_string(ret));
    }

    std::vector<IVEMotionVector> results;
    std::vector<uint16_t> errors;
    results.reserve(points.size());
    errors.reserve(points.size());
    IveCpuAccess next_access(m_lk_next_pts_mem->u64VirAddr);
    IveCpuAccess status_access(m_lk_status_mem->u64VirAddr);
    IveCpuAccess error_access(m_lk_error_mem->u64VirAddr);
    const auto* tracked = reinterpret_cast<const IVE_POINT_S25Q7_S*>(m_lk_next_pts_mem->u64VirAddr);
    const auto* status = reinterpret_cast<const RK_U8*>(m_lk_status_mem->u64VirAddr);
    const auto* error = reinterpret_cast<const IVE_U9Q7*>(m_lk_error_mem->u64VirAddr);
    for (size_t i = 0; i < points.size(); ++i) {
        const int64_t dx = static_cast<int64_t>(tracked[i].s25q7X) - std::get<0>(points[i]) * kLkCoordinateScale;
        const int64_t dy = static_cast<int64_t>(tracked[i].s25q7Y) - std::get<1>(points[i]) * kLkCoordinateScale;
        const bool valid = status[i] == 1 &&
                dx >= std::numeric_limits<short>::min() && dx <= std::numeric_limits<short>::max() &&
                dy >= std::numeric_limits<short>::min() && dy <= std::numeric_limits<short>::max();
        results.push_back({valid ? 0 : -1, valid ? static_cast<short>(dx) : short{0},
                          valid ? static_cast<short>(dy) : short{0}});
        errors.push_back(valid ? error[i] : std::numeric_limits<uint16_t>::max());
    }

    return std::make_tuple(std::move(results), std::move(errors));
}

std::vector<IVEMotionVector> IVE::lk_optical_flow_single(
        const ImageBuffer& prev_img, const ImageBuffer& next_img,
        const std::vector<std::tuple<int, int>>& points, uint8_t iterations,
        uint8_t min_eigenvalue, uint16_t epsilon) {
    // The older entry fixes max_level=3 even with empty upper levels; use the public level-zero path.
    // 旧入口即使上层图像为空也固定 max_level=3，单层计算改用公开的第零层路径。
    auto outputs = lk_optical_flow_config(prev_img, next_img, points, 0, iterations, min_eigenvalue, epsilon, {});
    return std::move(std::get<0>(outputs));
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

    // Invalidate initialized cache lines before DMA; the SDK's CPU unpacking requests write-only access.
    // DMA 前失效初始化缓存；SDK 的 CPU 解包仅请求只写访问。
    {
        IveCpuAccess output_access(m_st_corner_info_mem->u64VirAddr);
    }
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

std::vector<std::tuple<int, int, int>> IVE::st_corner_candidates(const ImageBuffer& src, uint8_t quality_level) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!src.is_valid() || src.width > 2048 || src.height > 2048) {
        throw std::invalid_argument("Shi-Tomasi candidates require a valid image no larger than 2048x2048.");
    }
    ensure_buffers(src.width, src.height);
    std::unique_ptr<ImageBuffer> holder;
    const auto& gray = ensure_gray_input(src, holder);
    copy_to_ive_buffer(gray, m_img_u8c1_in1.get());
    {
        IveCpuAccess access(m_st_candi_corner_mem->u64VirAddr);
        std::memset(reinterpret_cast<void*>(m_st_candi_corner_mem->u64VirAddr), 0, m_st_candi_corner_mem->u32Size);
    }
    IVE_ST_CANDI_CORNER_CTRL_S control{};
    control.u0q8QualityLevel = quality_level;
    IVE_HANDLE handle{};
    const RK_S32 ret = RK_MPI_IVE_STCandiCorner(&handle, m_img_u8c1_in1.get(),
            m_st_candi_corner_mem.get(), &control, true);
    if (ret != RK_SUCCESS) throw std::runtime_error("RK_MPI_IVE_STCandiCorner failed: " + std::to_string(ret));
    IveCpuAccess access(m_st_candi_corner_mem->u64VirAddr);
    const auto* bytes = reinterpret_cast<const unsigned char*>(m_st_candi_corner_mem->u64VirAddr);
    const size_t footer_offset = (m_st_candi_corner_mem->u32Size - sizeof(IVE_ST_CANDI_STACK_SIZE_S)) & ~size_t{3};
    RK_U32 count = 0;
    std::memcpy(&count, bytes + footer_offset, sizeof(count));
    if (count > footer_offset / sizeof(RK_U32)) throw std::runtime_error("IVE returned an invalid candidate count.");
    std::vector<std::tuple<int, int, int>> result;
    result.reserve(count);
    for (RK_U32 i = 0; i < count; ++i) {
        RK_U32 packed = 0;
        std::memcpy(&packed, bytes + static_cast<size_t>(i) * sizeof(packed), sizeof(packed));
        // Use the packed layout consumed by the SDK's corner postprocessor. / 使用 SDK 角点后处理实际读取的打包布局。
        const int x = packed & 0x7ff;
        const int y = (packed >> 11) & 0x7ff;
        const int lambda = (packed >> 22) & 0xff;
        if (x >= src.width || y >= src.height) throw std::runtime_error("IVE returned an out-of-bounds candidate corner.");
        result.emplace_back(x, y, lambda);
    }
    return result;
}

std::vector<std::tuple<int, int, int>> IVE::st_corner_sorted(
        const ImageBuffer& src, uint16_t max_corners, uint16_t min_dist, uint8_t quality_level) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!src.is_valid() || src.width > 2048 || src.height > 2048) {
        throw std::invalid_argument("Sorted Shi-Tomasi corners require a valid image no larger than 2048x2048.");
    }
    if (max_corners == 0) return {};
    ensure_buffers(src.width, src.height);
    std::unique_ptr<ImageBuffer> holder;
    const auto& gray = ensure_gray_input(src, holder);
    copy_to_ive_buffer(gray, m_img_u8c1_in1.get());
    for (const auto* memory : {m_st_ctrl_mem.get(), m_st_corner_info_mem.get()}) {
        IveCpuAccess access(memory->u64VirAddr);
        std::memset(reinterpret_cast<void*>(memory->u64VirAddr), 0, memory->u32Size);
    }
    IVE_ST_CORNER_SORTED_CTRL_S control{};
    control.stMem = *m_st_ctrl_mem;
    control.u0q8QualityLevel = quality_level;
    control.u16MaxCornerNum = std::min<uint16_t>(IVE_ST_MAX_CORNER_NUM, max_corners);
    control.u16MinDist = min_dist;
    IVE_HANDLE handle{};
    const RK_S32 ret = RK_MPI_IVE_STCorner_Sorted(&handle, m_img_u8c1_in1.get(),
            m_st_corner_info_mem.get(), &control, true);
    if (ret != RK_SUCCESS) throw std::runtime_error("RK_MPI_IVE_STCorner_Sorted failed: " + std::to_string(ret));
    IveCpuAccess access(m_st_corner_info_mem->u64VirAddr);
    const auto* info = reinterpret_cast<const IVE_ST_CORNER_INFO_S*>(m_st_corner_info_mem->u64VirAddr);
    if (info->u16CornerNum > control.u16MaxCornerNum) throw std::runtime_error("IVE returned an invalid sorted corner count.");
    std::vector<std::tuple<int, int, int>> result;
    result.reserve(info->u16CornerNum);
    for (RK_U16 i = 0; i < info->u16CornerNum; ++i) {
        if (info->astCorner[i].u16X >= src.width || info->astCorner[i].u16Y >= src.height) {
            throw std::runtime_error("IVE returned an out-of-bounds sorted corner.");
        }
        result.emplace_back(info->astCorner[i].u16X, info->astCorner[i].u16Y, info->u8Lambda[i]);
    }
    return result;
}

void IVE::match_bg_model(const ImageBuffer& current_img, IVEModel& bg_model, uint32_t frame_num) {
    std::lock_guard<std::mutex> lock(m_mutex);
    std::lock_guard<std::mutex> model_lock(bg_model.m_mutex);
    bg_model.validate_dimensions(current_img.width, current_img.height);
    if (bg_model.m_algorithm != IVEModel::Algorithm::Uninitialized && bg_model.m_algorithm != IVEModel::Algorithm::Codebook) {
        throw std::invalid_argument("Reset the IVE model before training a different background algorithm.");
    }
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
    if (bg_model.m_algorithm == IVEModel::Algorithm::Uninitialized) bg_model.clear_locked();

    IVE_HANDLE handle;
    if (RK_MPI_IVE_MatchBgModel(&handle, m_img_u8c1_in1.get(), &stBgModel, &stCtrl, RK_TRUE) != RK_SUCCESS) {
        bg_model.m_algorithm = IVEModel::Algorithm::Uninitialized;
        throw std::runtime_error("RK_MPI_IVE_MatchBgModel failed.");
    }
    bg_model.m_algorithm = IVEModel::Algorithm::Codebook;
}

std::tuple<ImageBuffer, ImageBuffer> IVE::update_bg_model(const ImageBuffer& current_img, IVEModel& bg_model, uint32_t frame_num) {
    std::lock_guard<std::mutex> lock(m_mutex);
    std::lock_guard<std::mutex> model_lock(bg_model.m_mutex);
    bg_model.validate_dimensions(current_img.width, current_img.height);
    if (bg_model.m_algorithm != IVEModel::Algorithm::Codebook) {
        throw std::invalid_argument("Train the codebook model before updating it.");
    }
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
        bg_model.m_algorithm = IVEModel::Algorithm::Uninitialized;
        throw std::runtime_error("RK_MPI_IVE_UpdateBgModel failed.");
    }
    return std::make_tuple(copy_from_ive_buffer(m_img_u8c1_out1.get(), current_img.width, current_img.height),
                           copy_from_ive_buffer(m_img_u8c1_out2.get(), current_img.width, current_img.height));
}

std::tuple<ImageBuffer, ImageBuffer> IVE::sad(const ImageBuffer& src1, const ImageBuffer& src2, IVE_SAD_MODE_E mode, uint16_t threshold, uint8_t min_val, uint8_t max_val) {
    return sad_config(src1, src2, mode, threshold, min_val, max_val,
                      IVE_SAD_OUT_MODE_BOTH, IVE_SAD_OUT_16_BITS);
}

std::tuple<ImageBuffer, ImageBuffer> IVE::sad_config(
        const ImageBuffer& src1, const ImageBuffer& src2, IVE_SAD_MODE_E mode,
        uint16_t threshold, uint8_t min_val, uint8_t max_val,
        IVE_SAD_OUT_MODE_E out_mode, IVE_SAD_OUT_BITS_E out_bits) {
    if (src1.format != visiong::kGray8Format || src2.format != visiong::kGray8Format) {
        throw std::invalid_argument("SAD requires GRAY8 input images.");
    }
    if (src1.width != src2.width || src1.height != src2.height) throw std::invalid_argument("SAD images must be same size.");
    if (mode < IVE_SAD_MODE_MB_4X4 || mode >= IVE_SAD_MODE_BUTT) throw std::invalid_argument("Invalid SAD block mode.");
    if (out_mode < IVE_SAD_OUT_MODE_SAD || out_mode >= IVE_SAD_OUT_MODE_BUTT) throw std::invalid_argument("Invalid SAD output mode.");
    if (out_bits != IVE_SAD_OUT_8_BITS && out_bits != IVE_SAD_OUT_16_BITS) throw std::invalid_argument("SAD output must use 8 or 16 bits.");
    const int block_size = 4 << static_cast<int>(mode);
    if (src1.width < block_size || src1.height < block_size ||
        src1.width % block_size || src1.height % block_size) {
        throw std::invalid_argument("SAD input dimensions must be multiples of the selected block size.");
    }
    const int out_w = src1.width / block_size;
    const int out_h = src1.height / block_size;
    ImageBuffer sad_image;
    ImageBuffer raw_sad;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        ensure_buffers(src1.width, src1.height);
        copy_to_ive_buffer(src1, m_img_u8c1_in1.get());
        copy_to_ive_buffer(src2, m_img_u8c1_in2.get());
        const auto compute_sad = [&](IVE_SAD_OUT_BITS_E bits) {
            IVE_IMAGE_S output{};
            if (create_image(&output, bits == IVE_SAD_OUT_16_BITS ? IVE_IMAGE_TYPE_U16C1 : IVE_IMAGE_TYPE_U8C1,
                             out_w, out_h) != RK_SUCCESS) {
                throw std::runtime_error("IVE SAD: failed to allocate an output image.");
            }
            IveMmzGuard guard;
            guard.set(output.au64PhyAddr[0], output.au64VirAddr[0]);
            {
                IveCpuAccess access(output.au64VirAddr[0]);
                const size_t bytes_per_pixel = bits == IVE_SAD_OUT_16_BITS ? 2 : 1;
                std::memset(reinterpret_cast<void*>(output.au64VirAddr[0]), 0,
                            static_cast<size_t>(output.au32Stride[0]) * output.u32Height * bytes_per_pixel);
            }
            IVE_SAD_CTRL_S control{};
            control.enMode = mode;
            control.enOutMode = IVE_SAD_OUT_MODE_SAD;
            control.enOutBits = bits;
            IVE_HANDLE handle{};
            const RK_S32 ret = RK_MPI_IVE_SAD(&handle, m_img_u8c1_in1.get(), m_img_u8c1_in2.get(),
                                             &output, nullptr, &control, true);
            if (ret != RK_SUCCESS) throw std::runtime_error("RK_MPI_IVE_SAD failed: " + std::to_string(ret));
            return copy_from_ive_buffer(&output, out_w, out_h);
        };
        // The SDK aliases its SAD and threshold descriptors; submit each operation separately.
        // SDK 将 SAD 与阈值输出描述符写入同一位置，因此分别提交运算。
        if (out_mode != IVE_SAD_OUT_MODE_SAD || out_bits == IVE_SAD_OUT_16_BITS) {
            raw_sad = compute_sad(IVE_SAD_OUT_16_BITS);
        }
        if (out_mode != IVE_SAD_OUT_MODE_THRESH) {
            sad_image = out_bits == IVE_SAD_OUT_16_BITS ? raw_sad : compute_sad(IVE_SAD_OUT_8_BITS);
        }
    }
    ImageBuffer threshold_image;
    if (out_mode != IVE_SAD_OUT_MODE_SAD) {
        threshold_image = threshold_u16(raw_sad, threshold, threshold,
                IVE_THRESH_U16_MODE_U16_TO_U8_MIN_MID_MAX, min_val, min_val, max_val);
    }
    return std::make_tuple(std::move(sad_image), std::move(threshold_image));
}

std::vector<ImageBuffer> IVE::create_pyramid(const ImageBuffer& src, int levels) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (src.format != visiong::kGray8Format) throw std::invalid_argument("Pyramid creation requires a GRAY8 source image.");
    if (levels <= 1 || levels > 4) throw std::invalid_argument("Pyramid levels must be 2-4.");
    if (!src.is_valid() || src.width <= 0 || src.height <= 0 || src.width > 4096 || src.height > 4096 ||
        (src.width >> (levels - 1)) < 8 || (src.height >> (levels - 1)) < 8) {
        throw std::invalid_argument("Pyramid dimensions must be in [1,4096] with every requested level at least 8x8.");
    }

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
        inData.u32Width = pyr_images[k + 1]->u32Width * 2;
        inData.u32Height = pyr_images[k + 1]->u32Height * 2;
        
        s32Ret = RK_MPI_IVE_DMA(&handle, &inData, &outData, &dma_ctrl, RK_TRUE);
        if (s32Ret != RK_SUCCESS) {
            throw std::runtime_error("IVE Pyramid: DMA failed at level " + std::to_string(k));
        }
        // Keep the destination stride supplied to DMA for the following filter.
        // 后续滤波沿用传给 DMA 的目标步长。
        
        // 3) Export next level. / 3）导出下一层。
        pyramid_imgs.push_back(copy_from_ive_buffer(pyr_images[k + 1].get()));
    }
    
    return pyramid_imgs;
}

