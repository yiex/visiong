// SPDX-License-Identifier: LGPL-3.0-or-later
#include "visiong/modules/IVE.h"
#include "visiong/core/RgaHelper.h"
#include "core/internal/rga_utils.h"
#include "visiong/common/pixel_format.h"
#include "core/internal/logger.h"
#include "core/internal/runtime_init.h"
#include "modules/internal/ive_memory.h"

#include <algorithm> // for std::max
#include <atomic>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
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

    void set(RK_U64 p, RK_U64 v) { phy = p; vir = v; }
};

struct PersistentMmzBuffer {
    RK_U64 phy = 0;
    RK_U64 vir = 0;
    RK_U32 size = 0;

    ~PersistentMmzBuffer() { reset(); }

    void reset() {
        if (vir) {
            free_mmz(phy, vir);
            phy = 0;
            vir = 0;
            size = 0;
        }
    }

    bool ensure_size(RK_U32 required_size) {
        if (vir && size >= required_size) {
            return true;
        }
        reset();
        if (mmz_alloc(&phy, reinterpret_cast<void**>(&vir), required_size) != RK_SUCCESS) {
            return false;
        }
        size = required_size;
        return true;
    }
};

struct CscMmzCache {
    PersistentMmzBuffer src;
    PersistentMmzBuffer dst;
};

thread_local CscMmzCache g_csc_mmz_cache;

struct CscModeTraits {
    bool yuv_input = false;
    bool yuv_output = false;
};

constexpr int kFiveByFiveMaskArea = 25;
constexpr int kMorphMaskWidth = 5;

std::atomic<bool> g_ive_log_enabled{false};
std::mutex g_ive_runtime_mutex;
int g_ive_runtime_refcount = 0;

void retain_ive_runtime() {
    std::lock_guard<std::mutex> lock(g_ive_runtime_mutex);
    if (g_ive_runtime_refcount > 0) {
        ++g_ive_runtime_refcount;
        return;
    }

    if (!visiong_init_sys_if_needed()) {
        throw std::runtime_error("IVE constructor failed: Could not initialize RK MPI system.");
    }
    if (RK_MPI_IVE_Init() != RK_SUCCESS) {
        throw std::runtime_error("Failed to initialize Rockchip IVE.");
    }
    g_ive_runtime_refcount = 1;
}

void release_ive_runtime() {
    std::lock_guard<std::mutex> lock(g_ive_runtime_mutex);
    if (g_ive_runtime_refcount <= 0) {
        g_ive_runtime_refcount = 0;
        return;
    }
    --g_ive_runtime_refcount;
    if (g_ive_runtime_refcount == 0) {
        RK_MPI_IVE_Deinit();
    }
}

void init_ive_log_from_env() {
    static bool inited = false;
    if (inited) return;
    const char* env = std::getenv("VISIONG_IVE_LOG");
    if (env && (env[0] == '1' || env[0] == 'y' || env[0] == 'Y')) {
        g_ive_log_enabled.store(true);
    }
    inited = true;
}

void flush_mmz_cache_by_vir_addr(RK_U64 vir_addr, bool cpu_to_device) {
    MB_BLK blk = RK_MPI_MB_VirAddr2Handle(reinterpret_cast<void*>(vir_addr));
    if (blk != MB_INVALID_HANDLE) {
        RK_MPI_SYS_MmzFlushCache(blk, cpu_to_device ? RK_TRUE : RK_FALSE);
    }
}

void copy_rows_to_mmz(
    RK_U64 dst_base,
    int dst_stride_bytes,
    const unsigned char* src_base,
    int src_stride_bytes,
    int row_count,
    int row_bytes) {
    for (int row = 0; row < row_count; ++row) {
        std::memcpy(
            reinterpret_cast<void*>(dst_base + static_cast<RK_U64>(row) * dst_stride_bytes),
            src_base + static_cast<size_t>(row) * src_stride_bytes,
            row_bytes);
    }
}

void fill_morph_mask(RK_U8 (&mask)[kFiveByFiveMaskArea], int kernel_size) {
    std::memset(mask, 0, sizeof(mask));
    const int radius = kernel_size / 2;
    for (int y = -radius; y <= radius; ++y) {
        for (int x = -radius; x <= radius; ++x) {
            const int row = 2 + y;
            const int col = 2 + x;
            mask[row * kMorphMaskWidth + col] = 255;
        }
    }
}

CscModeTraits get_csc_mode_traits(IVE_CSC_MODE_E mode) {
    const int mode_val = static_cast<int>(mode);
    CscModeTraits traits;
    traits.yuv_input = (mode_val <= 0x1F); // YUV2RGB / YUV2HSV
    traits.yuv_output = (mode_val >= 0x20 && mode_val <= 0x23); // RGB2YUV
    return traits;
}

void validate_csc_source_format(const ImageBuffer& src, const CscModeTraits& traits) {
    if (traits.yuv_input) {
        if (src.format != RK_FMT_YUV420SP && src.format != RK_FMT_YUV420SP_VU) {
            throw std::invalid_argument("IVE CSC expects YUV420SP input for selected mode.");
        }
        return;
    }

    if (src.format != RK_FMT_RGB888 && src.format != RK_FMT_BGR888) {
        throw std::invalid_argument("IVE CSC expects RGB888/BGR888 input for selected mode.");
    }
}

template <typename T>
void bind_yuv420sp_second_plane(T* image, int stride, int height) {
    image->au32Stride[1] = stride;
    image->au64PhyAddr[1] = image->au64PhyAddr[0] + static_cast<RK_U64>(stride) * height;
    image->au64VirAddr[1] = image->au64VirAddr[0] + static_cast<RK_U64>(stride) * height;
}

ImageBuffer convert_from_yuv420sp(
    IVE& ive,
    const ImageBuffer& src,
    bool full_range,
    IVE_CSC_MODE_E full_mode,
    IVE_CSC_MODE_E limit_mode,
    const char* api_name) {
    if (src.format != RK_FMT_YUV420SP && src.format != RK_FMT_YUV420SP_VU) {
        throw std::invalid_argument(std::string(api_name) + " requires YUV420SP or YUV420SP_VU input.");
    }
    return ive.csc(src, full_range ? full_mode : limit_mode);
}

ImageBuffer convert_from_rgb_pack3(
    IVE& ive,
    const ImageBuffer& src,
    bool full_range,
    IVE_CSC_MODE_E full_mode,
    IVE_CSC_MODE_E limit_mode,
    const char* api_name) {
    if (src.format != RK_FMT_RGB888 && src.format != RK_FMT_BGR888) {
        throw std::invalid_argument(std::string(api_name) + " requires RGB888 or BGR888 input.");
    }
    return ive.csc(src, full_range ? full_mode : limit_mode);
}
} // namespace

// Local pseudo pixel formats for typed IVE outputs. / Local pseudo 像素格式s 用于 typed IVE 输出s.
#ifndef RK_FMT_S16C1
#define RK_FMT_S16C1 static_cast<PIXEL_FORMAT_E>(0x1001)
#endif
#ifndef RK_FMT_U16C1
#define RK_FMT_U16C1 static_cast<PIXEL_FORMAT_E>(0x1002)
#endif
#ifndef RK_FMT_U64C1
#define RK_FMT_U64C1 static_cast<PIXEL_FORMAT_E>(0x1003)
#endif

static PIXEL_FORMAT_E ive_type_to_visiong_format(IVE_IMAGE_TYPE_E type) {
    switch (type) {
        case IVE_IMAGE_TYPE_U8C1: return visiong::kGray8Format;
        case IVE_IMAGE_TYPE_S16C1: return RK_FMT_S16C1;
        case IVE_IMAGE_TYPE_U16C1: return RK_FMT_U16C1;
        case IVE_IMAGE_TYPE_U64C1: return RK_FMT_U64C1;
        default: return static_cast<PIXEL_FORMAT_E>(type);
    }
}
static int get_bytes_per_pixel_for_ive_type(IVE_IMAGE_TYPE_E type) {
    switch (type) {
        case IVE_IMAGE_TYPE_U8C1: case IVE_IMAGE_TYPE_S8C1: return 1;
        case IVE_IMAGE_TYPE_S16C1: case IVE_IMAGE_TYPE_U16C1: return 2;
        case IVE_IMAGE_TYPE_S32C1: case IVE_IMAGE_TYPE_U32C1: return 4;
        case IVE_IMAGE_TYPE_S64C1: case IVE_IMAGE_TYPE_U64C1: return 8;
        default: return 1;
    }
}

// --- IVEModel Implementation --- / --- IVEModel 实现 ---
IVEModel::IVEModel(int width, int height, int model_size) {
    if (model_size <= 0) {
        mem_info.u32Size = width * height * 20; // Guesstimate for GMM
    } else {
        mem_info.u32Size = model_size;
    }
    if (create_mem_info(&mem_info, mem_info.u32Size) != RK_SUCCESS) {
        throw std::runtime_error("Failed to allocate memory for IVEModel.");
    }
    memset((void*)mem_info.u64VirAddr, 0, mem_info.u32Size);
}
IVEModel::~IVEModel() {
    if (mem_info.u64VirAddr != 0) {
        mmz_free((void*)mem_info.u64VirAddr);
    }
}

// --- IVE Singleton Implementation --- / --- IVE 单例实现 ---
IVE& IVE::get_instance() {
    init_ive_log_from_env();
    thread_local std::unique_ptr<IVE> instance;
    if (!instance) {
        instance.reset(new IVE());
    }
    return *instance;
}

void IVE::set_log_enabled(bool enabled) {
    g_ive_log_enabled.store(enabled);
}

bool IVE::is_log_enabled() {
    return g_ive_log_enabled.load();
}

IVE::IVE() {
    retain_ive_runtime();
    if (g_ive_log_enabled.load()) {
        VISIONG_LOG_INFO("IVE", "IVE engine instance created for current thread.");
    }
}

IVE::~IVE() {
    release_buffers();
    release_ive_runtime();
    if (g_ive_log_enabled.load()) {
        VISIONG_LOG_INFO("IVE", "IVE engine instance resources released.");
    }
}

void IVE::release_buffers() {
    if (m_width == 0 && m_height == 0) return;
    auto free_image = [](std::unique_ptr<IVE_IMAGE_S>& p) {
        if (p) {
            free_mmz(p->au64PhyAddr[0], p->au64VirAddr[0]);
            p.reset();
        }
    };
    auto free_mem = [](std::unique_ptr<IVE_MEM_INFO_S>& p) {
        if (p) {
            free_mmz(p->u64PhyAddr, p->u64VirAddr);
            p.reset();
        }
    };
    free_image(m_img_u8c1_in1);
    free_image(m_img_u8c1_in2);
    free_image(m_img_u8c1_out1);
    free_image(m_img_u8c1_out2);
    free_image(m_img_s16c1_out1);
    free_image(m_img_s16c1_out2);
    free_image(m_img_u16c1_in1);
    free_image(m_img_u64c1_out1);
    free_mem(m_mem_info);
    free_mem(m_lk_pts_mem);
    free_mem(m_lk_mv_mem);
    free_mem(m_st_candi_corner_mem);
    free_mem(m_st_corner_info_mem);
    free_mem(m_st_ctrl_mem);

    m_width = 0;
    m_height = 0;
    if (g_ive_log_enabled.load()) {
        VISIONG_LOG_INFO("IVE", "IVE buffers released.");
    }
}
void IVE::create_and_check_buffer(std::unique_ptr<IVE_IMAGE_S>& img, IVE_IMAGE_TYPE_E type, int width, int height, const std::string& name) {
    img = std::make_unique<IVE_IMAGE_S>();
    if (create_image(img.get(), type, width, height) != RK_SUCCESS) {
        throw std::runtime_error("IVE Failed to create buffer: " + name);
    }
}

// Called with m_mutex held by the public API; no nested locking here. / 调用到这里时 public API 已持有 m_mutex，因此此处不再做嵌套加锁。
void IVE::ensure_buffers(int width, int height) {
    if (m_width == width && m_height == height) {
        return;
    }

    release_buffers();

    m_width = width;
    m_height = height;

    try {
        // 1) Shared image workspaces. / 1）共享图像工作区。
        create_and_check_buffer(m_img_u8c1_in1, IVE_IMAGE_TYPE_U8C1, m_width, m_height, "u8c1_in1");
        create_and_check_buffer(m_img_u8c1_in2, IVE_IMAGE_TYPE_U8C1, m_width, m_height, "u8c1_in2");
        create_and_check_buffer(m_img_u8c1_out1, IVE_IMAGE_TYPE_U8C1, m_width, m_height, "u8c1_out1");
        create_and_check_buffer(m_img_u8c1_out2, IVE_IMAGE_TYPE_U8C1, m_width, m_height, "u8c1_out2");
        create_and_check_buffer(m_img_s16c1_out1, IVE_IMAGE_TYPE_S16C1, m_width, m_height, "s16c1_out1");
        create_and_check_buffer(m_img_s16c1_out2, IVE_IMAGE_TYPE_S16C1, m_width, m_height, "s16c1_out2");
        create_and_check_buffer(m_img_u16c1_in1, IVE_IMAGE_TYPE_U16C1, m_width, m_height, "u16c1_in1");
        create_and_check_buffer(m_img_u64c1_out1, IVE_IMAGE_TYPE_U64C1, m_width, m_height, "u64c1_out1");

        // 2) Shared metadata workspace. / 2）共享元数据工作区。
        m_mem_info = std::make_unique<IVE_MEM_INFO_S>();
        size_t mem_size = std::max({
            (size_t)m_width * m_height * sizeof(RK_U32) + sizeof(IVE_CANNY_STACK_SIZE_S),
            (size_t)sizeof(IVE_CCBLOB_S),
            (size_t)sizeof(IVE_NCC_DST_MEM_S)
        });
        if (create_mem_info(m_mem_info.get(), mem_size) != RK_SUCCESS) {
            throw std::runtime_error("IVE Failed to create m_mem_info");
        }

        // 3. LK optical-flow temporary buffers. / 3）LK 光流临时缓冲区。
        const int max_lk_points = LK_OPTICAL_FLOW_MAX_POINT_NUM;
        m_lk_pts_mem = std::make_unique<IVE_MEM_INFO_S>();
        if (create_mem_info(m_lk_pts_mem.get(), max_lk_points * sizeof(IVE_POINT_U16_S)) != RK_SUCCESS) {
            throw std::runtime_error("IVE Failed to create lk_pts_mem");
        }
        m_lk_mv_mem = std::make_unique<IVE_MEM_INFO_S>();
        if (create_mem_info(m_lk_mv_mem.get(), max_lk_points * sizeof(IVE_MV_S16_S)) != RK_SUCCESS) {
            throw std::runtime_error("IVE Failed to create lk_mv_mem");
        }

        // 4) Shi-Tomasi temporary buffers. / 4）Shi-Tomasi 临时缓冲区。
        m_st_candi_corner_mem = std::make_unique<IVE_MEM_INFO_S>();
        // Candidate buffer size follows the vendor sample contract. / 候选缓冲区大小遵循厂商示例约定。
        size_t candi_size = sizeof(IVE_ST_CANDI_STACK_SIZE_S) + (size_t)width * height * sizeof(RK_U16); 
        if (create_mem_info(m_st_candi_corner_mem.get(), candi_size) != RK_SUCCESS) {
            throw std::runtime_error("IVE Failed to create st_candi_corner_mem");
        }
        m_st_corner_info_mem = std::make_unique<IVE_MEM_INFO_S>();
        if (create_mem_info(m_st_corner_info_mem.get(), sizeof(IVE_ST_CORNER_INFO_S)) != RK_SUCCESS) {
            throw std::runtime_error("IVE Failed to create st_corner_info_mem");
        }
        
        // 4) Internal ST-corner control workspace. / 4）内部 ST 角点控制工作区。
        m_st_ctrl_mem = std::make_unique<IVE_MEM_INFO_S>();
        size_t st_ctrl_mem_size = (size_t)width * height + sizeof(IVE_ST_CORNER_MEM_S) * 2;
        if (create_mem_info(m_st_ctrl_mem.get(), st_ctrl_mem_size) != RK_SUCCESS) {
            throw std::runtime_error("IVE Failed to create st_ctrl_mem");
        }

    } catch (...) {
        release_buffers();
        throw;
    }
    
    if (g_ive_log_enabled.load()) {
        VISIONG_LOG_INFO("IVE", "IVE buffers (re)allocated for " << m_width << "x" << m_height);
    }
}

void IVE::copy_to_ive_buffer(const ImageBuffer& src, IVE_IMAGE_S* dst) {
    if (!src.is_valid()) {
        throw std::runtime_error("Cannot copy from an invalid ImageBuffer.");
    }
    if ((int)src.width != (int)dst->u32Width || (int)src.height != (int)dst->u32Height) {
        throw std::runtime_error("Image dimension mismatch in copy_to_ive_buffer.");
    }
    
    int item_size = get_bytes_per_pixel_for_ive_type(dst->enType);
    copy_data_with_stride(
        (void*)dst->au64VirAddr[0],
        dst->au32Stride[0] * item_size,
        src.get_data(),
        src.w_stride * item_size,
        src.height,
        src.width * item_size
    );

    flush_mmz_cache_by_vir_addr(dst->au64VirAddr[0], true);
}

ImageBuffer IVE::copy_from_ive_buffer(const IVE_IMAGE_S* src, int expected_w, int expected_h) {
    if (!src || !src->au64VirAddr[0]) return ImageBuffer();
    
    mmz_flush_start((void*)src->au64VirAddr[0]);
    
    PIXEL_FORMAT_E out_format = ive_type_to_visiong_format(src->enType);
    int item_size_bytes = get_bytes_per_pixel_for_ive_type(src->enType);
    
    int ive_w = static_cast<int>(src->u32Width);
    int ive_h = static_cast<int>(src->u32Height);
    int ive_stride = static_cast<int>(src->au32Stride[0]);
    
    int w = (expected_w > 0 && expected_h > 0) ? expected_w : ive_w;
    int h = (expected_w > 0 && expected_h > 0) ? expected_h : ive_h;

    if (ive_stride > 0 && ive_stride < w) {
        w = ive_stride;
    }

    size_t data_size = static_cast<size_t>(w) * h * item_size_bytes;
    std::vector<unsigned char> data(data_size);
    int src_stride_bytes = ive_stride * item_size_bytes;
    
    copy_data_from_stride(data.data(), (const void*)src->au64VirAddr[0], w * item_size_bytes, h, src_stride_bytes);
    
    mmz_flush_end((void*)src->au64VirAddr[0]);

    return ImageBuffer(w, h, out_format, std::move(data));
}

ImageBuffer IVE::morph_op(const ImageBuffer& src, bool is_dilate, int kernel_size) {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_buffers(src.width, src.height);
    copy_to_ive_buffer(src, m_img_u8c1_in1.get());

    RK_U8 mask[kFiveByFiveMaskArea];
    fill_morph_mask(mask, kernel_size);

    IVE_HANDLE handle;
    if (is_dilate) {
        IVE_DILATE_CTRL_S ctrl;
        std::memset(&ctrl, 0, sizeof(ctrl));
        std::memcpy(ctrl.au8Mask, mask, sizeof(mask));
        if (RK_MPI_IVE_Dilate(&handle, m_img_u8c1_in1.get(), m_img_u8c1_out1.get(), &ctrl, RK_TRUE) != RK_SUCCESS) {
            throw std::runtime_error("RK_MPI_IVE_Dilate failed.");
        }
    } else {
        IVE_ERODE_CTRL_S ctrl;
        std::memset(&ctrl, 0, sizeof(ctrl));
        std::memcpy(ctrl.au8Mask, mask, sizeof(mask));
        if (RK_MPI_IVE_Erode(&handle, m_img_u8c1_in1.get(), m_img_u8c1_out1.get(), &ctrl, RK_TRUE) != RK_SUCCESS) {
            throw std::runtime_error("RK_MPI_IVE_Erode failed.");
        }
    }
    return copy_from_ive_buffer(m_img_u8c1_out1.get(), m_width, m_height);
}

// --- Public Methods --- / --- 公共方法 ---
ImageBuffer IVE::filter(const ImageBuffer& src, const std::vector<int8_t>& mask) {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_buffers(src.width, src.height);
    if (src.format != visiong::kGray8Format) throw std::invalid_argument("filter requires a GRAY8 input.");
    if (mask.size() != 25) throw std::invalid_argument("filter mask must have 25 elements for a 5x5 kernel.");
    
    copy_to_ive_buffer(src, m_img_u8c1_in1.get());
    
    IVE_FILTER_CTRL_S ctrl;
    std::memset(&ctrl, 0, sizeof(ctrl));
    ctrl.u8OutMode = 0; // U8 output
    memcpy(ctrl.as8Mask, mask.data(), 25);

    // Compute normalization from kernel sum. / 根据卷积核求和结果计算归一化系数。
    int norm_sum = std::accumulate(mask.begin(), mask.end(), 0);

    uint8_t calculated_norm = 0;
    if (norm_sum > 1) {
        calculated_norm = static_cast<uint8_t>(std::ceil(std::log2(norm_sum)));
    }
    ctrl.u8Norm = calculated_norm;
    
    IVE_HANDLE handle;
    if (RK_MPI_IVE_Filter(&handle, m_img_u8c1_in1.get(), m_img_u8c1_out1.get(), &ctrl, RK_TRUE) != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_Filter failed.");
    }
    
    return copy_from_ive_buffer(m_img_u8c1_out1.get(), m_width, m_height);
}

std::tuple<ImageBuffer, ImageBuffer> IVE::sobel(const ImageBuffer& src, IVE_SOBEL_OUT_CTRL_E out_ctrl, IVE_IMAGE_TYPE_E out_format) {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_buffers(src.width, src.height);
    if (src.format != visiong::kGray8Format) throw std::invalid_argument("sobel requires a GRAY8 input.");
    if (out_format != IVE_IMAGE_TYPE_S16C1 && out_format != IVE_IMAGE_TYPE_U8C1) throw std::invalid_argument("Sobel output format must be S16C1 or U8C1");
    copy_to_ive_buffer(src, m_img_u8c1_in1.get());
    
    IVE_SOBEL_CTRL_S ctrl; 
    memset(&ctrl, 0, sizeof(ctrl)); 
    ctrl.u8OutCtrl = out_ctrl;
    
    // Configure output data mode. / 配置输出数据模式。
    // u8OutMode: 0=RK_U8, 1=RK_S8, 2=RK_U16, 3=RK_S16.
    if (out_format == IVE_IMAGE_TYPE_S16C1) {
        ctrl.u8OutMode = 3; // S16 output
    } else {
        ctrl.u8OutMode = 0; // U8 output
    }
    
    // 3x3 Sobel kernel embedded in the center of 5x5 mask. / 将 3x3 Sobel 卷积核嵌入到 5x5 掩码中心。
    RK_S8 mask[25] = {
        0,  0,  0,  0,  0,
        0, -1,  0,  1,  0,
        0, -2,  0,  2,  0,
        0, -1,  0,  1,  0,
        0,  0,  0,  0,  0
    };
    memcpy(ctrl.as8Mask, mask, sizeof(mask));
    
    IVE_HANDLE handle;
    IVE_DST_IMAGE_S* dst_h = (out_format == IVE_IMAGE_TYPE_S16C1) ? m_img_s16c1_out1.get() : m_img_u8c1_out1.get();
    IVE_DST_IMAGE_S* dst_v = (out_format == IVE_IMAGE_TYPE_S16C1) ? m_img_s16c1_out2.get() : m_img_u8c1_out2.get();
    if (RK_MPI_IVE_Sobel(&handle, m_img_u8c1_in1.get(), dst_h, (out_ctrl == IVE_SOBEL_OUT_CTRL_BOTH ? dst_v : nullptr), &ctrl, RK_TRUE) != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_Sobel failed.");
    }
    ImageBuffer result_h = copy_from_ive_buffer(dst_h, m_width, m_height);
    ImageBuffer result_v = (out_ctrl == IVE_SOBEL_OUT_CTRL_BOTH) ? copy_from_ive_buffer(dst_v, m_width, m_height) : ImageBuffer();
    return std::make_tuple(std::move(result_h), std::move(result_v));
}

ImageBuffer IVE::canny(const ImageBuffer& src, uint16_t high_thresh, uint16_t low_thresh) {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_buffers(src.width, src.height);
    if (src.format != visiong::kGray8Format) throw std::invalid_argument("canny requires a GRAY8 input.");
    copy_to_ive_buffer(src, m_img_u8c1_in1.get());
    
    IVE_CANNY_EDGE_CTRL_S ctrl; 
    memset(&ctrl, 0, sizeof(ctrl)); 
    ctrl.stMem = *m_mem_info.get(); 
    ctrl.u16HighThr = high_thresh; 
    ctrl.u16LowThr = low_thresh;
    
    // Use a 3x3 Sobel gradient kernel embedded in 5x5 mask layout. / 在 5x5 掩码布局中使用嵌入式 3x3 Sobel 梯度核。
    RK_S8 mask[25] = {
        0,  0,  0,  0,  0,
        0, -1,  0,  1,  0,
        0, -2,  0,  2,  0,
        0, -1,  0,  1,  0,
        0,  0,  0,  0,  0
    };
    memcpy(ctrl.as8Mask, mask, sizeof(mask));
    
    IVE_HANDLE handle;
    if (RK_MPI_IVE_CannyEdge(&handle, m_img_u8c1_in1.get(), m_img_u8c1_out1.get(), m_mem_info.get(), &ctrl, RK_TRUE) != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_CannyEdge failed.");
    }
    return copy_from_ive_buffer(m_img_u8c1_out1.get(), m_width, m_height);
}

ImageBuffer IVE::mag_and_ang(const ImageBuffer& src, uint16_t threshold, bool return_magnitude) {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_buffers(src.width, src.height);
    if (src.format != visiong::kGray8Format) throw std::invalid_argument("mag_and_ang requires a GRAY8 input.");
    copy_to_ive_buffer(src, m_img_u8c1_in1.get());
    
    IVE_MAG_AND_ANG_CTRL_S ctrl; 
    memset(&ctrl, 0, sizeof(ctrl)); 
    ctrl.stMem = *m_mem_info.get(); 
    ctrl.u16Thr = threshold;
    
    // Use a 5x5 Scharr-like kernel as recommended for MagAndAng. / 按 MagAndAng 的建议使用 5x5 类 Scharr 卷积核。
    RK_S8 mask[25] = {
        -1, -2,  0,  2,  1,
        -4, -8,  0,  8,  4,
        -6, -12, 0, 12,  6,
        -4, -8,  0,  8,  4,
        -1, -2,  0,  2,  1
    };
    memcpy(ctrl.as8Mask, mask, sizeof(mask));
    
    IVE_HANDLE handle;
    if (RK_MPI_IVE_MagAndAng(&handle, m_img_u8c1_in1.get(), m_img_s16c1_out1.get(), m_img_u8c1_out1.get(), &ctrl, RK_TRUE) != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_MagAndAng failed.");
    }
    return copy_from_ive_buffer(return_magnitude ? (IVE_IMAGE_S*)m_img_s16c1_out1.get() : (IVE_IMAGE_S*)m_img_u8c1_out1.get(), m_width, m_height);
}

ImageBuffer IVE::dilate(const ImageBuffer& src, int kernel_size) {
    if (src.format != visiong::kGray8Format) throw std::invalid_argument("dilate requires a GRAY8 input.");
    if (kernel_size != 3 && kernel_size != 5) throw std::invalid_argument("Kernel size must be 3 or 5.");
    return morph_op(src, true, kernel_size);
}

ImageBuffer IVE::erode(const ImageBuffer& src, int kernel_size) {
    if (src.format != visiong::kGray8Format) throw std::invalid_argument("erode requires a GRAY8 input.");
    if (kernel_size != 3 && kernel_size != 5) throw std::invalid_argument("Kernel size must be 3 or 5.");
    return morph_op(src, false, kernel_size);
}

ImageBuffer IVE::ordered_stat_filter(const ImageBuffer& src, IVE_ORD_STAT_FILTER_MODE_E mode) {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_buffers(src.width, src.height);
    if (src.format != visiong::kGray8Format) throw std::invalid_argument("ordered_stat_filter requires a GRAY8 input.");
    copy_to_ive_buffer(src, m_img_u8c1_in1.get());
    IVE_ORD_STAT_FILTER_CTRL_S ctrl; memset(&ctrl, 0, sizeof(ctrl)); ctrl.enMode = mode;
    IVE_HANDLE handle;
    if (RK_MPI_IVE_OrdStatFilter(&handle, m_img_u8c1_in1.get(), m_img_u8c1_out1.get(), &ctrl, RK_TRUE) != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_OrdStatFilter failed.");
    }
    return copy_from_ive_buffer(m_img_u8c1_out1.get(), m_width, m_height);
}

ImageBuffer IVE::logic_op(const ImageBuffer& src1, const ImageBuffer& src2, IVE_LOGICOP_MODE_E op) {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_buffers(src1.width, src1.height);
    if (src1.width != src2.width || src1.height != src2.height) throw std::invalid_argument("logic_op requires images of the same size.");
    copy_to_ive_buffer(src1, m_img_u8c1_in1.get());
    copy_to_ive_buffer(src2, m_img_u8c1_in2.get());
    IVE_HANDLE handle;
    RK_S32 ret;
    switch(op) {
        case IVE_LOGICOP_MODE_AND: ret = RK_MPI_IVE_And(&handle, m_img_u8c1_in1.get(), m_img_u8c1_in2.get(), m_img_u8c1_out1.get(), RK_TRUE); break;
        case IVE_LOGICOP_MODE_OR:  ret = RK_MPI_IVE_Or(&handle, m_img_u8c1_in1.get(), m_img_u8c1_in2.get(), m_img_u8c1_out1.get(), RK_TRUE); break;
        case IVE_LOGICOP_MODE_XOR: ret = RK_MPI_IVE_Xor(&handle, m_img_u8c1_in1.get(), m_img_u8c1_in2.get(), m_img_u8c1_out1.get(), RK_TRUE); break;
        default: throw std::invalid_argument("Unsupported logic operation.");
    }
    if (ret != RK_SUCCESS) throw std::runtime_error("Logical operation failed.");
    return copy_from_ive_buffer(m_img_u8c1_out1.get(), m_width, m_height);
}

ImageBuffer IVE::add(const ImageBuffer& src1, const ImageBuffer& src2) {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_buffers(src1.width, src1.height);
    if (src1.width != src2.width || src1.height != src2.height) throw std::invalid_argument("add requires images of the same size.");
    copy_to_ive_buffer(src1, m_img_u8c1_in1.get());
    copy_to_ive_buffer(src2, m_img_u8c1_in2.get());
    
    // IVE_ADD_CTRL_S: dst = u0q16X * src1 + u0q16Y * src2. / IVE_ADD_CTRL_S：dst = u0q16X * src1 + u0q16Y * src2。
    // U0.16 fixed-point scale: 65535 ~= 1.0, 32768 = 0.5. / U0.16 定点缩放：65535≈1.0，32768=0.5。
    // Use 0.5 / 0.5 for average blending.
    IVE_ADD_CTRL_S ctrl;
    memset(&ctrl, 0, sizeof(ctrl));
    ctrl.u0q16X = 32768;  // 0.5 in U0.16 format
    ctrl.u0q16Y = 32768;  // 0.5 in U0.16 format
    
    IVE_HANDLE handle;
    if (RK_MPI_IVE_Add(&handle, m_img_u8c1_in1.get(), m_img_u8c1_in2.get(), m_img_u8c1_out1.get(), &ctrl, RK_TRUE) != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_Add failed.");
    }
    return copy_from_ive_buffer(m_img_u8c1_out1.get(), m_width, m_height);
}

ImageBuffer IVE::sub(const ImageBuffer& src1, const ImageBuffer& src2, IVE_SUB_MODE_E mode) {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_buffers(src1.width, src1.height);
    if (src1.width != src2.width || src1.height != src2.height) throw std::invalid_argument("sub requires images of the same size.");
    copy_to_ive_buffer(src1, m_img_u8c1_in1.get());
    copy_to_ive_buffer(src2, m_img_u8c1_in2.get());

    IVE_SUB_CTRL_S ctrl;
    memset(&ctrl, 0, sizeof(ctrl));
    ctrl.enMode = mode;
    IVE_HANDLE handle;
    if (RK_MPI_IVE_Sub(&handle, m_img_u8c1_in1.get(), m_img_u8c1_in2.get(), m_img_u8c1_out1.get(), &ctrl, RK_TRUE) != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_Sub failed.");
    }
    return copy_from_ive_buffer(m_img_u8c1_out1.get(), m_width, m_height);
}

ImageBuffer IVE::threshold(const ImageBuffer& src, uint8_t low_thresh, uint8_t high_thresh, IVE_THRESH_MODE_E mode) {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_buffers(src.width, src.height);
    if (src.format != visiong::kGray8Format) throw std::invalid_argument("threshold requires a GRAY8 input.");
    copy_to_ive_buffer(src, m_img_u8c1_in1.get());
    IVE_THRESH_U8_CTRL_S ctrl; memset(&ctrl, 0, sizeof(ctrl)); ctrl.enMode = mode; ctrl.u8LowThr = low_thresh; ctrl.u8HighThr = high_thresh; ctrl.u8MinVal = 0; ctrl.u8MidVal = 0; ctrl.u8MaxVal = 255;
    IVE_HANDLE handle;
    if (RK_MPI_IVE_Thresh(&handle, m_img_u8c1_in1.get(), m_img_u8c1_out1.get(), &ctrl, RK_TRUE) != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_Thresh failed.");
    }
    return copy_from_ive_buffer(m_img_u8c1_out1.get(), m_width, m_height);
}

ImageBuffer IVE::cast_16bit_to_8bit(const ImageBuffer& src, IVE_16BIT_TO_8BIT_MODE_E mode) {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_buffers(src.width, src.height);
    if (src.format != RK_FMT_S16C1 && src.format != RK_FMT_U16C1) throw std::invalid_argument("cast_16bit_to_8bit requires an S16C1 or U16C1 input.");
    
    IVE_16BIT_TO_8BIT_CTRL_S ctrl; 
    memset(&ctrl, 0, sizeof(ctrl)); 
    ctrl.enMode = mode;
    
    // Tune scaling parameters following vendor sample behavior. / 按厂商示例行为调节缩放参数。
    // Vendor sample baseline: numerator=255, denominator=255*4=1020. / 厂商示例基线参数：numerator=255，denominator=255*4=1020。
    if (mode == IVE_16BIT_TO_8BIT_MODE_S16_TO_U8_ABS) {
        // |S16| -> U8 / 将 |S16| 映射为 U8。
        ctrl.u8Numerator = 255;
        ctrl.u16Denominator = 255 * 4;  // 1020
        ctrl.s8Bias = 0;
    } else if (mode == IVE_16BIT_TO_8BIT_MODE_S16_TO_U8_BIAS) {
        // Signed S16 -> U8 with midpoint bias near 128. / 带接近 128 中点偏置的有符号 S16 -> U8。
        ctrl.u8Numerator = 1;
        ctrl.u16Denominator = 4;
        ctrl.s8Bias = 127;
    } else if (mode == IVE_16BIT_TO_8BIT_MODE_U16_TO_U8) {
        // U16 -> U8 / 将 U16 映射为 U8。
        ctrl.u8Numerator = 255;
        ctrl.u16Denominator = 255 * 4;
        ctrl.s8Bias = 0;
    } else {
        // S16 -> S8 / 将 S16 映射为 S8。
        ctrl.u8Numerator = 1;
        ctrl.u16Denominator = 128;
        ctrl.s8Bias = 0;
    }
    
    IVE_SRC_IMAGE_S* src_ive = (src.format == RK_FMT_S16C1) ? (IVE_SRC_IMAGE_S*)m_img_s16c1_out1.get() : (IVE_SRC_IMAGE_S*)m_img_u16c1_in1.get();
    copy_to_ive_buffer(src, src_ive);
    
    IVE_HANDLE handle;
    if (RK_MPI_IVE_16BitTo8Bit(&handle, src_ive, m_img_u8c1_out1.get(), &ctrl, RK_TRUE) != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_16BitTo8Bit failed.");
    }
    return copy_from_ive_buffer(m_img_u8c1_out1.get(), m_width, m_height);
}

std::vector<uint32_t> IVE::hist(const ImageBuffer& src) {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_buffers(src.width, src.height);
    if (src.format != visiong::kGray8Format) throw std::invalid_argument("hist requires a GRAY8 input.");
    copy_to_ive_buffer(src, m_img_u8c1_in1.get());
    if (m_mem_info->u32Size < IVE_HIST_NUM * sizeof(RK_U32)) throw std::runtime_error("Pre-allocated memory is too small for histogram.");
    IVE_HANDLE handle;
    if (RK_MPI_IVE_Hist(&handle, m_img_u8c1_in1.get(), m_mem_info.get(), RK_TRUE) != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_Hist failed.");
    }
    mmz_flush_start((void*)m_mem_info->u64VirAddr);
    std::vector<uint32_t> hist_data(IVE_HIST_NUM);
    memcpy(hist_data.data(), (void*)m_mem_info->u64VirAddr, IVE_HIST_NUM * sizeof(RK_U32));
    mmz_flush_end((void*)m_mem_info->u64VirAddr);
    return hist_data;
}

ImageBuffer IVE::equalize_hist(const ImageBuffer& src) {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_buffers(src.width, src.height);
    if (src.format != visiong::kGray8Format) throw std::invalid_argument("equalize_hist requires a GRAY8 input.");
    copy_to_ive_buffer(src, m_img_u8c1_in1.get());
    
    IVE_EQHIST_CTRL_S ctrl; 
    memset(&ctrl, 0, sizeof(ctrl));
    // Enable equalization mode (not histogram-only mode). / 启用均衡化模式（而非仅直方图模式）。
    // IVE_EQUALIZE_MODE_EQHIST (0x2): perform histogram equalization.
    ctrl.enMode = IVE_EQUALIZE_MODE_EQHIST;
    
    IVE_HANDLE handle;
    if (RK_MPI_IVE_EqualizeHist(&handle, m_img_u8c1_in1.get(), m_img_u8c1_out1.get(), &ctrl, RK_TRUE) != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_EqualizeHist failed.");
    }
    return copy_from_ive_buffer(m_img_u8c1_out1.get(), m_width, m_height);
}

ImageBuffer IVE::integral(const ImageBuffer& src, IVE_INTEG_OUT_CTRL_E mode) {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_buffers(src.width, src.height);
    if (src.format != visiong::kGray8Format) throw std::invalid_argument("integral requires a GRAY8 input.");
    copy_to_ive_buffer(src, m_img_u8c1_in1.get());
    IVE_INTEG_CTRL_S ctrl; memset(&ctrl, 0, sizeof(ctrl)); ctrl.enOutCtrl = mode; ctrl.stMem = *m_mem_info.get();
    IVE_HANDLE handle;
    if (RK_MPI_IVE_Integ(&handle, m_img_u8c1_in1.get(), m_img_u64c1_out1.get(), &ctrl, RK_TRUE) != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_Integ failed.");
    }
    return copy_from_ive_buffer(m_img_u64c1_out1.get(), m_width, m_height);
}

std::vector<Blob> IVE::ccl(const ImageBuffer& src, int min_area) {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_buffers(src.width, src.height);
    if (src.format != visiong::kGray8Format) throw std::invalid_argument("ccl requires a GRAY8 (binarized) input.");
    copy_to_ive_buffer(src, m_img_u8c1_out1.get()); 
    if (m_mem_info->u32Size < sizeof(IVE_CCBLOB_S)) throw std::runtime_error("Pre-allocated memory is too small for CCL.");
    IVE_CCL_CTRL_S ctrl; memset(&ctrl, 0, sizeof(ctrl)); ctrl.enMode = IVE_CCL_MODE_8C; ctrl.stMem = *m_mem_info.get();
    IVE_HANDLE handle;
    if (RK_MPI_IVE_CCL(&handle, m_img_u8c1_out1.get(), m_mem_info.get(), &ctrl, RK_TRUE) != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_CCL failed.");
    }
    mmz_flush_start((void*)m_mem_info->u64VirAddr);
    IVE_CCBLOB_S* blob_info = (IVE_CCBLOB_S*)m_mem_info->u64VirAddr;
    std::vector<Blob> results;
    for (int i = 0; i < blob_info->u8RegionNum; ++i) {
        if ((int)blob_info->astRegion[i].u32Area > min_area) {
            results.emplace_back(blob_info->astRegion[i].u16Left, blob_info->astRegion[i].u16Top,
                                 blob_info->astRegion[i].u16Right - blob_info->astRegion[i].u16Left + 1,
                                 blob_info->astRegion[i].u16Bottom - blob_info->astRegion[i].u16Top + 1,
                                 0, 0, blob_info->astRegion[i].u32Area);
        }
    }
    mmz_flush_end((void*)m_mem_info->u64VirAddr);
    return results;
}

double IVE::ncc(const ImageBuffer& src1, const ImageBuffer& src2) {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_buffers(src1.width, src1.height);
    if (src1.width != src2.width || src1.height != src2.height) throw std::invalid_argument("ncc requires images of the same size.");
    copy_to_ive_buffer(src1, m_img_u8c1_in1.get());
    copy_to_ive_buffer(src2, m_img_u8c1_in2.get());
    if (m_mem_info->u32Size < sizeof(IVE_NCC_DST_MEM_S)) throw std::runtime_error("Pre-allocated memory is too small for NCC.");
    
    IVE_HANDLE handle;
    if (RK_MPI_IVE_NCC(&handle, m_img_u8c1_in1.get(), m_img_u8c1_in2.get(), m_mem_info.get(), RK_TRUE) != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_NCC failed.");
    }

    mmz_flush_start((void*)m_mem_info->u64VirAddr);
    IVE_NCC_DST_MEM_S* ncc_result = (IVE_NCC_DST_MEM_S*)m_mem_info->u64VirAddr;
    
    // Read NCC aggregates from shared memory. / 从共享内存中读取 NCC 聚合结果。
    double numerator = static_cast<double>(ncc_result->u64Numerator);
    double quad_sum1 = static_cast<double>(ncc_result->u64QuadSum1);
    double quad_sum2 = static_cast<double>(ncc_result->u64QuadSum2);
    
    mmz_flush_end((void*)m_mem_info->u64VirAddr);

    double denominator = sqrt(quad_sum1) * sqrt(quad_sum2);
    if (denominator < 1e-9) { // avoid unstable divide-by-zero path
        return 0.0;
    }

    // Final normalized NCC score. / 最终归一化 NCC 得分。
    return numerator / denominator;
}

ImageBuffer IVE::csc(const ImageBuffer& src, IVE_CSC_MODE_E mode) {
    std::lock_guard<std::mutex> lock(m_mutex);

    const CscModeTraits traits = get_csc_mode_traits(mode);
    std::unique_ptr<ImageBuffer> converted_src;
    const ImageBuffer* src_ptr = &src;
    if (traits.yuv_input && src.format == RK_FMT_YUV420SP_VU) {
        converted_src = std::make_unique<ImageBuffer>(src.to_format(RK_FMT_YUV420SP));
        src_ptr = converted_src.get();
    }
    validate_csc_source_format(*src_ptr, traits);

    const int w = src_ptr->width;
    const int h = src_ptr->height;
    const int stride = calc_stride(w, kImageAlign);
    const IVE_IMAGE_TYPE_E src_type = traits.yuv_input ? IVE_IMAGE_TYPE_YUV420SP : IVE_IMAGE_TYPE_U8C3_PACKAGE;
    const IVE_IMAGE_TYPE_E dst_type = traits.yuv_output ? IVE_IMAGE_TYPE_YUV420SP : IVE_IMAGE_TYPE_U8C3_PACKAGE;

    IVE_SRC_IMAGE_S src_ive;
    std::memset(&src_ive, 0, sizeof(src_ive));
    src_ive.enType = src_type;
    src_ive.u32Width = w;
    src_ive.u32Height = h;
    src_ive.au32Stride[0] = stride;

    const RK_U32 src_size = traits.yuv_input ? stride * h * 3 / 2 : stride * h * 3;
    if (!g_csc_mmz_cache.src.ensure_size(src_size)) {
        throw std::runtime_error("IVE CSC: failed to allocate source memory.");
    }
    src_ive.au64PhyAddr[0] = g_csc_mmz_cache.src.phy;
    src_ive.au64VirAddr[0] = g_csc_mmz_cache.src.vir;

    if (traits.yuv_input) {
        bind_yuv420sp_second_plane(&src_ive, stride, h);
    }

    const unsigned char* src_data = static_cast<const unsigned char*>(src_ptr->get_data());
    if (traits.yuv_input) {
        const int y_src_stride = src_ptr->w_stride;
        const int y_plane_size = y_src_stride * h;
        copy_rows_to_mmz(src_ive.au64VirAddr[0], stride, src_data, y_src_stride, h, w);
        copy_rows_to_mmz(src_ive.au64VirAddr[1], stride, src_data + y_plane_size, y_src_stride, h / 2, w);
    } else {
        const int src_row_bytes = src_ptr->w_stride * 3;
        const int dst_row_bytes = stride * 3;
        copy_rows_to_mmz(src_ive.au64VirAddr[0], dst_row_bytes, src_data, src_row_bytes, h, w * 3);
    }
    flush_mmz_cache_by_vir_addr(src_ive.au64VirAddr[0], true);

    IVE_DST_IMAGE_S dst_ive;
    std::memset(&dst_ive, 0, sizeof(dst_ive));
    dst_ive.enType = dst_type;
    dst_ive.u32Width = w;
    dst_ive.u32Height = h;
    dst_ive.au32Stride[0] = stride;

    const RK_U32 dst_size = traits.yuv_output ? stride * h * 3 / 2 : stride * h * 3;
    if (!g_csc_mmz_cache.dst.ensure_size(dst_size)) {
        throw std::runtime_error("IVE CSC: failed to allocate destination memory.");
    }
    dst_ive.au64PhyAddr[0] = g_csc_mmz_cache.dst.phy;
    dst_ive.au64VirAddr[0] = g_csc_mmz_cache.dst.vir;

    if (traits.yuv_output) {
        bind_yuv420sp_second_plane(&dst_ive, stride, h);
    }

    IVE_CSC_CTRL_S ctrl;
    std::memset(&ctrl, 0, sizeof(ctrl));
    ctrl.enMode = mode;

    IVE_HANDLE handle;
    if (RK_MPI_IVE_CSC(&handle, &src_ive, &dst_ive, &ctrl, RK_TRUE) != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_CSC failed.");
    }

    mmz_flush_start(reinterpret_cast<void*>(dst_ive.au64VirAddr[0]));

    const PIXEL_FORMAT_E out_format = traits.yuv_output ? RK_FMT_YUV420SP : RK_FMT_RGB888;
    std::vector<unsigned char> out_data(
        traits.yuv_output ? static_cast<size_t>(w) * h * 3 / 2 : static_cast<size_t>(w) * h * 3);

    if (traits.yuv_output) {
        copy_data_from_stride(out_data.data(), reinterpret_cast<void*>(dst_ive.au64VirAddr[0]), w, h, stride);
        copy_data_from_stride(
            out_data.data() + static_cast<size_t>(w) * h,
            reinterpret_cast<void*>(dst_ive.au64VirAddr[1]),
            w,
            h / 2,
            stride);
    } else {
        copy_data_from_stride(out_data.data(), reinterpret_cast<void*>(dst_ive.au64VirAddr[0]), w * 3, h, stride * 3);
    }

    mmz_flush_end(reinterpret_cast<void*>(dst_ive.au64VirAddr[0]));

    return ImageBuffer(w, h, out_format, std::move(out_data));
}

// Convenience wrappers around CSC modes. / 围绕 CSC 模式的便捷封装。
ImageBuffer IVE::yuv_to_rgb(const ImageBuffer& src, bool full_range) {
    return convert_from_yuv420sp(
        *this,
        src,
        full_range,
        IVE_CSC_MODE_FULL_BT601_YUV2RGB,
        IVE_CSC_MODE_LIMIT_BT601_YUV2RGB,
        "yuv_to_rgb");
}

ImageBuffer IVE::yuv_to_hsv(const ImageBuffer& src, bool full_range) {
    return convert_from_yuv420sp(
        *this,
        src,
        full_range,
        IVE_CSC_MODE_FULL_BT601_YUV2HSV,
        IVE_CSC_MODE_LIMIT_BT601_YUV2HSV,
        "yuv_to_hsv");
}

ImageBuffer IVE::rgb_to_yuv(const ImageBuffer& src, bool full_range) {
    return convert_from_rgb_pack3(
        *this,
        src,
        full_range,
        IVE_CSC_MODE_FULL_BT601_RGB2YUV,
        IVE_CSC_MODE_LIMIT_BT601_RGB2YUV,
        "rgb_to_yuv");
}

ImageBuffer IVE::rgb_to_hsv(const ImageBuffer& src, bool full_range) {
    return convert_from_rgb_pack3(
        *this,
        src,
        full_range,
        IVE_CSC_MODE_FULL_BT601_RGB2HSV,
        IVE_CSC_MODE_LIMIT_BT601_RGB2HSV,
        "rgb_to_hsv");
}

// --- FULLY PORTED NEW IVE FUNCTIONS (Corrected) --- / --- 已完整移植的新 IVE 函数（修正版）---
ImageBuffer IVE::dma(const ImageBuffer& src, IVE_DMA_MODE_E mode) {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_buffers(src.width, src.height);
    copy_to_ive_buffer(src, m_img_u8c1_in1.get());
    IVE_SRC_DATA_S stSrcData;
    IVE_DST_DATA_S stDstData;
    RK_MPI_IVE_CvtImageToData(m_img_u8c1_in1.get(), &stSrcData);
    RK_MPI_IVE_CvtImageToData(m_img_u8c1_out1.get(), &stDstData);
    IVE_DMA_CTRL_S stCtrl;
    memset(&stCtrl, 0, sizeof(stCtrl));
    stCtrl.enMode = mode;
    stCtrl.u64Val = 0;
    IVE_HANDLE handle;
    if (RK_MPI_IVE_DMA(&handle, &stSrcData, &stDstData, &stCtrl, RK_TRUE) != RK_SUCCESS) {
         throw std::runtime_error("RK_MPI_IVE_DMA failed.");
    }
    return copy_from_ive_buffer(m_img_u8c1_out1.get(), m_width, m_height);
}

ImageBuffer IVE::cast_8bit_to_8bit(const ImageBuffer& src, int8_t bias, uint8_t numerator, uint8_t denominator) {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_buffers(src.width, src.height);
    if (src.format != visiong::kGray8Format) throw std::invalid_argument("cast_8bit_to_8bit requires a GRAY8 input.");
    copy_to_ive_buffer(src, m_img_u8c1_in1.get());
    IVE_8BIT_TO_8BIT_CTRL_S ctrl;
    memset(&ctrl, 0, sizeof(ctrl));
    ctrl.enMode = IVE_8BIT_TO_8BIT_MODE_U8_TO_U8;
    ctrl.s8Bias = bias;
    ctrl.u8Numerator = numerator;
    ctrl.u8Denominator = denominator;
    IVE_HANDLE handle;
    if (RK_MPI_IVE_8BitTo8Bit(&handle, m_img_u8c1_in1.get(), m_img_u8c1_out1.get(), &ctrl, RK_TRUE) != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_8BitTo8Bit failed.");
    }
    return copy_from_ive_buffer(m_img_u8c1_out1.get(), m_width, m_height);
}

ImageBuffer IVE::map(const ImageBuffer& src, const std::vector<uint8_t>& lut) {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_buffers(src.width, src.height);
    if (src.format != visiong::kGray8Format) throw std::invalid_argument("map requires a GRAY8 input.");
    if (lut.size() != 256) throw std::invalid_argument("map requires a LUT with 256 elements.");
    copy_to_ive_buffer(src, m_img_u8c1_in1.get());
    
    // Reuse pre-allocated memory for LUT upload to avoid per-call allocation churn. / 复用预分配内存来上传 LUT，避免每次调用都发生分配抖动。
    if (m_mem_info->u32Size < 256) throw std::runtime_error("Pre-allocated memory is too small for LUT.");
    
    // Alias the first 256 bytes as LUT table storage. / 将前 256 字节别名为 LUT 表存储区。
    IVE_MEM_INFO_S stMap;
    stMap.u64PhyAddr = m_mem_info->u64PhyAddr;
    stMap.u64VirAddr = m_mem_info->u64VirAddr;
    stMap.u32Size = 256;
    
    memcpy((void*)stMap.u64VirAddr, lut.data(), 256 * sizeof(uint8_t));
    
    // Flush LUT cache so hardware sees the latest table. / 刷新 LUT 缓存，让硬件读取到最新表内容。
    flush_mmz_cache_by_vir_addr(stMap.u64VirAddr, true);
    
    IVE_MAP_CTRL_S ctrl; 
    memset(&ctrl, 0, sizeof(ctrl)); 
    ctrl.enMode = IVE_MAP_MODE_U8;
    
    IVE_HANDLE handle;
    if (RK_MPI_IVE_Map(&handle, m_img_u8c1_in1.get(), &stMap, m_img_u8c1_out1.get(), &ctrl, RK_TRUE) != RK_SUCCESS)
        throw std::runtime_error("RK_MPI_IVE_Map failed.");
    return copy_from_ive_buffer(m_img_u8c1_out1.get(), m_width, m_height);
}

