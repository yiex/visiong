// SPDX-License-Identifier: LGPL-3.0-or-later
#include "visiong/modules/IVE.h"
#include "visiong/core/RgaHelper.h"
#include "visiong/core/BufferStateMachine.h"
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
#include <limits>
#include <stdexcept>
#include <vector>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

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
    bool hsv_output = false;
};

template <typename Source, typename Destination>
ImageBuffer widen_ive_image(const IVE_IMAGE_S& image, PIXEL_FORMAT_E format) {
    std::vector<unsigned char> bytes(static_cast<size_t>(image.u32Width) * image.u32Height * sizeof(Destination));
    auto* output = reinterpret_cast<Destination*>(bytes.data());
    auto* address = reinterpret_cast<void*>(image.au64VirAddr[0]);
    if (mmz_flush_start(address) != RK_SUCCESS) throw std::runtime_error("IVE: Failed to read an output buffer.");
    for (RK_U32 y = 0; y < image.u32Height; ++y) {
        const auto* row = reinterpret_cast<const Source*>(image.au64VirAddr[0]) + static_cast<size_t>(y) * image.au32Stride[0];
        for (RK_U32 x = 0; x < image.u32Width; ++x) output[static_cast<size_t>(y) * image.u32Width + x] = row[x];
    }
    if (mmz_flush_end(address) != RK_SUCCESS) throw std::runtime_error("IVE: Failed to end output buffer access.");
    return ImageBuffer(image.u32Width, image.u32Height, format, std::move(bytes));
}

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

void copy_image_rows(
    RK_U64 dst_base,
    int dst_stride_bytes,
    const unsigned char* src_base,
    int src_stride_bytes,
    int row_count,
    int row_bytes,
    bool swap_rb = false,
    bool swap_uv = false,
    int range_scale = 0) {
    for (int row = 0; row < row_count; ++row) {
        auto* destination = reinterpret_cast<unsigned char*>(dst_base + static_cast<RK_U64>(row) * dst_stride_bytes);
        const auto* source = src_base + static_cast<size_t>(row) * src_stride_bytes;
        int offset = 0;
        if (swap_rb) {
#if defined(__ARM_NEON)
            for (; offset + 48 <= row_bytes; offset += 48) {
                const auto values = vld3q_u8(source + offset);
                uint8x16x3_t swapped{{values.val[2], values.val[1], values.val[0]}};
                vst3q_u8(destination + offset, swapped);
            }
#endif
            for (; offset < row_bytes; offset += 3) {
                destination[offset] = source[offset + 2];
                destination[offset + 1] = source[offset + 1];
                destination[offset + 2] = source[offset];
            }
        } else if (range_scale) {
#if defined(__ARM_NEON)
            for (; offset + 16 <= row_bytes; offset += 16) {
                uint8x16_t values = vld1q_u8(source + offset);
                if (swap_uv) values = vrev16q_u8(values);
                auto low = vmull_u8(vget_low_u8(values), vdup_n_u8(range_scale));
                auto high = vmull_u8(vget_high_u8(values), vdup_n_u8(range_scale));
                low = vaddq_u16(low, vdupq_n_u16(127));
                high = vaddq_u16(high, vdupq_n_u16(127));
                low = vaddq_u16(vaddq_u16(low, vshrq_n_u16(low, 8)), vdupq_n_u16(1));
                high = vaddq_u16(vaddq_u16(high, vshrq_n_u16(high, 8)), vdupq_n_u16(1));
                values = vaddq_u8(vcombine_u8(vshrn_n_u16(low, 8), vshrn_n_u16(high, 8)), vdupq_n_u8(16));
                vst1q_u8(destination + offset, values);
            }
#endif
            for (; offset < row_bytes; ++offset) {
                const unsigned value = source[swap_uv ? offset ^ 1 : offset];
                destination[offset] = static_cast<unsigned char>(16 + (value * range_scale + 127) / 255);
            }
        } else if (swap_uv) {
#if defined(__ARM_NEON)
            for (; offset + 16 <= row_bytes; offset += 16) {
                vst1q_u8(destination + offset, vrev16q_u8(vld1q_u8(source + offset)));
            }
#endif
            for (; offset < row_bytes; offset += 2) {
                destination[offset] = source[offset + 1];
                destination[offset + 1] = source[offset];
            }
        } else {
            std::memcpy(destination, source, row_bytes);
        }
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
    if (mode_val < 0 || (mode_val & 15) > 3 || (mode_val & ~0x33) != 0) {
        throw std::invalid_argument("Invalid IVE CSC mode.");
    }
    CscModeTraits traits;
    traits.yuv_input = (mode_val <= 0x1F); // YUV2RGB / YUV2HSV
    traits.yuv_output = (mode_val >= 0x20 && mode_val <= 0x23); // RGB2YUV
    traits.hsv_output = (mode_val >= 0x10 && mode_val <= 0x13) || (mode_val >= 0x30 && mode_val <= 0x33);
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
    const char* api_name,
    bool input_full_range) {
    if (src.format != RK_FMT_YUV420SP && src.format != RK_FMT_YUV420SP_VU) {
        throw std::invalid_argument(std::string(api_name) + " requires YUV420SP or YUV420SP_VU input.");
    }
    return ive.csc(src, full_range ? full_mode : limit_mode, input_full_range);
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

static PIXEL_FORMAT_E ive_type_to_visiong_format(IVE_IMAGE_TYPE_E type) {
    switch (type) {
        case IVE_IMAGE_TYPE_U8C1: return visiong::kGray8Format;
        case IVE_IMAGE_TYPE_S8C1: return visiong::kS8C1Format;
        case IVE_IMAGE_TYPE_S16C1: return visiong::kS16C1Format;
        case IVE_IMAGE_TYPE_U16C1: return visiong::kU16C1Format;
        case IVE_IMAGE_TYPE_U64C1: return visiong::kU64C1Format;
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
IVEModel::IVEModel(int width, int height, int model_size) : m_width(width), m_height(height) {
    if (width <= 0 || height <= 0 || width > 4096 || height > 4096) {
        throw std::invalid_argument("IVE model dimensions must be in [1,4096].");
    }
    if (model_size < 0) throw std::invalid_argument("IVE model_size cannot be negative.");
    if (model_size == 0) {
        const RK_U64 pixels = static_cast<RK_U64>(calc_stride(width)) * height;
        const RK_U64 background_pixels = static_cast<RK_U64>(calc_stride(width)) * calc_stride(height);
        const RK_U64 bytes = std::max(pixels * 5 * 4 + sizeof(IVE_GMM_NUM_INFO_S), background_pixels * sizeof(IVE_BG_MODEL_PIX_S));
        if (bytes > std::numeric_limits<RK_U32>::max()) throw std::invalid_argument("IVE model is too large.");
        mem_info.u32Size = static_cast<RK_U32>(bytes);
    } else {
        mem_info.u32Size = model_size;
    }
    if (create_mem_info(&mem_info, mem_info.u32Size) != RK_SUCCESS) {
        throw std::runtime_error("Failed to allocate memory for IVEModel.");
    }
    try {
        clear_locked();
    } catch (...) {
        mmz_free(reinterpret_cast<void*>(mem_info.u64VirAddr));
        mem_info = {};
        throw;
    }
}

void IVEModel::validate_dimensions(int width, int height) const {
    if (width != m_width || height != m_height) {
        throw std::invalid_argument("IVE model dimensions must match the source image.");
    }
}

void IVEModel::clear_locked() {
    m_algorithm = Algorithm::Uninitialized;
    m_gmm_models = 0;
    void* address = reinterpret_cast<void*>(mem_info.u64VirAddr);
    if (mmz_flush_start(address) != RK_SUCCESS) throw std::runtime_error("Failed to begin IVE model reset.");
    std::memset(address, 0, mem_info.u32Size);
    if (mmz_flush_end(address) != RK_SUCCESS) throw std::runtime_error("Failed to end IVE model reset.");
    if (m_gmm_matched.au64VirAddr[0]) {
        address = reinterpret_cast<void*>(m_gmm_matched.au64VirAddr[0]);
        if (mmz_flush_start(address) != RK_SUCCESS) throw std::runtime_error("Failed to begin GMM match reset.");
        std::memset(address, 0, static_cast<size_t>(m_gmm_matched.au32Stride[0]) * m_height);
        if (mmz_flush_end(address) != RK_SUCCESS) throw std::runtime_error("Failed to end GMM match reset.");
    }
    if (m_gmm_output.u64VirAddr) {
        address = reinterpret_cast<void*>(m_gmm_output.u64VirAddr);
        if (mmz_flush_start(address) != RK_SUCCESS) throw std::runtime_error("Failed to begin GMM output reset.");
        std::memset(address, 0, m_gmm_output.u32Size);
        if (mmz_flush_end(address) != RK_SUCCESS) throw std::runtime_error("Failed to end GMM output reset.");
    }
}

void IVEModel::prepare_gmm_locked(bool separate_output) {
    if (!m_gmm_matched.au64VirAddr[0]) {
        if (create_image(&m_gmm_matched, IVE_IMAGE_TYPE_U8C1, m_width, m_height) != RK_SUCCESS) {
            throw std::runtime_error("Failed to allocate GMM match state.");
        }
        // GMM matched records use packed rows; the runtime has no separate match-stride register.
        // GMM 匹配记录采用紧凑行布局，运行库没有独立的匹配图步长寄存器。
        m_gmm_matched.au32Stride[0] = m_width;
    }
    if (separate_output && !m_gmm_output.u64VirAddr &&
        create_mem_info(&m_gmm_output, mem_info.u32Size) != RK_SUCCESS) {
        throw std::runtime_error("Failed to allocate GMM output state.");
    }
}

void IVEModel::reset() {
    std::lock_guard<std::mutex> lock(m_mutex);
    clear_locked();
}
IVEModel::~IVEModel() {
    if (m_gmm_matched.au64VirAddr[0]) mmz_free(reinterpret_cast<void*>(m_gmm_matched.au64VirAddr[0]));
    if (m_gmm_output.u64VirAddr) mmz_free(reinterpret_cast<void*>(m_gmm_output.u64VirAddr));
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
    free_mem(m_lk_next_pts_mem);
    free_mem(m_lk_status_mem);
    free_mem(m_lk_error_mem);
    free_mem(m_lk_pyramid_mem);
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
    if (width <= 0 || height <= 0 || width > 4096 || height > 4096) {
        throw std::invalid_argument("IVE image dimensions must be in [1,4096].");
    }
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
            static_cast<size_t>(calc_stride(m_width)) * m_height * 6,
            (size_t)sizeof(IVE_CCBLOB_S),
            (size_t)sizeof(IVE_NCC_DST_MEM_S)
        });
        if (create_mem_info(m_mem_info.get(), mem_size) != RK_SUCCESS) {
            throw std::runtime_error("IVE Failed to create m_mem_info");
        }

        // 3. LK optical-flow temporary buffers. / 3）LK 光流临时缓冲区。
        const int max_lk_points = LK_OPTICAL_FLOW_MAX_POINT_NUM;
        m_lk_pts_mem = std::make_unique<IVE_MEM_INFO_S>();
        if (create_mem_info(m_lk_pts_mem.get(), max_lk_points * sizeof(IVE_POINT_S25Q7_S)) != RK_SUCCESS) {
            throw std::runtime_error("IVE Failed to create lk_pts_mem");
        }
        m_lk_next_pts_mem = std::make_unique<IVE_MEM_INFO_S>();
        if (create_mem_info(m_lk_next_pts_mem.get(), max_lk_points * sizeof(IVE_POINT_S25Q7_S)) != RK_SUCCESS) {
            throw std::runtime_error("IVE Failed to create lk_next_pts_mem");
        }
        m_lk_status_mem = std::make_unique<IVE_MEM_INFO_S>();
        if (create_mem_info(m_lk_status_mem.get(), calc_stride(max_lk_points)) != RK_SUCCESS) {
            throw std::runtime_error("IVE Failed to create lk_status_mem");
        }
        m_lk_error_mem = std::make_unique<IVE_MEM_INFO_S>();
        if (create_mem_info(m_lk_error_mem.get(), calc_stride(max_lk_points * sizeof(IVE_U9Q7))) != RK_SUCCESS) {
            throw std::runtime_error("IVE Failed to create lk_error_mem");
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
        size_t st_ctrl_mem_size = (size_t)calc_stride(width) * height + sizeof(IVE_ST_CORNER_MEM_S) * 2;
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
    if (src.w_stride < src.width || src.h_stride < src.height ||
        static_cast<uint64_t>(src.w_stride) * item_size > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
        static_cast<uint64_t>(src.get_size()) <
            (static_cast<uint64_t>(src.height - 1) * src.w_stride + src.width) * item_size) {
        throw std::invalid_argument("IVE input storage is smaller than its dimensions and stride.");
    }
    visiong::bufstate::prepare_cpu_read(src);
    if (mmz_flush_start(reinterpret_cast<void*>(dst->au64VirAddr[0])) != RK_SUCCESS) {
        throw std::runtime_error("IVE: Failed to begin image buffer access.");
    }
    copy_data_with_stride(
        (void*)dst->au64VirAddr[0],
        dst->au32Stride[0] * item_size,
        src.get_data(),
        src.w_stride * item_size,
        src.height,
        src.width * item_size
    );

    if (mmz_flush_end(reinterpret_cast<void*>(dst->au64VirAddr[0])) != RK_SUCCESS) {
        throw std::runtime_error("IVE: Failed to end image buffer access.");
    }
}

ImageBuffer IVE::copy_from_ive_buffer(const IVE_IMAGE_S* src, int expected_w, int expected_h) {
    if (!src || !src->au64VirAddr[0]) return ImageBuffer();
    
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
    if (mmz_flush_start(reinterpret_cast<void*>(src->au64VirAddr[0])) != RK_SUCCESS) {
        throw std::runtime_error("IVE: Failed to begin image output access.");
    }
    copy_data_from_stride(data.data(), (const void*)src->au64VirAddr[0], w * item_size_bytes, h, src_stride_bytes);
    if (mmz_flush_end(reinterpret_cast<void*>(src->au64VirAddr[0])) != RK_SUCCESS) {
        throw std::runtime_error("IVE: Failed to end image output access.");
    }

    return ImageBuffer(w, h, out_format, std::move(data));
}

ImageBuffer IVE::morph_op(const ImageBuffer& src, bool is_dilate, int kernel_size) {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_buffers(src.width, src.height);
    copy_to_ive_buffer(src, m_img_u8c1_in1.get());

    RK_U8 mask[kFiveByFiveMaskArea];
    fill_morph_mask(mask, kernel_size);

    IVE_HANDLE handle;
    if (kernel_size == 3) {
        // Use the 3x3 rank operator to avoid stale bits in the runtime's 5x5 mask.
        // 使用 3x3 秩滤波算子，避免运行库 5x5 模板寄存器残留旧掩码位。
        IVE_ORD_STAT_FILTER_CTRL_S ctrl{};
        ctrl.enMode = static_cast<IVE_ORD_STAT_FILTER_MODE_E>(is_dilate ? 2 : 0);
        if (RK_MPI_IVE_OrdStatFilter(&handle, m_img_u8c1_in1.get(), m_img_u8c1_out1.get(), &ctrl, RK_TRUE) != RK_SUCCESS) {
            throw std::runtime_error("IVE 3x3 morphology failed.");
        }
    } else if (is_dilate) {
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
    if (out_ctrl < IVE_SOBEL_OUT_CTRL_BOTH || out_ctrl >= IVE_SOBEL_OUT_CTRL_BUTT) throw std::invalid_argument("Invalid Sobel output direction.");
    if (RK_MPI_IVE_Sobel(&handle, m_img_u8c1_in1.get(), dst_h, dst_v, &ctrl, RK_TRUE) != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_Sobel failed.");
    }
    ImageBuffer result_h = copy_from_ive_buffer(out_ctrl == IVE_SOBEL_OUT_CTRL_VER ? dst_v : dst_h, m_width, m_height);
    ImageBuffer result_v = (out_ctrl == IVE_SOBEL_OUT_CTRL_BOTH) ? copy_from_ive_buffer(dst_v, m_width, m_height) : ImageBuffer();
    return std::make_tuple(std::move(result_h), std::move(result_v));
}

ImageBuffer IVE::canny(const ImageBuffer& src, uint16_t high_thresh, uint16_t low_thresh) {
    return canny_ex(src, high_thresh, low_thresh, {});
}

ImageBuffer IVE::canny_ex(const ImageBuffer& src, uint16_t high_thresh, uint16_t low_thresh,
                         const std::vector<int8_t>& mask) {
    if (low_thresh > high_thresh) throw std::invalid_argument("Canny low threshold must not exceed its high threshold.");
    if (!mask.empty() && mask.size() != 25) throw std::invalid_argument("Canny mask must contain 25 coefficients.");
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
    const RK_S8 default_mask[25] = {
        0,  0,  0,  0,  0,
        0, -1,  0,  1,  0,
        0, -2,  0,  2,  0,
        0, -1,  0,  1,  0,
        0,  0,  0,  0,  0
    };
    memcpy(ctrl.as8Mask, mask.empty() ? default_mask : mask.data(), sizeof(ctrl.as8Mask));
    
    IVE_HANDLE handle;
    if (RK_MPI_IVE_CannyEdge(&handle, m_img_u8c1_in1.get(), m_img_u8c1_out1.get(), nullptr, &ctrl, RK_TRUE) != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_CannyEdge failed.");
    }
    return copy_from_ive_buffer(m_img_u8c1_out1.get(), m_width, m_height);
}

ImageBuffer IVE::mag_and_ang(const ImageBuffer& src, uint16_t threshold, bool return_magnitude) {
    auto outputs = mag_and_ang_ex(src, threshold, {});
    return return_magnitude ? std::get<0>(outputs) : std::get<1>(outputs);
}

std::tuple<ImageBuffer, ImageBuffer> IVE::mag_and_ang_ex(const ImageBuffer& src, uint16_t threshold,
                                                       const std::vector<int8_t>& mask) {
    if (!mask.empty() && mask.size() != 25) throw std::invalid_argument("MagAndAng mask must contain 25 coefficients.");
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_buffers(src.width, src.height);
    if (src.format != visiong::kGray8Format) throw std::invalid_argument("mag_and_ang requires a GRAY8 input.");
    copy_to_ive_buffer(src, m_img_u8c1_in1.get());
    
    IVE_MAG_AND_ANG_CTRL_S ctrl; 
    memset(&ctrl, 0, sizeof(ctrl)); 
    ctrl.stMem = *m_mem_info.get(); 
    ctrl.u16Thr = threshold;
    
    // Use a 5x5 Scharr-like kernel as recommended for MagAndAng. / 按 MagAndAng 的建议使用 5x5 类 Scharr 卷积核。
    const RK_S8 default_mask[25] = {
        -1, -2,  0,  2,  1,
        -4, -8,  0,  8,  4,
        -6, -12, 0, 12,  6,
        -4, -8,  0,  8,  4,
        -1, -2,  0,  2,  1
    };
    memcpy(ctrl.as8Mask, mask.empty() ? default_mask : mask.data(), sizeof(ctrl.as8Mask));
    
    IVE_HANDLE handle;
    IVE_IMAGE_S magnitude = *m_img_s16c1_out1;
    magnitude.enType = IVE_IMAGE_TYPE_U16C1;
    if (RK_MPI_IVE_MagAndAng(&handle, m_img_u8c1_in1.get(), &magnitude, m_img_u8c1_out1.get(), &ctrl, RK_TRUE) != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_MagAndAng failed.");
    }
    ImageBuffer magnitude_image = copy_from_ive_buffer(&magnitude, m_width, m_height);
    ImageBuffer angle_image = copy_from_ive_buffer(m_img_u8c1_out1.get(), m_width, m_height);
    if (threshold != 0) {
        // The runtime ignores u16Thr. Apply the public inclusive minimum after the hardware computation.
        // 运行库忽略 u16Thr；硬件计算后按公开的包含等号的最小幅值规则抑制输出。
        auto* values = static_cast<uint16_t*>(magnitude_image.get_data());
        auto* angles = static_cast<uint8_t*>(angle_image.get_data());
        for (int y = 0; y < m_height; ++y) {
            for (int x = 0; x < m_width; ++x) {
                auto& value = values[static_cast<size_t>(y) * magnitude_image.w_stride + x];
                if (value < threshold) {
                    value = 0;
                    angles[static_cast<size_t>(y) * angle_image.w_stride + x] = 0;
                }
            }
        }
    }
    return std::make_tuple(std::move(magnitude_image), std::move(angle_image));
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

ImageBuffer IVE::morphology(const ImageBuffer& src, const std::vector<uint8_t>& mask, bool dilate_op) {
    if (mask.size() != 25 || std::any_of(mask.begin(), mask.end(), [](uint8_t value) { return value > 1; })) {
        throw std::invalid_argument("Morphology requires a 25-element mask containing only zero and one.");
    }
    const int active = std::accumulate(mask.begin(), mask.end(), 0);
    if (active == 0) throw std::invalid_argument("Morphology requires at least one active mask element.");
    if (!src.is_valid() || src.format != visiong::kGray8Format || src.width <= 0 || src.height <= 0 ||
        src.width > 4096 || src.height > 4096 ||
        src.w_stride < src.width || src.h_stride < src.height ||
        static_cast<uint64_t>(src.get_size()) < static_cast<uint64_t>(src.height - 1) * src.w_stride + src.width) {
        throw std::invalid_argument("Morphology requires a valid GRAY8 image.");
    }
    visiong::bufstate::prepare_cpu_read(src);
    const auto* input = static_cast<const unsigned char*>(src.get_data());
    for (int y = 0; y < src.height; ++y) {
        for (int x = 0; x < src.width; ++x) {
            const uint8_t value = input[static_cast<size_t>(y) * src.w_stride + x];
            if (value != 0 && value != 255) throw std::invalid_argument("Binary morphology input must contain only 0 and 255.");
        }
    }
    // A U16 sum and hardware threshold avoid sticky mask bits in the native morphology path.
    // U16 求和加硬件阈值避开原生形态学路径中不能可靠清除的掩码位。
    const std::vector<int8_t> coefficients(mask.begin(), mask.end());
    const ImageBuffer sums = filter_ex(src, coefficients, 0, IVE_IMAGE_TYPE_U16C1);
    ImageBuffer output = threshold_u16(sums, dilate_op ? 0 : active * 255 - 1, 65535,
                                       IVE_THRESH_U16_MODE_U16_TO_U8_MIN_MID_MAX, 0, 255, 255);
    auto* pixels = static_cast<unsigned char*>(output.get_data());
    // Define outside-image pixels as zero, independently of the filter's edge extension.
    // 图外像素定义为零，边缘结果不依赖滤波器的边界扩展规则。
    for (int y = 0; y < src.height; ++y) {
        for (int x = 0; x < src.width; ++x) {
            if (x >= 2 && x < src.width - 2 && y >= 2 && y < src.height - 2) continue;
            bool result = !dilate_op;
            for (int ky = 0; ky < 5; ++ky) {
                for (int kx = 0; kx < 5; ++kx) {
                    if (!mask[ky * 5 + kx]) continue;
                    const int sx = x + kx - 2;
                    const int sy = y + ky - 2;
                    const bool set = sx >= 0 && sx < src.width && sy >= 0 && sy < src.height &&
                        input[static_cast<size_t>(sy) * src.w_stride + sx] != 0;
                    result = dilate_op ? result || set : result && set;
                }
            }
            pixels[static_cast<size_t>(y) * output.w_stride + x] = result ? 255 : 0;
        }
    }
    return output;
}

ImageBuffer IVE::ordered_stat_filter(const ImageBuffer& src, IVE_ORD_STAT_FILTER_MODE_E mode) {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_buffers(src.width, src.height);
    if (src.format != visiong::kGray8Format) throw std::invalid_argument("ordered_stat_filter requires a GRAY8 input.");
    copy_to_ive_buffer(src, m_img_u8c1_in1.get());
    IVE_ORD_STAT_FILTER_CTRL_S ctrl{};
    // The runtime register encoding is MIN=0, MEDIAN=1, MAX=2. / 运行库寄存器编码为最小值 0、中值 1、最大值 2。
    switch (mode) {
        case IVE_ORD_STAT_FILTER_MODE_MEDIAN: ctrl.enMode = static_cast<IVE_ORD_STAT_FILTER_MODE_E>(1); break;
        case IVE_ORD_STAT_FILTER_MODE_MAX: ctrl.enMode = static_cast<IVE_ORD_STAT_FILTER_MODE_E>(2); break;
        case IVE_ORD_STAT_FILTER_MODE_MIN: ctrl.enMode = static_cast<IVE_ORD_STAT_FILTER_MODE_E>(0); break;
        default: throw std::invalid_argument("Invalid order statistic filter mode.");
    }
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
    if (src1.format != visiong::kGray8Format || src2.format != visiong::kGray8Format) throw std::invalid_argument("logic_op requires GRAY8 inputs.");
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
    return add_weighted(src1, src2, 32768, 32768);
}

ImageBuffer IVE::sub(const ImageBuffer& src1, const ImageBuffer& src2, IVE_SUB_MODE_E mode) {
    if (src1.format != visiong::kGray8Format || src2.format != visiong::kGray8Format) throw std::invalid_argument("sub requires GRAY8 inputs.");
    if (mode != IVE_SUB_MODE_ABS && mode != IVE_SUB_MODE_SHIFT) throw std::invalid_argument("Invalid subtraction mode.");
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
    return threshold_u8(src, low_thresh, high_thresh, mode, 0, 0, 255);
}

ImageBuffer IVE::cast_16bit_to_8bit(const ImageBuffer& src, IVE_16BIT_TO_8BIT_MODE_E mode) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!src.is_valid() || src.width <= 0 || src.height <= 0 || src.width > 4096 || src.height > 4096 ||
        src.w_stride < src.width || src.h_stride < src.height ||
        static_cast<uint64_t>(src.w_stride) * sizeof(uint16_t) > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
        static_cast<uint64_t>(src.get_size()) <
            (static_cast<uint64_t>(src.height - 1) * src.w_stride + src.width) * sizeof(uint16_t)) {
        throw std::invalid_argument("cast_16bit_to_8bit requires valid dimensions, stride and 16-bit input storage.");
    }
    ensure_buffers(src.width, src.height);
    if (src.format != visiong::kS16C1Format && src.format != visiong::kU16C1Format) throw std::invalid_argument("cast_16bit_to_8bit requires an S16C1 or U16C1 input.");
    
    const bool unsigned_input = mode == IVE_16BIT_TO_8BIT_MODE_U16_TO_U8;
    const bool signed_output = mode == IVE_16BIT_TO_8BIT_MODE_S16_TO_S8;
    if (mode < IVE_16BIT_TO_8BIT_MODE_S16_TO_S8 || mode >= IVE_16BIT_TO_8BIT_MODE_BUTT ||
        (unsigned_input ? src.format != visiong::kU16C1Format : src.format != visiong::kS16C1Format)) {
        throw std::invalid_argument("16-bit cast mode does not match the source format.");
    }

    // The SDK fixes the cast input to U16. Rebase signed values before hardware scaling.
    // SDK 将转换输入固定为 U16，先调整有符号数值，再交给硬件完成缩放。
    if (unsigned_input) {
        copy_to_ive_buffer(src, m_img_u16c1_in1.get());
    } else {
        visiong::bufstate::prepare_cpu_read(src);
        auto* target = reinterpret_cast<RK_U16*>(m_img_u16c1_in1->au64VirAddr[0]);
        mmz_flush_start(target);
        const bool absolute = mode == IVE_16BIT_TO_8BIT_MODE_S16_TO_U8_ABS;
        const int offset = signed_output ? 16384 : 508;
        const int limit = absolute ? 32768 : (signed_output ? 32640 : 1020);
        for (int y = 0; y < src.height; ++y) {
            const auto* source = static_cast<const int16_t*>(src.get_data()) + static_cast<size_t>(y) * src.w_stride;
            auto* row = target + static_cast<size_t>(y) * m_img_u16c1_in1->au32Stride[0];
            int x = 0;
#if defined(__ARM_NEON)
            for (; x + 8 <= src.width; x += 8) {
                const int16x8_t values = vld1q_s16(source + x);
                int32x4_t low = vmovl_s16(vget_low_s16(values));
                int32x4_t high = vmovl_s16(vget_high_s16(values));
                if (absolute) {
                    low = vabsq_s32(low);
                    high = vabsq_s32(high);
                } else {
                    low = vaddq_s32(low, vdupq_n_s32(offset));
                    high = vaddq_s32(high, vdupq_n_s32(offset));
                }
                low = vminq_s32(vmaxq_s32(low, vdupq_n_s32(0)), vdupq_n_s32(limit));
                high = vminq_s32(vmaxq_s32(high, vdupq_n_s32(0)), vdupq_n_s32(limit));
                vst1q_u16(row + x, vcombine_u16(vqmovun_s32(low), vqmovun_s32(high)));
            }
#endif
            for (; x < src.width; ++x) {
                const int value = absolute ? std::abs(static_cast<int>(source[x])) : source[x] + offset;
                row[x] = static_cast<RK_U16>(std::max(0, std::min(limit, value)));
            }
        }
        mmz_flush_end(target);
    }

    IVE_16BIT_TO_8BIT_CTRL_S control{};
    control.enMode = IVE_16BIT_TO_8BIT_MODE_U16_TO_U8;
    control.u8Numerator = 1;
    control.u16Denominator = signed_output ? 128 : 4;
    IVE_HANDLE handle{};
    IVE_IMAGE_S cast_source = *m_img_u16c1_in1;
    // Cast HAL adds width to this field to obtain byte stride. / Cast HAL 将宽度加到此字段以获得字节步长。
    cast_source.au32Stride[0] = 2 * m_img_u16c1_in1->au32Stride[0] - cast_source.u32Width;
    if (RK_MPI_IVE_16BitTo8Bit(&handle, &cast_source, m_img_u8c1_out1.get(), &control, RK_TRUE) != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_16BitTo8Bit failed.");
    }
    ImageBuffer output = copy_from_ive_buffer(m_img_u8c1_out1.get(), m_width, m_height);
    if (signed_output) {
        auto* bytes = static_cast<unsigned char*>(output.get_data());
        size_t index = 0;
#if defined(__ARM_NEON)
        for (; index + 16 <= output.get_size(); index += 16) {
            vst1q_u8(bytes + index, veorq_u8(vld1q_u8(bytes + index), vdupq_n_u8(0x80)));
        }
#endif
        for (; index < output.get_size(); ++index) bytes[index] ^= 0x80;
        output.format = visiong::kS8C1Format;
    }
    return output;
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
    if (mode != IVE_INTEG_OUT_CTRL_SUM && mode != IVE_INTEG_OUT_CTRL_SQSUM && mode != IVE_INTEG_OUT_CTRL_COMBINE) {
        throw std::invalid_argument("Invalid integral output mode.");
    }
    IVE_IMAGE_S output = *m_img_u64c1_out1;
    if (mode == IVE_INTEG_OUT_CTRL_SUM) output.enType = IVE_IMAGE_TYPE_U32C1;
    IVE_IMAGE_S hardware_output = output;
    const RK_U32 item_size = mode == IVE_INTEG_OUT_CTRL_SUM ? 4 : 8;
    // Integral HAL adds (item_size-1)*width instead of multiplying pixel stride.
    // 积分 HAL 通过加上 (item_size-1)*width 换算步长，未直接乘像素步长。
    hardware_output.au32Stride[0] = item_size * output.au32Stride[0] - (item_size - 1) * output.u32Width;
    IVE_HANDLE handle;
    if (RK_MPI_IVE_Integ(&handle, m_img_u8c1_in1.get(), &hardware_output, &ctrl, RK_TRUE) != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_Integ failed.");
    }
    // SUM is produced as U32; keep the public U64 result format. / SUM 硬件输出为 U32，对外保持 U64 结果格式。
    if (mode == IVE_INTEG_OUT_CTRL_SUM) return widen_ive_image<RK_U32, RK_U64>(output, visiong::kU64C1Format);
    return copy_from_ive_buffer(m_img_u64c1_out1.get(), m_width, m_height);
}

std::vector<Blob> IVE::ccl(const ImageBuffer& src, int min_area) {
    return ccl_config(src, min_area, IVE_CCL_MODE_8C, 0, 0);
}

std::vector<Blob> IVE::ccl_config(const ImageBuffer& src, int min_area, IVE_CCL_MODE_E mode,
                                  uint16_t init_area_threshold, uint16_t step) {
    if (min_area < 0) throw std::invalid_argument("CCL minimum area cannot be negative.");
    if (mode != IVE_CCL_MODE_4C && mode != IVE_CCL_MODE_8C) throw std::invalid_argument("Invalid CCL connectivity.");
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_buffers(src.width, src.height);
    if (src.format != visiong::kGray8Format) throw std::invalid_argument("ccl requires a GRAY8 (binarized) input.");
    copy_to_ive_buffer(src, m_img_u8c1_out1.get()); 
    IVE_MEM_INFO_S blob_memory{};
    if (create_mem_info(&blob_memory, sizeof(IVE_CCBLOB_S)) != RK_SUCCESS) throw std::runtime_error("Failed to allocate CCL output memory.");
    IveMmzGuard blob_guard;
    blob_guard.set(blob_memory.u64PhyAddr, blob_memory.u64VirAddr);
    void* blob_address = reinterpret_cast<void*>(blob_memory.u64VirAddr);
    if (mmz_flush_start(blob_address) != RK_SUCCESS) throw std::runtime_error("Failed to initialize CCL output memory.");
    std::memset(blob_address, 0, sizeof(IVE_CCBLOB_S));
    if (mmz_flush_end(blob_address) != RK_SUCCESS) throw std::runtime_error("Failed to synchronize CCL output memory.");
    IVE_CCL_CTRL_S ctrl{};
    ctrl.enMode = mode;
    ctrl.u16InitAreaThr = init_area_threshold;
    ctrl.u16Step = step;
    ctrl.stMem = *m_mem_info;
    IVE_HANDLE handle;
    if (RK_MPI_IVE_CCL(&handle, m_img_u8c1_out1.get(), &blob_memory, &ctrl, RK_TRUE) != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_CCL failed.");
    }
    if (mmz_flush_start(blob_address) != RK_SUCCESS) throw std::runtime_error("Failed to read CCL output memory.");
    IVE_CCBLOB_S blob_copy{};
    std::memcpy(&blob_copy, blob_address, sizeof(blob_copy));
    if (mmz_flush_end(blob_address) != RK_SUCCESS) throw std::runtime_error("Failed to end CCL output access.");
    const IVE_CCBLOB_S* blob_info = &blob_copy;
    if (blob_info->s8LabelStatus != 0) {
        throw std::runtime_error("IVE CCL could not label all regions.");
    }
    std::vector<Blob> results;
    for (int i = 0; i < IVE_MAX_REGION_NUM; ++i) {
        if (blob_info->astRegion[i].u32Area != 0 && blob_info->astRegion[i].u32Area >= static_cast<unsigned int>(min_area)) {
            results.emplace_back(blob_info->astRegion[i].u16Left, blob_info->astRegion[i].u16Top,
                                 blob_info->astRegion[i].u16Right - blob_info->astRegion[i].u16Left + 1,
                                 blob_info->astRegion[i].u16Bottom - blob_info->astRegion[i].u16Top + 1,
                                 (blob_info->astRegion[i].u16Left + blob_info->astRegion[i].u16Right) / 2,
                                 (blob_info->astRegion[i].u16Top + blob_info->astRegion[i].u16Bottom) / 2,
                                 blob_info->astRegion[i].u32Area, static_cast<unsigned int>(i + 1));
        }
    }
    return results;
}

double IVE::ncc(const ImageBuffer& src1, const ImageBuffer& src2) {
    const auto aggregates = ncc_raw(src1, src2);
    const double denominator = std::sqrt(static_cast<double>(std::get<1>(aggregates))) *
                               std::sqrt(static_cast<double>(std::get<2>(aggregates)));
    return denominator > 0.0 ? static_cast<double>(std::get<0>(aggregates)) / denominator : 0.0;
}

ImageBuffer IVE::csc(const ImageBuffer& src, IVE_CSC_MODE_E mode, bool input_full_range) {
    std::unique_lock<std::mutex> lock(m_mutex);

    const CscModeTraits traits = get_csc_mode_traits(mode);
    if (input_full_range && !traits.yuv_input) throw std::invalid_argument("input_full_range applies only to YUV input.");
    const ImageBuffer* src_ptr = &src;
    validate_csc_source_format(*src_ptr, traits);

    const int w = src_ptr->width;
    const int h = src_ptr->height;
    if (!src.is_valid() || w <= 0 || h <= 0 || w > 4096 || h > 4096 ||
        src.w_stride < w || src.h_stride < h) {
        throw std::invalid_argument("IVE CSC requires valid image dimensions and strides.");
    }
    const uint64_t input_size = traits.yuv_input ?
        static_cast<uint64_t>(src.w_stride) * src.h_stride + static_cast<uint64_t>(src.w_stride) * (h / 2) :
        (static_cast<uint64_t>(h - 1) * src.w_stride + w) * 3;
    if (static_cast<uint64_t>(src.get_size()) < input_size ||
        (!traits.yuv_input && static_cast<uint64_t>(src.w_stride) * 3 > static_cast<uint64_t>(std::numeric_limits<int>::max()))) {
        throw std::invalid_argument("IVE CSC input storage or byte stride is invalid.");
    }
    if ((traits.yuv_input || traits.yuv_output) && ((w & 1) || (h & 1))) {
        throw std::invalid_argument("IVE YUV conversion requires even image dimensions.");
    }
    const int stride = calc_stride(w, kImageAlign);
    const IVE_IMAGE_TYPE_E src_type = traits.yuv_input ? IVE_IMAGE_TYPE_YUV420SP : IVE_IMAGE_TYPE_U8C3_PACKAGE;
    const IVE_IMAGE_TYPE_E dst_type = traits.yuv_output ? IVE_IMAGE_TYPE_YUV420SP : IVE_IMAGE_TYPE_U8C3_PACKAGE;

    if (h > 2046) {
        // Preserve the cached common-size path; tall frames share the compact byte interface's tiling.
        // 保留常用尺寸的缓存路径；高图像复用紧凑字节接口的分块布局。
        visiong::bufstate::prepare_cpu_read(*src_ptr);
        const auto* pixels = static_cast<const unsigned char*>(src_ptr->get_data());
        const size_t plane_size = static_cast<size_t>(w) * h;
        std::vector<uint8_t> compact(traits.yuv_input ? plane_size * 3 / 2 : plane_size * 3);
        if (traits.yuv_input) {
            for (int y = 0; y < h; ++y) {
                std::memcpy(compact.data() + static_cast<size_t>(y) * w,
                            pixels + static_cast<size_t>(y) * src_ptr->w_stride, w);
            }
            const size_t chroma_offset = static_cast<size_t>(src_ptr->w_stride) * src_ptr->h_stride;
            for (int y = 0; y < h / 2; ++y) {
                std::memcpy(compact.data() + plane_size + static_cast<size_t>(y) * w,
                            pixels + chroma_offset + static_cast<size_t>(y) * src_ptr->w_stride, w);
            }
        } else {
            for (int y = 0; y < h; ++y) {
                std::memcpy(compact.data() + static_cast<size_t>(y) * w * 3,
                            pixels + static_cast<size_t>(y) * src_ptr->w_stride * 3, static_cast<size_t>(w) * 3);
            }
        }
        const bool bgr_order = !traits.yuv_input && src_ptr->format == RK_FMT_BGR888;
        const bool vu_order = traits.yuv_input && src_ptr->format == RK_FMT_YUV420SP_VU;
        lock.unlock();
        std::vector<uint8_t> output = csc_bytes(std::move(compact), w, h, src_type, dst_type, mode,
                                               input_full_range, bgr_order, vu_order);
        const PIXEL_FORMAT_E format = traits.yuv_output ? RK_FMT_YUV420SP :
            (traits.hsv_output ? visiong::kHsv888Format : RK_FMT_RGB888);
        return ImageBuffer(w, h, format, std::move(output));
    }

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
    visiong::bufstate::prepare_cpu_read(*src_ptr);
    mmz_flush_start(reinterpret_cast<void*>(src_ive.au64VirAddr[0]));
    if (traits.yuv_input) {
        const int y_src_stride = src_ptr->w_stride;
        const size_t y_plane_size = static_cast<size_t>(y_src_stride) * src_ptr->h_stride;
        copy_image_rows(src_ive.au64VirAddr[0], stride, src_data, y_src_stride, h, w, false, false,
                input_full_range ? 219 : 0);
        copy_image_rows(src_ive.au64VirAddr[1], stride, src_data + y_plane_size, y_src_stride, h / 2, w,
                false, src_ptr->format == RK_FMT_YUV420SP, input_full_range ? 224 : 0);
    } else {
        const int src_row_bytes = src_ptr->w_stride * 3;
        const int dst_row_bytes = stride * 3;
        copy_image_rows(src_ive.au64VirAddr[0], dst_row_bytes, src_data, src_row_bytes, h, w * 3,
                src_ptr->format == RK_FMT_RGB888);
    }
    mmz_flush_end(reinterpret_cast<void*>(src_ive.au64VirAddr[0]));

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

    const PIXEL_FORMAT_E out_format = traits.yuv_output ? RK_FMT_YUV420SP :
            (traits.hsv_output ? visiong::kHsv888Format : RK_FMT_RGB888);
    std::vector<unsigned char> out_data(
        traits.yuv_output ? static_cast<size_t>(w) * h * 3 / 2 : static_cast<size_t>(w) * h * 3);

    if (traits.yuv_output) {
        copy_data_from_stride(out_data.data(), reinterpret_cast<void*>(dst_ive.au64VirAddr[0]), w, h, stride);
        copy_image_rows(reinterpret_cast<RK_U64>(out_data.data() + static_cast<size_t>(w) * h), w,
                reinterpret_cast<const unsigned char*>(dst_ive.au64VirAddr[1]), stride, h / 2, w, false, true);
    } else {
        copy_image_rows(reinterpret_cast<RK_U64>(out_data.data()), w * 3,
                reinterpret_cast<const unsigned char*>(dst_ive.au64VirAddr[0]), stride * 3, h, w * 3, !traits.hsv_output);
    }

    mmz_flush_end(reinterpret_cast<void*>(dst_ive.au64VirAddr[0]));

    return ImageBuffer(w, h, out_format, std::move(out_data));
}

// Convenience wrappers around CSC modes. / 围绕 CSC 模式的便捷封装。
ImageBuffer IVE::yuv_to_rgb(const ImageBuffer& src, bool full_range, bool input_full_range) {
    return convert_from_yuv420sp(
        *this,
        src,
        full_range,
        IVE_CSC_MODE_FULL_BT601_YUV2RGB,
        IVE_CSC_MODE_LIMIT_BT601_YUV2RGB,
        "yuv_to_rgb", input_full_range);
}

ImageBuffer IVE::yuv_to_hsv(const ImageBuffer& src, bool full_range, bool input_full_range) {
    return convert_from_yuv420sp(
        *this,
        src,
        full_range,
        IVE_CSC_MODE_FULL_BT601_YUV2HSV,
        IVE_CSC_MODE_LIMIT_BT601_YUV2HSV,
        "yuv_to_hsv", input_full_range);
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

// DMA and lookup operations. / DMA 及查表算子。
ImageBuffer IVE::dma(const ImageBuffer& src, IVE_DMA_MODE_E mode) {
    return dma_transfer(src, mode);
}

ImageBuffer IVE::cast_8bit_to_8bit(const ImageBuffer& src, int8_t bias, uint8_t numerator, uint8_t denominator) {
    if (denominator == 0) throw std::invalid_argument("Cast denominator must be positive.");
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
    
    mmz_flush_start(reinterpret_cast<void*>(stMap.u64VirAddr));
    memcpy((void*)stMap.u64VirAddr, lut.data(), 256 * sizeof(uint8_t));
    
    // Flush LUT cache so hardware sees the latest table. / 刷新 LUT 缓存，让硬件读取到最新表内容。
    mmz_flush_end(reinterpret_cast<void*>(stMap.u64VirAddr));
    
    IVE_MAP_CTRL_S ctrl; 
    memset(&ctrl, 0, sizeof(ctrl)); 
    ctrl.enMode = IVE_MAP_MODE_U8;
    
    IVE_HANDLE handle;
    if (RK_MPI_IVE_Map(&handle, m_img_u8c1_in1.get(), &stMap, m_img_u8c1_out1.get(), &ctrl, RK_TRUE) != RK_SUCCESS)
        throw std::runtime_error("RK_MPI_IVE_Map failed.");
    return copy_from_ive_buffer(m_img_u8c1_out1.get(), m_width, m_height);
}

