// SPDX-License-Identifier: LGPL-3.0-or-later
#include "visiong/modules/IVE.h"
#include "visiong/common/pixel_format.h"
#include "visiong/core/BufferStateMachine.h"

#include "modules/internal/ive_memory.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_range(int value, int minimum, int maximum, const char* name) {
    if (value < minimum || value > maximum) {
        throw std::invalid_argument(std::string(name) + " is outside [" +
            std::to_string(minimum) + "," + std::to_string(maximum) + "].");
    }
}

void check_operator_input(const ImageBuffer& source, PIXEL_FORMAT_E format, size_t item_size,
                           const char* operation = "Threshold") {
    if (!source.is_valid() || source.width <= 0 || source.height <= 0 ||
        source.width > 4096 || source.height > 4096) {
        throw std::invalid_argument(std::string(operation) + " requires a valid image with dimensions in [1,4096].");
    }
    if (source.format != format) {
        throw std::invalid_argument(std::string(operation) + " input format does not match the selected operator.");
    }
    if (source.w_stride < source.width || source.h_stride < source.height ||
        static_cast<uint64_t>(source.w_stride) * item_size > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
        static_cast<uint64_t>(source.get_size()) <
            (static_cast<uint64_t>(source.height - 1) * source.w_stride + source.width) * item_size) {
        throw std::invalid_argument(std::string(operation) + " input storage is smaller than its image dimensions.");
    }
}

void check_threshold_values(int low, int high, int input_min, int input_max,
                          int minimum, int middle, int maximum, int output_min, int output_max,
                          bool use_high) {
    check_range(low, input_min, input_max, "low_thresh");
    check_range(high, input_min, input_max, "high_thresh");
    if (use_high && low > high) {
        throw std::invalid_argument("low_thresh must not exceed high_thresh for a three-region threshold.");
    }
    check_range(minimum, output_min, output_max, "min_val");
    check_range(middle, output_min, output_max, "mid_val");
    check_range(maximum, output_min, output_max, "max_val");
}

ImageBuffer copy_threshold_s8(const IVE_IMAGE_S& image) {
    using visiong::ive_internal::mmz_flush_end;
    using visiong::ive_internal::mmz_flush_start;
    std::vector<unsigned char> bytes(static_cast<size_t>(image.u32Width) * image.u32Height);
    void* address = reinterpret_cast<void*>(image.au64VirAddr[0]);
    if (mmz_flush_start(address) != RK_SUCCESS) {
        throw std::runtime_error("Threshold failed to begin output buffer access.");
    }
    for (RK_U32 y = 0; y < image.u32Height; ++y) {
        std::memcpy(bytes.data() + static_cast<size_t>(y) * image.u32Width,
                    static_cast<const unsigned char*>(address) + static_cast<size_t>(y) * image.au32Stride[0],
                    image.u32Width);
    }
    if (mmz_flush_end(address) != RK_SUCCESS) {
        throw std::runtime_error("Threshold failed to end output buffer access.");
    }
    return ImageBuffer(image.u32Width, image.u32Height, visiong::kS8C1Format, std::move(bytes));
}

class DmaImage {
public:
    DmaImage(int width, int height, IVE_IMAGE_TYPE_E type = IVE_IMAGE_TYPE_U8C1) {
        if (visiong::ive_internal::create_image(&image, type, width, height) != RK_SUCCESS) {
            throw std::runtime_error("IVE failed to allocate its output buffer.");
        }
    }
    ~DmaImage() {
        visiong::ive_internal::free_mmz(image.au64PhyAddr[0], image.au64VirAddr[0]);
    }
    DmaImage(const DmaImage&) = delete;
    DmaImage& operator=(const DmaImage&) = delete;

    IVE_IMAGE_S image{};
};

IVE_DATA_S dma_data(const IVE_IMAGE_S& image) {
    IVE_DATA_S data{};
    data.u64PhyAddr = image.au64PhyAddr[0];
    data.u64VirAddr = image.au64VirAddr[0];
    data.u32Stride = image.au32Stride[0];
    data.u32Width = image.u32Width;
    data.u32Height = image.u32Height;
    return data;
}

RK_U8 filter_output_mode(IVE_IMAGE_TYPE_E type) {
    switch (type) {
        case IVE_IMAGE_TYPE_U8C1: return 0;
        case IVE_IMAGE_TYPE_S8C1: return 1;
        case IVE_IMAGE_TYPE_U16C1: return 2;
        case IVE_IMAGE_TYPE_S16C1: return 3;
        default: throw std::invalid_argument("IVE convolution output must be U8C1, S8C1, U16C1 or S16C1.");
    }
}

void check_convolution(const ImageBuffer& src, const std::vector<int8_t>& mask, int norm) {
    check_operator_input(src, visiong::kGray8Format, 1, "Convolution");
    if (mask.size() != 25) throw std::invalid_argument("IVE convolution requires a 25-element signed 8-bit mask.");
    check_range(norm, 0, 15, "norm");
}

void check_gray_pair(const ImageBuffer& src1, const ImageBuffer& src2, const char* operation) {
    check_operator_input(src1, visiong::kGray8Format, 1, operation);
    check_operator_input(src2, visiong::kGray8Format, 1, operation);
    if (src1.width != src2.width || src1.height != src2.height) {
        throw std::invalid_argument(std::string(operation) + " requires images of the same size.");
    }
}

void copy_mmz_bytes(IVE_MEM_INFO_S& memory, void* bytes, size_t size, bool write) {
    if (size > memory.u32Size) throw std::runtime_error("IVE memory buffer is too small.");
    void* address = reinterpret_cast<void*>(memory.u64VirAddr);
    if (visiong::ive_internal::mmz_flush_start(address) != RK_SUCCESS) {
        throw std::runtime_error("IVE failed to begin memory access.");
    }
    if (write) std::memcpy(address, bytes, size);
    else std::memcpy(bytes, address, size);
    if (visiong::ive_internal::mmz_flush_end(address) != RK_SUCCESS) {
        throw std::runtime_error("IVE failed to end memory access.");
    }
}

uint32_t cast_scale(int numerator, int denominator, int maximum_denominator, int bias) {
    check_range(numerator, 0, 255, "numerator");
    check_range(denominator, 1, maximum_denominator, "denominator");
    check_range(bias, -128, 127, "bias");
    // The native wrapper rounds its floating ratio to unsigned Q8.24 before programming the hardware.
    // 原生封装先将浮点比例四舍五入为无符号 Q8.24，再配置硬件。
    return static_cast<uint32_t>(((static_cast<uint64_t>(numerator) << 24) + denominator / 2) / denominator);
}

int64_t cast_round(int value, uint32_t scale) {
    const int64_t product = static_cast<int64_t>(value) * scale;
    constexpr int64_t half = int64_t{1} << 23;
    constexpr int64_t unit = int64_t{1} << 24;
    return product >= 0 ? (product + half) / unit : -((-product + half - 1) / unit);
}

bool cast_negative_half(int value, uint32_t scale) {
    return value < 0 && ((static_cast<uint64_t>(-value) * scale) & 0xffffffu) == 0x800000u;
}

}  // namespace

ImageBuffer IVE::threshold_u8(const ImageBuffer& src, int low_thresh, int high_thresh,
                             IVE_THRESH_MODE_E mode, int min_val, int mid_val, int max_val) {
    check_operator_input(src, visiong::kGray8Format, 1);
    switch (mode) {
        case IVE_THRESH_MODE_BINARY:
        case IVE_THRESH_MODE_TRUNC:
        case IVE_THRESH_MODE_TO_MINVAL:
        case IVE_THRESH_MODE_MIN_MID_MAX:
        case IVE_THRESH_MODE_ORI_MID_MAX:
        case IVE_THRESH_MODE_MIN_MID_ORI:
        case IVE_THRESH_MODE_MIN_ORI_MAX:
        case IVE_THRESH_MODE_ORI_MID_ORI:
            break;
        default:
            throw std::invalid_argument("Unsupported U8 threshold mode.");
    }
    check_threshold_values(low_thresh, high_thresh, 0, 255, min_val, mid_val, max_val, 0, 255,
                           mode >= IVE_THRESH_MODE_MIN_MID_MAX);

    if (mode == IVE_THRESH_MODE_MIN_MID_MAX || mode == IVE_THRESH_MODE_ORI_MID_ORI) {
        // The runtime misselects regions for these modes; one hardware Map applies an exact 256-entry LUT.
        // 运行库对这两个模式的区域选择有误；以一次硬件 Map 应用精确的 256 项查表。
        std::vector<uint8_t> table(256);
        for (int value = 0; value < 256; ++value) {
            if (mode == IVE_THRESH_MODE_MIN_MID_MAX) {
                table[value] = static_cast<uint8_t>(value <= low_thresh ? min_val : (value <= high_thresh ? mid_val : max_val));
            } else {
                table[value] = static_cast<uint8_t>(value > low_thresh && value <= high_thresh ? mid_val : value);
            }
        }
        return map(src, table);
    }

    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_buffers(src.width, src.height);
    copy_to_ive_buffer(src, m_img_u8c1_in1.get());
    IVE_THRESH_U8_CTRL_S control{};
    control.enMode = mode;
    control.u8LowThr = static_cast<RK_U8>(low_thresh);
    control.u8HighThr = static_cast<RK_U8>(high_thresh);
    control.u8MinVal = static_cast<RK_U8>(min_val);
    control.u8MidVal = static_cast<RK_U8>(mid_val);
    control.u8MaxVal = static_cast<RK_U8>(max_val);
    IVE_HANDLE handle{};
    if (RK_MPI_IVE_Thresh(&handle, m_img_u8c1_in1.get(), m_img_u8c1_out1.get(), &control, RK_TRUE) != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_Thresh failed.");
    }
    return copy_from_ive_buffer(m_img_u8c1_out1.get(), src.width, src.height);
}

ImageBuffer IVE::threshold_s16(const ImageBuffer& src, int low_thresh, int high_thresh,
                              IVE_THRESH_S16_MODE_E mode, int min_val, int mid_val, int max_val) {
    check_operator_input(src, visiong::kS16C1Format, sizeof(int16_t));
    bool signed_output = false;
    switch (mode) {
        case IVE_THRESH_S16_MODE_S16_TO_S8_MIN_MID_MAX:
        case IVE_THRESH_S16_MODE_S16_TO_S8_MIN_ORI_MAX:
            signed_output = true;
            break;
        case IVE_THRESH_S16_MODE_S16_TO_U8_MIN_MID_MAX:
        case IVE_THRESH_S16_MODE_S16_TO_U8_MIN_ORI_MAX:
            break;
        default:
            throw std::invalid_argument("Unsupported S16 threshold mode.");
    }
    check_threshold_values(low_thresh, high_thresh, -32768, 32767, min_val, mid_val, max_val,
                           signed_output ? -128 : 0, signed_output ? 127 : 255, true);
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_buffers(src.width, src.height);
    copy_to_ive_buffer(src, m_img_s16c1_out1.get());
    IVE_THRESH_S16_CTRL_S control{};
    control.enMode = mode;
    control.s16LowThr = static_cast<RK_S16>(low_thresh);
    control.s16HighThr = static_cast<RK_S16>(high_thresh);
    // IVE_8BIT_U contains two separate bytes in this SDK, rather than overlapping union members.
    // 此 SDK 的 IVE_8BIT_U 包含两个独立字节，并非成员重叠的联合体。
    if (signed_output) {
        control.un8MinVal.s8Val = static_cast<char>(min_val);
        control.un8MidVal.s8Val = static_cast<char>(mid_val);
        control.un8MaxVal.s8Val = static_cast<char>(max_val);
    } else {
        control.un8MinVal.u8Val = static_cast<RK_U8>(min_val);
        control.un8MidVal.u8Val = static_cast<RK_U8>(mid_val);
        control.un8MaxVal.u8Val = static_cast<RK_U8>(max_val);
    }
    IVE_IMAGE_S output = *m_img_u8c1_out1;
    output.enType = signed_output ? IVE_IMAGE_TYPE_S8C1 : IVE_IMAGE_TYPE_U8C1;
    IVE_HANDLE handle{};
    if (RK_MPI_IVE_Thresh_S16(&handle, m_img_s16c1_out1.get(), &output, &control, RK_TRUE) != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_Thresh_S16 failed.");
    }
    return signed_output ? copy_threshold_s8(output) : copy_from_ive_buffer(&output, src.width, src.height);
}

ImageBuffer IVE::threshold_u16(const ImageBuffer& src, int low_thresh, int high_thresh,
                              IVE_THRESH_U16_MODE_E mode, int min_val, int mid_val, int max_val) {
    check_operator_input(src, visiong::kU16C1Format, sizeof(uint16_t));
    switch (mode) {
        case IVE_THRESH_U16_MODE_U16_TO_U8_MIN_MID_MAX:
        case IVE_THRESH_U16_MODE_U16_TO_U8_MIN_ORI_MAX:
            break;
        default:
            throw std::invalid_argument("Unsupported U16 threshold mode.");
    }
    check_threshold_values(low_thresh, high_thresh, 0, 65535, min_val, mid_val, max_val, 0, 255, true);
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_buffers(src.width, src.height);
    copy_to_ive_buffer(src, m_img_u16c1_in1.get());
    IVE_THRESH_U16_CTRL_S control{};
    control.enMode = mode;
    control.u16LowThr = static_cast<RK_U16>(low_thresh);
    control.u16HighThr = static_cast<RK_U16>(high_thresh);
    control.u8MinVal = static_cast<RK_U8>(min_val);
    control.u8MidVal = static_cast<RK_U8>(mid_val);
    control.u8MaxVal = static_cast<RK_U8>(max_val);
    IVE_HANDLE handle{};
    if (RK_MPI_IVE_Thresh_U16(&handle, m_img_u16c1_in1.get(), m_img_u8c1_out1.get(), &control, RK_TRUE) != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_Thresh_U16 failed.");
    }
    return copy_from_ive_buffer(m_img_u8c1_out1.get(), src.width, src.height);
}

ImageBuffer IVE::dma_transfer(const ImageBuffer& src, IVE_DMA_MODE_E mode, uint64_t value,
                             int hor_seg_size, int elem_size, int ver_seg_rows,
                             int output_width, int output_height) {
    check_operator_input(src, visiong::kGray8Format, 1, "DMA");
    check_range(output_width, 0, 4096, "output_width");
    check_range(output_height, 0, 4096, "output_height");
    bool fill = false;
    int expected_width = src.width;
    int expected_height = src.height;
    switch (mode) {
        case IVE_DMA_MODE_DIRECT_COPY:
            break;
        case IVE_DMA_MODE_INTERVAL_COPY:
            if (hor_seg_size != 2 && hor_seg_size != 3 && hor_seg_size != 4 &&
                hor_seg_size != 8 && hor_seg_size != 16) {
                throw std::invalid_argument("DMA hor_seg_size must be one of 2,3,4,8,16.");
            }
            if (elem_size <= 0 || elem_size >= hor_seg_size) {
                throw std::invalid_argument("DMA elem_size must be positive and smaller than hor_seg_size.");
            }
            check_range(ver_seg_rows, 1, 255, "ver_seg_rows");
            if (src.width % hor_seg_size || src.height % ver_seg_rows) {
                throw std::invalid_argument("DMA interval copy requires complete horizontal segments and vertical row groups.");
            }
            expected_width = src.width / hor_seg_size * elem_size;
            expected_height = src.height / ver_seg_rows;
            break;
        case IVE_DMA_MODE_SET_3BYTE:
            if (value > 0xffffffu) {
                throw std::invalid_argument("DMA SET_3BYTE value must fit in 24 bits.");
            }
            fill = true;
            break;
        case IVE_DMA_MODE_SET_8BYTE:
            fill = true;
            break;
        default:
            throw std::invalid_argument("Unsupported DMA mode.");
    }
    const int width = output_width ? output_width : expected_width;
    const int height = output_height ? output_height : expected_height;
    if (!fill && (width != expected_width || height != expected_height)) {
        throw std::invalid_argument("DMA copy output dimensions must match the dimensions derived from the source and copy controls.");
    }

    std::lock_guard<std::mutex> lock(m_mutex);
    DmaImage output(width, height);
    if (mode == IVE_DMA_MODE_INTERVAL_COPY) {
        // The SDK sample uses a packed interval destination; the allocation still includes aligned row capacity.
        // SDK 示例对间隔复制使用紧凑输出步长，实际分配仍保留对齐行所需的容量。
        output.image.au32Stride[0] = width;
    }
    IVE_DATA_S destination = dma_data(output.image);
    IVE_DATA_S source{};
    IVE_DMA_CTRL_S control{};
    control.enMode = mode;
    control.u64Val = value;
    if (mode == IVE_DMA_MODE_INTERVAL_COPY) {
        control.u8HorSegSize = static_cast<RK_U8>(hor_seg_size);
        control.u8ElemSize = static_cast<RK_U8>(elem_size);
        control.u8VerSegRows = static_cast<RK_U8>(ver_seg_rows);
    }
    if (!fill) {
        ensure_buffers(src.width, src.height);
        copy_to_ive_buffer(src, m_img_u8c1_in1.get());
        source = dma_data(*m_img_u8c1_in1);
    }

    // SET modes fill the first DATA operand; the second operand is unused. The source ImageBuffer stays intact.
    // SET 模式填充第一个 DATA 操作数，第二个操作数不使用；输入 ImageBuffer 保持不变。
    IVE_HANDLE handle{};
    if (RK_MPI_IVE_DMA(&handle, fill ? &destination : &source, fill ? nullptr : &destination,
                       &control, RK_TRUE) != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_DMA failed.");
    }
    return copy_from_ive_buffer(&output.image, width, height);
}

ImageBuffer IVE::filter_ex(const ImageBuffer& src, const std::vector<int8_t>& mask,
                          int norm, IVE_IMAGE_TYPE_E out_format) {
    check_convolution(src, mask, norm);
    const RK_U8 output_mode = filter_output_mode(out_format);
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_buffers(src.width, src.height);
    copy_to_ive_buffer(src, m_img_u8c1_in1.get());
    DmaImage output(src.width, src.height, out_format);
    IVE_FILTER_CTRL_S control{};
    control.u8Norm = static_cast<RK_U8>(norm);
    control.u8OutMode = output_mode;
    std::copy(mask.begin(), mask.end(), control.as8Mask);
    IVE_HANDLE handle{};
    if (RK_MPI_IVE_Filter(&handle, m_img_u8c1_in1.get(), &output.image, &control, RK_TRUE) != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_Filter failed.");
    }
    return out_format == IVE_IMAGE_TYPE_S8C1 ? copy_threshold_s8(output.image) :
        copy_from_ive_buffer(&output.image, src.width, src.height);
}

std::tuple<ImageBuffer, ImageBuffer> IVE::sobel_ex(
    const ImageBuffer& src, const std::vector<int8_t>& mask, int norm,
    IVE_SOBEL_OUT_CTRL_E out_ctrl, IVE_IMAGE_TYPE_E out_format) {
    check_convolution(src, mask, norm);
    const RK_U8 output_mode = filter_output_mode(out_format);
    if (out_ctrl != IVE_SOBEL_OUT_CTRL_BOTH && out_ctrl != IVE_SOBEL_OUT_CTRL_HOR && out_ctrl != IVE_SOBEL_OUT_CTRL_VER) {
        throw std::invalid_argument("Unsupported Sobel output direction.");
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_buffers(src.width, src.height);
    copy_to_ive_buffer(src, m_img_u8c1_in1.get());
    // This runtime requires both destination pointers even in a single-direction mode.
    // 此运行库在单方向模式下同样要求两个输出指针均有效。
    DmaImage horizontal(src.width, src.height, out_format);
    DmaImage vertical(src.width, src.height, out_format);
    IVE_SOBEL_CTRL_S control{};
    control.u8OutCtrl = static_cast<RK_U8>(out_ctrl);
    control.u8Norm = static_cast<RK_U8>(norm);
    control.u8OutMode = output_mode;
    std::copy(mask.begin(), mask.end(), control.as8Mask);
    IVE_HANDLE handle{};
    if (RK_MPI_IVE_Sobel(&handle, m_img_u8c1_in1.get(), &horizontal.image, &vertical.image, &control, RK_TRUE) != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_Sobel failed.");
    }
    auto read = [&](const IVE_IMAGE_S& image) {
        return out_format == IVE_IMAGE_TYPE_S8C1 ? copy_threshold_s8(image) :
            copy_from_ive_buffer(&image, src.width, src.height);
    };
    return std::make_tuple(out_ctrl == IVE_SOBEL_OUT_CTRL_VER ? ImageBuffer() : read(horizontal.image),
                           out_ctrl == IVE_SOBEL_OUT_CTRL_HOR ? ImageBuffer() : read(vertical.image));
}

ImageBuffer IVE::map_typed(const ImageBuffer& src, const std::vector<int>& lut, IVE_MAP_MODE_E mode) {
    check_operator_input(src, visiong::kGray8Format, 1, "Map");
    if (lut.size() != 256) throw std::invalid_argument("Map requires exactly 256 LUT entries.");
    IVE_IMAGE_TYPE_E type;
    int minimum = 0;
    int maximum;
    size_t item_size;
    switch (mode) {
        case IVE_MAP_MODE_U8: type = IVE_IMAGE_TYPE_U8C1; maximum = 255; item_size = 1; break;
        case IVE_MAP_MODE_S16: type = IVE_IMAGE_TYPE_S16C1; minimum = -32768; maximum = 32767; item_size = 2; break;
        case IVE_MAP_MODE_U16: type = IVE_IMAGE_TYPE_U16C1; maximum = 65535; item_size = 2; break;
        default: throw std::invalid_argument("Unsupported Map output mode.");
    }
    std::vector<unsigned char> bytes(256 * item_size);
    for (size_t i = 0; i < lut.size(); ++i) {
        check_range(lut[i], minimum, maximum, "LUT entry");
        if (item_size == 1) bytes[i] = static_cast<unsigned char>(lut[i]);
        else {
            const uint16_t value = static_cast<uint16_t>(lut[i]);
            std::memcpy(bytes.data() + i * 2, &value, 2);
        }
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_buffers(src.width, src.height);
    copy_to_ive_buffer(src, m_img_u8c1_in1.get());
    if (m_mem_info->u32Size < bytes.size()) throw std::runtime_error("Map LUT workspace is too small.");
    IVE_MEM_INFO_S table = *m_mem_info;
    table.u32Size = static_cast<RK_U32>(bytes.size());
    copy_mmz_bytes(table, bytes.data(), bytes.size(), true);
    DmaImage output(src.width, src.height, type);
    IVE_MAP_CTRL_S control{};
    // Both word LUTs preserve all 16 bits; the native U16 branch omits byte-stride conversion.
    // 两种字查表均保留全部 16 位；原生 U16 分支遗漏了字节步长换算。
    control.enMode = mode == IVE_MAP_MODE_U16 ? IVE_MAP_MODE_S16 : mode;
    IVE_IMAGE_S hardware_output = output.image;
    if (mode == IVE_MAP_MODE_U16) hardware_output.enType = IVE_IMAGE_TYPE_S16C1;
    IVE_HANDLE handle{};
    if (RK_MPI_IVE_Map(&handle, m_img_u8c1_in1.get(), &table, &hardware_output, &control, RK_TRUE) != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_Map failed.");
    }
    return copy_from_ive_buffer(&output.image, src.width, src.height);
}

ImageBuffer IVE::add_weighted(const ImageBuffer& src1, const ImageBuffer& src2, int weight1, int weight2) {
    check_gray_pair(src1, src2, "Weighted add");
    check_range(weight1, 0, 65535, "weight1");
    check_range(weight2, 0, 65535, "weight2");
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_buffers(src1.width, src1.height);
    copy_to_ive_buffer(src1, m_img_u8c1_in1.get());
    copy_to_ive_buffer(src2, m_img_u8c1_in2.get());
    IVE_ADD_CTRL_S control{};
    control.u0q16X = static_cast<RK_U16>(weight1);
    control.u0q16Y = static_cast<RK_U16>(weight2);
    IVE_HANDLE handle{};
    if (RK_MPI_IVE_Add(&handle, m_img_u8c1_in1.get(), m_img_u8c1_in2.get(), m_img_u8c1_out1.get(), &control, RK_TRUE) != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_Add failed.");
    }
    return copy_from_ive_buffer(m_img_u8c1_out1.get(), src1.width, src1.height);
}

std::tuple<uint64_t, uint64_t, uint64_t> IVE::ncc_raw(const ImageBuffer& src1, const ImageBuffer& src2) {
    check_gray_pair(src1, src2, "NCC");
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_buffers(src1.width, src1.height);
    copy_to_ive_buffer(src1, m_img_u8c1_in1.get());
    copy_to_ive_buffer(src2, m_img_u8c1_in2.get());
    IVE_NCC_DST_MEM_S result{};
    copy_mmz_bytes(*m_mem_info, &result, sizeof(result), true);
    IVE_HANDLE handle{};
    if (RK_MPI_IVE_NCC(&handle, m_img_u8c1_in1.get(), m_img_u8c1_in2.get(), m_mem_info.get(), RK_TRUE) != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_NCC failed.");
    }
    copy_mmz_bytes(*m_mem_info, &result, sizeof(result), false);
    return std::make_tuple(result.u64Numerator, result.u64QuadSum1, result.u64QuadSum2);
}

std::tuple<ImageBuffer, std::vector<uint32_t>> IVE::equalize_hist_ex(
    const ImageBuffer& src, IVE_EQUALIZE_MODE_E mode, const std::vector<uint32_t>& histogram) {
    check_operator_input(src, visiong::kGray8Format, 1, "EqualizeHist");
    if (mode != IVE_EQUALIZE_MODE_HIST && mode != IVE_EQUALIZE_MODE_EQHIST && mode != IVE_EQUALIZE_MODE_EQHIST_WITH_EXT_HIST) {
        throw std::invalid_argument("Unsupported histogram equalization mode.");
    }
    if (mode == IVE_EQUALIZE_MODE_HIST) {
        if (!histogram.empty()) throw std::invalid_argument("HIST mode does not consume an external histogram.");
        return std::make_tuple(ImageBuffer(), hist(src));
    }
    if (mode == IVE_EQUALIZE_MODE_EQHIST_WITH_EXT_HIST) {
        if (histogram.size() != 256) throw std::invalid_argument("External histogram requires exactly 256 bins.");
        uint64_t total = 0;
        for (uint32_t count : histogram) total += count;
        if (total != static_cast<uint64_t>(src.width) * src.height) {
            throw std::invalid_argument("External histogram bin counts must sum to the image pixel count.");
        }
    } else if (!histogram.empty()) {
        throw std::invalid_argument("EQHIST mode calculates its own histogram; use EQHIST_WITH_EXT_HIST for supplied bins.");
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_buffers(src.width, src.height);
    copy_to_ive_buffer(src, m_img_u8c1_in1.get());
    IVE_EQHIST_CTRL_S control{};
    control.enMode = mode;
    control.u32HistMem = *m_mem_info;
    control.u32HistMem.u32Size = sizeof(IVE_EQUALIZE_RKST_CTRL_MEM_S);
    if (m_mem_info->u32Size < control.u32HistMem.u32Size) {
        throw std::runtime_error("EqualizeHist workspace is too small.");
    }
    if (!histogram.empty()) std::copy(histogram.begin(), histogram.end(), control.u32HistArray);
    IVE_HANDLE handle{};
    if (RK_MPI_IVE_EqualizeHist(&handle, m_img_u8c1_in1.get(), m_img_u8c1_out1.get(), &control, RK_TRUE) != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_EqualizeHist failed.");
    }
    return std::make_tuple(copy_from_ive_buffer(m_img_u8c1_out1.get(), src.width, src.height), std::vector<uint32_t>{});
}

ImageBuffer IVE::cast_8bit_ex(const ImageBuffer& src, IVE_8BIT_TO_8BIT_MODE_E mode,
                             int numerator, int denominator, int bias) {
    bool unsigned_input = false;
    bool signed_output = false;
    bool absolute = false;
    switch (mode) {
        case IVE_8BIT_TO_8BIT_MODE_S8_TO_S8: signed_output = true; break;
        case IVE_8BIT_TO_8BIT_MODE_S8_TO_U8_ABS: absolute = true; break;
        case IVE_8BIT_TO_8BIT_MODE_S8_TO_U8_BIAS: break;
        case IVE_8BIT_TO_8BIT_MODE_U8_TO_U8: unsigned_input = true; break;
        default: throw std::invalid_argument("Unsupported 8-bit cast mode.");
    }
    check_operator_input(src, unsigned_input ? visiong::kGray8Format : visiong::kS8C1Format, 1, "8-bit cast");
    const uint32_t scale = cast_scale(numerator, denominator, 255, bias);
    std::vector<int> table(256);
    for (int bits = 0; bits < 256; ++bits) {
        int value = unsigned_input || bits < 128 ? bits : bits - 256;
        if (absolute) value = std::abs(value);
        const int result = static_cast<int>(std::clamp<int64_t>(cast_round(value, scale) + bias,
                                                               signed_output ? -128 : 0, signed_output ? 127 : 255));
        table[bits] = static_cast<uint8_t>(result);
    }
    // The public 8-bit cast wrapper discards signed modes. One hardware Map implements the complete byte domain.
    // 公开的 8 位转换封装忽略有符号模式；一次硬件 Map 覆盖完整的字节输入域。
    if (unsigned_input) return map_typed(src, table, IVE_MAP_MODE_U8);
    visiong::bufstate::prepare_cpu_read(src);
    std::vector<unsigned char> bytes(static_cast<size_t>(src.width) * src.height);
    for (int y = 0; y < src.height; ++y) {
        std::memcpy(bytes.data() + static_cast<size_t>(y) * src.width,
                    static_cast<const unsigned char*>(src.get_data()) + static_cast<size_t>(y) * src.w_stride, src.width);
    }
    ImageBuffer raw(src.width, src.height, visiong::kGray8Format, std::move(bytes));
    ImageBuffer output = map_typed(raw, table, IVE_MAP_MODE_U8);
    if (signed_output) output.format = visiong::kS8C1Format;
    return output;
}

ImageBuffer IVE::cast_16bit_ex(const ImageBuffer& src, IVE_16BIT_TO_8BIT_MODE_E mode,
                              int numerator, int denominator, int bias) {
    bool unsigned_input = false;
    bool signed_output = false;
    bool absolute = false;
    switch (mode) {
        case IVE_16BIT_TO_8BIT_MODE_S16_TO_S8: signed_output = true; break;
        case IVE_16BIT_TO_8BIT_MODE_S16_TO_U8_ABS: absolute = true; break;
        case IVE_16BIT_TO_8BIT_MODE_S16_TO_U8_BIAS: break;
        case IVE_16BIT_TO_8BIT_MODE_U16_TO_U8: unsigned_input = true; break;
        default: throw std::invalid_argument("Unsupported 16-bit cast mode.");
    }
    check_operator_input(src, unsigned_input ? visiong::kU16C1Format : visiong::kS16C1Format, sizeof(uint16_t), "16-bit cast");
    const uint32_t scale = cast_scale(numerator, denominator, 65535, bias);
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_buffers(src.width, src.height);
    if (unsigned_input) {
        copy_to_ive_buffer(src, m_img_u16c1_in1.get());
    } else {
        visiong::bufstate::prepare_cpu_read(src);
        auto* target = reinterpret_cast<RK_U16*>(m_img_u16c1_in1->au64VirAddr[0]);
        if (visiong::ive_internal::mmz_flush_start(target) != RK_SUCCESS) {
            throw std::runtime_error("16-bit cast failed to begin input staging.");
        }
        for (int y = 0; y < src.height; ++y) {
            const auto* source = static_cast<const int16_t*>(src.get_data()) + static_cast<size_t>(y) * src.w_stride;
            auto* row = target + static_cast<size_t>(y) * m_img_u16c1_in1->au32Stride[0];
            for (int x = 0; x < src.width; ++x) row[x] = static_cast<RK_U16>(std::abs(static_cast<int>(source[x])));
        }
        if (visiong::ive_internal::mmz_flush_end(target) != RK_SUCCESS) {
            throw std::runtime_error("16-bit cast failed to end input staging.");
        }
    }
    IVE_16BIT_TO_8BIT_CTRL_S control{};
    control.enMode = IVE_16BIT_TO_8BIT_MODE_U16_TO_U8;
    control.u16Denominator = static_cast<RK_U16>(denominator);
    control.u8Numerator = static_cast<RK_U8>(numerator);
    control.s8Bias = static_cast<RK_S8>(signed_output ? 0 : bias);
    IVE_HANDLE handle{};
    IVE_IMAGE_S cast_source = *m_img_u16c1_in1;
    // Cast HAL adds width to this field to obtain byte stride. / Cast HAL 将宽度加到此字段以获得字节步长。
    cast_source.au32Stride[0] = 2 * m_img_u16c1_in1->au32Stride[0] - cast_source.u32Width;
    if (RK_MPI_IVE_16BitTo8Bit(&handle, &cast_source, m_img_u8c1_out1.get(), &control, RK_TRUE) != RK_SUCCESS) {
        throw std::runtime_error("RK_MPI_IVE_16BitTo8Bit failed.");
    }
    ImageBuffer output = copy_from_ive_buffer(m_img_u8c1_out1.get(), src.width, src.height);
    if (unsigned_input || absolute) return output;

    // Restore signs after one unsigned hardware scaling call. No per-pixel software division is needed.
    // 在一次无符号硬件缩放后恢复符号，无需逐像素软件除法。
    auto* result = static_cast<unsigned char*>(output.get_data());
    for (int y = 0; y < src.height; ++y) {
        const auto* source = static_cast<const int16_t*>(src.get_data()) + static_cast<size_t>(y) * src.w_stride;
        auto* row = result + static_cast<size_t>(y) * output.w_stride;
        for (int x = 0; x < src.width; ++x) {
            const bool negative = source[x] < 0;
            const int correction = cast_negative_half(source[x], scale) ? 1 : 0;
            if (signed_output) {
                const bool saturated_negative = negative &&
                    static_cast<uint64_t>(-static_cast<int>(source[x])) * scale >= (uint64_t{255} << 24) + 0x800000u;
                const int value = saturated_negative ? -128 :
                    (negative ? -static_cast<int>(row[x]) + correction : row[x]) + bias;
                row[x] = static_cast<uint8_t>(std::clamp(value, -128, 127));
            } else if (negative) {
                row[x] = static_cast<uint8_t>(std::clamp(2 * bias - static_cast<int>(row[x]) + correction, 0, 255));
            }
        }
    }
    if (signed_output) output.format = visiong::kS8C1Format;
    return output;
}
