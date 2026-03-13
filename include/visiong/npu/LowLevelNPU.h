// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_NPU_LOWLEVELNPU_H
#define VISIONG_NPU_LOWLEVELNPU_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

struct LowLevelTensorInfo {
    int index = 0;
    std::string name;
    std::vector<int64_t> dims;
    std::string format;
    std::string type;
    std::string quant_type;
    int32_t zero_point = 0;
    float scale = 0.0f;
    uint32_t num_elements = 0;
    uint32_t size_bytes = 0;
    uint32_t size_with_stride_bytes = 0;
    uint32_t w_stride = 0;
    uint32_t h_stride = 0;
    bool pass_through = false;
};

class ImageBuffer;
struct LowLevelNPUImpl;

class LowLevelNPU {
public:
    explicit LowLevelNPU(const std::string& model_path, uint32_t init_flags = 0);
    ~LowLevelNPU();

    LowLevelNPU(const LowLevelNPU&) = delete;
    LowLevelNPU& operator=(const LowLevelNPU&) = delete;
    LowLevelNPU(LowLevelNPU&&) = delete;
    LowLevelNPU& operator=(LowLevelNPU&&) = delete;

    bool is_initialized() const;
    int num_inputs() const;
    int num_outputs() const;

    std::vector<LowLevelTensorInfo> input_tensors() const;
    std::vector<LowLevelTensorInfo> output_tensors() const;
    LowLevelTensorInfo input_tensor(int index) const;
    LowLevelTensorInfo output_tensor(int index) const;

    std::vector<int64_t> input_shape(int index) const;
    std::vector<int64_t> output_shape(int index) const;

    std::pair<std::string, std::string> sdk_versions() const;
    int64_t last_run_us() const;

    void set_core_mask(const std::string& core_mask);
    void set_input_attr(int index,
                        const std::string& tensor_type,
                        const std::string& tensor_format,
                        bool pass_through);
    void reset_input_attr(int index);

    void set_input_buffer(int index,
                          const void* data,
                          size_t bytes,
                          bool zero_pad = true,
                          bool sync_to_device = true);
    void set_input_from_float(int index,
                              const float* data,
                              size_t count,
                              bool quantize_if_needed = true,
                              bool zero_pad = true,
                              bool sync_to_device = true);

    void set_input_image(int index,
                         const ImageBuffer& image,
                         const std::string& color_order = "rgb",
                         bool keep_aspect = true,
                         int pad_value = 114,
                         bool driver_convert = true);

    void sync_input_to_device(int index);
    void sync_output_from_device(int index);
    void sync_all_outputs_from_device();

    void run(bool sync_outputs = true,
             bool non_block = false,
             int timeout_ms = 0);
    void wait(int timeout_ms = 0);

    std::vector<uint8_t> output_bytes(int index,
                                      bool with_stride = false,
                                      bool sync_from_device = true) const;
    std::vector<float> output_float(int index,
                                    bool dequantize_if_needed = true,
                                    bool sync_from_device = true) const;

    int input_dma_fd(int index) const;
    int output_dma_fd(int index) const;

private:
    int check_input_index_locked(int index) const;
    int check_output_index_locked(int index) const;

    void sync_input_to_device_locked(int index);
    void sync_output_from_device_locked(int index) const;
    void query_perf_run_locked();

    std::unique_ptr<LowLevelNPUImpl> m_impl;
};

#endif  // VISIONG_NPU_LOWLEVELNPU_H

