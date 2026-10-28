// SPDX-License-Identifier: LGPL-3.0-or-later
#include "modules/internal/ive_runtime_compat.h"
#include "modules/internal/ive_digest.h"

#include <cstddef>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <pthread.h>
#include <stdexcept>

namespace visiong::ive_internal {
namespace {

struct GmmControl {
    uint8_t algorithm, pic_format, first, bg_output, max_models, var_factor, learning_mode, update_var;
    uint16_t weight_init, weight_add, weight_reduce, weight_threshold;
    uint8_t var_generation, padding0;
    uint16_t bg_ratio, init_var, min_var, max_var;
    uint8_t var_threshold, padding1;
    uint32_t current_models;
    uint32_t reserved[2];
};
static_assert(sizeof(GmmControl) == 40 && offsetof(GmmControl, current_models) == 28, "GMM control layout");
static_assert(sizeof(IVE_IMAGE_S) == 80 && offsetof(IVE_IMAGE_S, au32Stride) == 48, "IVE image layout");

struct GmmRuntime {
    using Setup = int (*)(int, void*, IVE_IMAGE_S*, IVE_IMAGE_S*);
    using Stream = int (*)(bool, void*);
    Setup setup = nullptr;
    Stream stream = nullptr;
    void* postprocess = nullptr;
    pthread_mutex_t* mutex = nullptr;
};

const GmmRuntime& checked_runtime() {
#if defined(__arm__) && !defined(__aarch64__)
    static const GmmRuntime runtime = [] {
        const char* error = "Multi-model GMM requires a verified RV1106 IVE runtime.";
        void* function = dlsym(RTLD_DEFAULT, "RK_MPI_IVE_GMM");
        Dl_info location{};
        if (!function || !dladdr(function, &location) || !location.dli_fbase || !location.dli_fname) {
            throw std::runtime_error(error);
        }
        std::ifstream file(location.dli_fname, std::ios::binary | std::ios::ate);
        if (!file || file.tellg() != std::streamoff(135976)) throw std::runtime_error(error);
        std::vector<uint8_t> bytes(135976);
        file.seekg(0);
        if (!file.read(reinterpret_cast<char*>(bytes.data()), bytes.size())) throw std::runtime_error(error);
        const auto digest = runtime_digest::sha256_hex(std::move(bytes));
        if (digest != "2446c5b1720c083b89338c33cdf3f289c8fc94b29386a0404481881a06cc3455" &&
            digest != "80c5c104f5bf02b88dafd1ed0867d90f2439fded2464174080d30ff97bd03409") {
            throw std::runtime_error(error);
        }
        const auto base = reinterpret_cast<uintptr_t>(location.dli_fbase);
        const auto word = [base](size_t offset) {
            uint32_t value;
            std::memcpy(&value, reinterpret_cast<const void*>(base + offset), sizeof(value));
            return value;
        };
        GmmRuntime result;
        result.setup = reinterpret_cast<GmmRuntime::Setup>(dlsym(RTLD_DEFAULT, "rve_entity_setup"));
        result.stream = reinterpret_cast<GmmRuntime::Stream>(dlsym(RTLD_DEFAULT, "rve_pipeline_stream_on"));
        result.postprocess = dlsym(RTLD_DEFAULT, "rk_ive_post_process");
        if (reinterpret_cast<uintptr_t>(function) - base != 0x6f24 ||
            reinterpret_cast<uintptr_t>(result.setup) - base != 0x1369c ||
            reinterpret_cast<uintptr_t>(result.stream) - base != 0x1b18c ||
            reinterpret_cast<uintptr_t>(result.postprocess) - base != 0xfd10 ||
            word(0x6f24) != 0xe92d4810 || word(0x1369c) != 0xe92d4810 ||
            word(0x7300) != 0xe08f3003 || word(0x7398) != 0x00029b80) {
            throw std::runtime_error(error);
        }
        result.mutex = reinterpret_cast<pthread_mutex_t*>(base + 0x30e88);
        return result;
    }();
    return runtime;
#else
    throw std::runtime_error("Multi-model GMM requires a verified RV1106 IVE runtime.");
#endif
}

IVE_IMAGE_S model_image(const IVE_MEM_INFO_S& memory) {
    IVE_IMAGE_S image{};
    image.au64PhyAddr[0] = memory.u64PhyAddr;
    image.au64VirAddr[0] = memory.u64VirAddr;
    image.au32Stride[0] = memory.u32Size;
    image.u32Width = memory.u32Size;
    image.u32Height = 1;
    image.enType = IVE_IMAGE_TYPE_U8C1;
    return image;
}

class RuntimeLock {
public:
    explicit RuntimeLock(pthread_mutex_t* mutex) : mutex_(mutex) {
        if (pthread_mutex_lock(mutex_)) throw std::runtime_error("Failed to lock the IVE runtime.");
    }
    ~RuntimeLock() { pthread_mutex_unlock(mutex_); }
private:
    pthread_mutex_t* mutex_;
};

}  // namespace

void require_gmm_separate_output_runtime() { (void)checked_runtime(); }

RK_S32 gmm_separate_output(const IVE_IMAGE_S& source, const IVE_IMAGE_S* factor,
                          IVE_IMAGE_S& foreground, IVE_IMAGE_S& background,
                          IVE_IMAGE_S& matched, const IVE_MEM_INFO_S& model,
                          const IVE_MEM_INFO_S& output, int max_models, bool first_frame,
                          RK_U32 current_models) {
    const auto& runtime = checked_runtime();
    IVE_IMAGE_S inputs[10]{};
    IVE_IMAGE_S outputs[4]{};
    inputs[0] = source;
    if (factor) inputs[1] = *factor;
    inputs[2] = model_image(model);
    inputs[3] = matched;
    outputs[0] = foreground;
    outputs[1] = background;
    outputs[2] = matched;
    outputs[3] = model_image(output);
    GmmControl control{};
    control.algorithm = factor ? 1 : 0;
    control.first = first_frame;
    control.bg_output = 1;
    control.max_models = max_models;
    control.var_factor = factor ? 1 : 0;
    control.learning_mode = factor ? 0 : 1;
    control.update_var = 1;
    control.weight_init = 16;
    control.weight_add = 4;
    control.weight_reduce = 1016;
    control.weight_threshold = 4;
    control.var_generation = 9;
    control.bg_ratio = 712;
    control.init_var = 225;
    control.min_var = 200;
    control.max_var = 512;
    control.var_threshold = 16;
    control.current_models = current_models;
    RuntimeLock lock(runtime.mutex);
    // Keep the vendor scheduler, postprocessor and shared lock; only separate model output.
    // 保留厂商调度器、后处理及共享锁，仅分离模型输出。
    const int result = runtime.setup(0, &control, inputs, outputs);
    if (result < 0) return result;
    return runtime.stream(true, runtime.postprocess) < 0 ? RK_FAILURE : RK_SUCCESS;
}

}  // namespace visiong::ive_internal
