// SPDX-License-Identifier: LGPL-3.0-or-later
#include "rknn_engine.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string.h>

#include "engine_helper.h"
#include "logging.h"

namespace {

using SteadyClock = std::chrono::steady_clock;

static inline double elapsed_ms(const SteadyClock::time_point& start, const SteadyClock::time_point& end)
{
    return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
}

static bool rknn_profile_enabled()
{
    static const bool enabled = []() {
        const char* env = std::getenv("VISIONG_PROFILE_RKNN");
        return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

struct RKRunTiming
{
    double query_in_ms = 0.0;
    double prep_in_ms = 0.0;
    double set_in_ms = 0.0;
    double query_out_ms = 0.0;
    double prep_out_ms = 0.0;
    double set_out_ms = 0.0;
    double run_ms = 0.0;
    double copy_out_ms = 0.0;
    double cleanup_ms = 0.0;
    double total_ms = 0.0;
};

struct RKProfileBucket
{
    uint64_t calls = 0;
    double query_in_ms = 0.0;
    double prep_in_ms = 0.0;
    double set_in_ms = 0.0;
    double query_out_ms = 0.0;
    double prep_out_ms = 0.0;
    double set_out_ms = 0.0;
    double run_ms = 0.0;
    double copy_out_ms = 0.0;
    double cleanup_ms = 0.0;
    double total_ms = 0.0;
};

static const char* classify_rknn_run(const std::vector<tensor_data_s>& inputs, const std::vector<tensor_data_s>& outputs)
{
    if (inputs.size() == 2 && outputs.size() == 2)
    {
        return "H";
    }

    if (inputs.size() == 1 && outputs.size() == 1)
    {
        return (inputs[0].attr.size > 100000) ? "X" : "T";
    }

    return "UNK";
}

static void record_rknn_profile(const char* tag, const RKRunTiming& t)
{
    static std::mutex profile_mu;
    static RKProfileBucket bucket_t;
    static RKProfileBucket bucket_x;
    static RKProfileBucket bucket_h;
    static RKProfileBucket bucket_unk;

    RKProfileBucket* bucket = &bucket_unk;
    if (std::strcmp(tag, "T") == 0)
    {
        bucket = &bucket_t;
    }
    else if (std::strcmp(tag, "X") == 0)
    {
        bucket = &bucket_x;
    }
    else if (std::strcmp(tag, "H") == 0)
    {
        bucket = &bucket_h;
    }

    std::lock_guard<std::mutex> lock(profile_mu);
    bucket->calls += 1;
    bucket->query_in_ms += t.query_in_ms;
    bucket->prep_in_ms += t.prep_in_ms;
    bucket->set_in_ms += t.set_in_ms;
    bucket->query_out_ms += t.query_out_ms;
    bucket->prep_out_ms += t.prep_out_ms;
    bucket->set_out_ms += t.set_out_ms;
    bucket->run_ms += t.run_ms;
    bucket->copy_out_ms += t.copy_out_ms;
    bucket->cleanup_ms += t.cleanup_ms;
    bucket->total_ms += t.total_ms;

    if ((bucket->calls % 240) == 0)
    {
        const double inv = 1.0 / static_cast<double>(bucket->calls);
        NN_LOG_INFO("[RKNNProfile][%s] calls=%llu avg_ms: total=%.4f query_in=%.4f prep_in=%.4f set_in=%.4f query_out=%.4f prep_out=%.4f set_out=%.4f run=%.4f copy_out=%.4f cleanup=%.4f",
                    tag,
                    static_cast<unsigned long long>(bucket->calls),
                    bucket->total_ms * inv,
                    bucket->query_in_ms * inv,
                    bucket->prep_in_ms * inv,
                    bucket->set_in_ms * inv,
                    bucket->query_out_ms * inv,
                    bucket->prep_out_ms * inv,
                    bucket->set_out_ms * inv,
                    bucket->run_ms * inv,
                    bucket->copy_out_ms * inv,
                    bucket->cleanup_ms * inv);
    }
}

} // namespace

nn_error_e RKEngine::LoadModelFile(const char *model_file)
{
    int model_len = 0;
    auto model = load_model(model_file, &model_len);
    if (model == nullptr)
    {
        NN_LOG_ERROR("load model file %s fail!", model_file);
        return NN_LOAD_MODEL_FAIL;
    }

    int ret = rknn_init(&rknn_ctx_, model, model_len, 0, NULL);
    free(model);
    model = nullptr;
    if (ret < 0)
    {
        NN_LOG_ERROR("rknn_init fail! ret=%d", ret);
        return NN_RKNN_INIT_FAIL;
    }

    NN_LOG_INFO("rknn_init success!");
    ctx_created_ = true;

    rknn_sdk_version version;
    ret = rknn_query(rknn_ctx_, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0)
    {
        NN_LOG_ERROR("rknn_query fail! ret=%d", ret);
        return NN_RKNN_QUERY_FAIL;
    }
    NN_LOG_INFO("RKNN API version: %s", version.api_version);
    NN_LOG_INFO("RKNN Driver version: %s", version.drv_version);

    rknn_input_output_num io_num;
    ret = rknn_query(rknn_ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        NN_LOG_ERROR("rknn_query fail! ret=%d", ret);
        return NN_RKNN_QUERY_FAIL;
    }
    NN_LOG_INFO("model input num: %d, output num: %d", io_num.n_input, io_num.n_output);

    input_num_ = io_num.n_input;
    output_num_ = io_num.n_output;

    NN_LOG_INFO("input tensors:");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (uint32_t i = 0; i < io_num.n_input; ++i)
    {
        input_attrs[i].index = i;
        ret = rknn_query(rknn_ctx_, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(input_attrs[i]));
        if (ret != RKNN_SUCC)
        {
            NN_LOG_ERROR("rknn_query fail! ret=%d", ret);
            return NN_RKNN_QUERY_FAIL;
        }
        print_tensor_attr(&(input_attrs[i]));
        NN_LOG_INFO("    input_stride: w_stride=%u, h_stride=%u, size_with_stride=%u, pass_through=%u",
                    input_attrs[i].w_stride,
                    input_attrs[i].h_stride,
                    input_attrs[i].size_with_stride,
                    input_attrs[i].pass_through);
        in_shapes_.push_back(rknn_tensor_attr_convert(input_attrs[i]));
    }

    NN_LOG_INFO("output tensors:");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (uint32_t i = 0; i < io_num.n_output; ++i)
    {
        output_attrs[i].index = i;
        ret = rknn_query(rknn_ctx_, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(output_attrs[i]));
        if (ret != RKNN_SUCC)
        {
            NN_LOG_ERROR("rknn_query fail! ret=%d", ret);
            return NN_RKNN_QUERY_FAIL;
        }
        print_tensor_attr(&(output_attrs[i]));
        out_shapes_.push_back(rknn_tensor_attr_convert(output_attrs[i]));
    }

    return NN_SUCCESS;
}

const std::vector<tensor_attr_s> &RKEngine::GetInputShapes()
{
    return in_shapes_;
}

const std::vector<tensor_attr_s> &RKEngine::GetOutputShapes()
{
    return out_shapes_;
}

nn_error_e RKEngine::Run(std::vector<tensor_data_s> &inputs, std::vector<tensor_data_s> &outputs, bool want_float)
{
    (void)want_float;

    if (inputs.size() != static_cast<size_t>(input_num_))
    {
        NN_LOG_ERROR("inputs num not match! inputs.size()=%zu, input_num_=%u", inputs.size(), input_num_);
        return NN_IO_NUM_NOT_MATCH;
    }
    if (outputs.size() != static_cast<size_t>(output_num_))
    {
        NN_LOG_ERROR("outputs num not match! outputs.size()=%zu, output_num_=%u", outputs.size(), output_num_);
        return NN_IO_NUM_NOT_MATCH;
    }

    const bool do_profile = rknn_profile_enabled();
    const char* profile_tag = "UNK";
    RKRunTiming timing;
    SteadyClock::time_point total_start;
    if (do_profile)
    {
        profile_tag = classify_rknn_run(inputs, outputs);
        total_start = SteadyClock::now();
    }

    std::vector<rknn_tensor_mem*> input_mems(inputs.size(), nullptr);
    std::vector<rknn_tensor_mem*> output_mems(outputs.size(), nullptr);

    auto cleanup_io_mem = [&]() {
        for (auto *mem : input_mems)
        {
            if (mem != nullptr)
            {
                rknn_destroy_mem(rknn_ctx_, mem);
            }
        }
        for (auto *mem : output_mems)
        {
            if (mem != nullptr)
            {
                rknn_destroy_mem(rknn_ctx_, mem);
            }
        }
    };

    for (size_t i = 0; i < inputs.size(); ++i)
    {
        rknn_tensor_attr in_attr;
        memset(&in_attr, 0, sizeof(in_attr));
        in_attr.index = static_cast<uint32_t>(i);

        int ret = 0;
        if (do_profile)
        {
            const SteadyClock::time_point stage_start = SteadyClock::now();
            ret = rknn_query(rknn_ctx_, RKNN_QUERY_NATIVE_INPUT_ATTR, &in_attr, sizeof(in_attr));
            timing.query_in_ms += elapsed_ms(stage_start, SteadyClock::now());
        }
        else
        {
            ret = rknn_query(rknn_ctx_, RKNN_QUERY_NATIVE_INPUT_ATTR, &in_attr, sizeof(in_attr));
        }
        if (ret != RKNN_SUCC)
        {
            NN_LOG_ERROR("query native input attr failed, ret=%d, idx=%d", ret, i);
            cleanup_io_mem();
            return NN_RKNN_INPUT_SET_FAIL;
        }

        if (in_attr.fmt == RKNN_TENSOR_NC1HWC2)
        {
            in_attr.pass_through = 1;
        }
        else
        {
            in_attr.type = rknn_type_convert(inputs[i].attr.type);
            in_attr.fmt = rknn_layout_convert(inputs[i].attr.layout);
            in_attr.pass_through = 0;
        }

        const uint32_t mem_size = (in_attr.size_with_stride > 0) ? in_attr.size_with_stride : in_attr.size;
        if (do_profile)
        {
            const SteadyClock::time_point stage_start = SteadyClock::now();
            input_mems[i] = rknn_create_mem(rknn_ctx_, mem_size);
            if (input_mems[i] == nullptr)
            {
                NN_LOG_ERROR("rknn_create_mem input failed, idx=%d, size=%u", i, mem_size);
                cleanup_io_mem();
                return NN_RKNN_INPUT_SET_FAIL;
            }

            memset(input_mems[i]->virt_addr, 0, mem_size);
            const uint32_t copy_size = (inputs[i].attr.size < mem_size) ? inputs[i].attr.size : mem_size;
            memcpy(input_mems[i]->virt_addr, inputs[i].data, copy_size);
            timing.prep_in_ms += elapsed_ms(stage_start, SteadyClock::now());
        }
        else
        {
            input_mems[i] = rknn_create_mem(rknn_ctx_, mem_size);
            if (input_mems[i] == nullptr)
            {
                NN_LOG_ERROR("rknn_create_mem input failed, idx=%d, size=%u", i, mem_size);
                cleanup_io_mem();
                return NN_RKNN_INPUT_SET_FAIL;
            }

            memset(input_mems[i]->virt_addr, 0, mem_size);
            const uint32_t copy_size = (inputs[i].attr.size < mem_size) ? inputs[i].attr.size : mem_size;
            memcpy(input_mems[i]->virt_addr, inputs[i].data, copy_size);
        }

        if (do_profile)
        {
            const SteadyClock::time_point stage_start = SteadyClock::now();
            ret = rknn_set_io_mem(rknn_ctx_, input_mems[i], &in_attr);
            timing.set_in_ms += elapsed_ms(stage_start, SteadyClock::now());
        }
        else
        {
            ret = rknn_set_io_mem(rknn_ctx_, input_mems[i], &in_attr);
        }
        if (ret < 0)
        {
            NN_LOG_ERROR("rknn_set_io_mem input failed, ret=%d, idx=%d", ret, i);
            cleanup_io_mem();
            return NN_RKNN_INPUT_SET_FAIL;
        }
    }

    for (size_t i = 0; i < outputs.size(); ++i)
    {
        rknn_tensor_attr out_attr;
        memset(&out_attr, 0, sizeof(out_attr));
        out_attr.index = static_cast<uint32_t>(i);

        int ret = 0;
        if (do_profile)
        {
            const SteadyClock::time_point stage_start = SteadyClock::now();
            ret = rknn_query(rknn_ctx_, RKNN_QUERY_OUTPUT_ATTR, &out_attr, sizeof(out_attr));
            timing.query_out_ms += elapsed_ms(stage_start, SteadyClock::now());
        }
        else
        {
            ret = rknn_query(rknn_ctx_, RKNN_QUERY_OUTPUT_ATTR, &out_attr, sizeof(out_attr));
        }
        if (ret != RKNN_SUCC)
        {
            NN_LOG_ERROR("query output attr failed, ret=%d, idx=%d", ret, i);
            cleanup_io_mem();
            return NN_RKNN_OUTPUT_GET_FAIL;
        }

        out_attr.type = rknn_type_convert(outputs[i].attr.type);
        out_attr.fmt = rknn_layout_convert(outputs[i].attr.layout);
        out_attr.pass_through = 0;

        const uint32_t mem_size = (out_attr.size_with_stride > 0) ? out_attr.size_with_stride : out_attr.size;
        if (do_profile)
        {
            const SteadyClock::time_point stage_start = SteadyClock::now();
            output_mems[i] = rknn_create_mem(rknn_ctx_, mem_size);
            if (output_mems[i] == nullptr)
            {
                NN_LOG_ERROR("rknn_create_mem output failed, idx=%d, size=%u", i, mem_size);
                cleanup_io_mem();
                return NN_RKNN_OUTPUT_GET_FAIL;
            }
            timing.prep_out_ms += elapsed_ms(stage_start, SteadyClock::now());
        }
        else
        {
            output_mems[i] = rknn_create_mem(rknn_ctx_, mem_size);
            if (output_mems[i] == nullptr)
            {
                NN_LOG_ERROR("rknn_create_mem output failed, idx=%d, size=%u", i, mem_size);
                cleanup_io_mem();
                return NN_RKNN_OUTPUT_GET_FAIL;
            }
        }

        if (do_profile)
        {
            const SteadyClock::time_point stage_start = SteadyClock::now();
            ret = rknn_set_io_mem(rknn_ctx_, output_mems[i], &out_attr);
            timing.set_out_ms += elapsed_ms(stage_start, SteadyClock::now());
        }
        else
        {
            ret = rknn_set_io_mem(rknn_ctx_, output_mems[i], &out_attr);
        }
        if (ret < 0)
        {
            NN_LOG_ERROR("rknn_set_io_mem output failed, ret=%d, idx=%d", ret, i);
            cleanup_io_mem();
            return NN_RKNN_OUTPUT_GET_FAIL;
        }
    }

    int ret = 0;
    if (do_profile)
    {
        const SteadyClock::time_point stage_start = SteadyClock::now();
        ret = rknn_run(rknn_ctx_, nullptr);
        timing.run_ms += elapsed_ms(stage_start, SteadyClock::now());
    }
    else
    {
        ret = rknn_run(rknn_ctx_, nullptr);
    }
    if (ret < 0)
    {
        NN_LOG_ERROR("rknn_run fail! ret=%d", ret);
        cleanup_io_mem();
        return NN_RKNN_RUNTIME_ERROR;
    }

    if (do_profile)
    {
        const SteadyClock::time_point stage_start = SteadyClock::now();
        for (size_t i = 0; i < outputs.size(); ++i)
        {
            if (output_mems[i] == nullptr)
            {
                cleanup_io_mem();
                return NN_RKNN_OUTPUT_GET_FAIL;
            }
            const uint32_t mem_size = output_mems[i]->size;
            const uint32_t copy_size = (outputs[i].attr.size < mem_size) ? outputs[i].attr.size : mem_size;
            memcpy(outputs[i].data, output_mems[i]->virt_addr, copy_size);
        }
        timing.copy_out_ms += elapsed_ms(stage_start, SteadyClock::now());
    }
    else
    {
        for (size_t i = 0; i < outputs.size(); ++i)
        {
            if (output_mems[i] == nullptr)
            {
                cleanup_io_mem();
                return NN_RKNN_OUTPUT_GET_FAIL;
            }
            const uint32_t mem_size = output_mems[i]->size;
            const uint32_t copy_size = (outputs[i].attr.size < mem_size) ? outputs[i].attr.size : mem_size;
            memcpy(outputs[i].data, output_mems[i]->virt_addr, copy_size);
        }
    }

    if (do_profile)
    {
        const SteadyClock::time_point cleanup_start = SteadyClock::now();
        cleanup_io_mem();
        const SteadyClock::time_point cleanup_end = SteadyClock::now();
        timing.cleanup_ms += elapsed_ms(cleanup_start, cleanup_end);
        timing.total_ms += elapsed_ms(total_start, cleanup_end);
        record_rknn_profile(profile_tag, timing);
    }
    else
    {
        cleanup_io_mem();
    }
    return NN_SUCCESS;
}

RKEngine::~RKEngine()
{
    if (ctx_created_)
    {
        rknn_destroy(rknn_ctx_);
        NN_LOG_INFO("rknn context destroyed!");
    }
}

std::shared_ptr<NNEngine> CreateRKNNEngine()
{
    return std::make_shared<RKEngine>();
}

