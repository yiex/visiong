// SPDX-License-Identifier: LGPL-3.0-or-later
#include "visiong/npu/NPU.h"
#include "npu/internal/npu_common.h"
#include "visiong/core/ImageBuffer.h"
#include "visiong/core/BufferStateMachine.h"
#include "visiong/core/RgaHelper.h"
#include "core/internal/rga_utils.h"
#include "im2d.h"
#include "internal/models/yolov5.h"
#include "internal/models/yolo11.h"
#include "internal/models/yolo11_seg.h"
#include "internal/models/yolo11_pose.h"
#include "internal/models/retinaface.h"
#include "internal/models/facenet.h"
#include "internal/models/lprnet.h"
#include "internal/rknn_model_utils.h"
#include "visiong/common/pixel_format.h"
#include "common/internal/string_utils.h"
#include "core/internal/logger.h"
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>


namespace {

struct SourceDmaContext {
    std::unique_ptr<RgaDmaBuffer> uploaded_dma;
    std::unique_ptr<RgaDmaBuffer> wrapped_dma;
    const RgaDmaBuffer* current = nullptr;
};

void upload_image_to_dma(const ImageBuffer& image, RgaDmaBuffer& dma) {
    const int bpp = get_bpp_for_format(image.format);
    copy_data_with_stride(
        dma.get_vir_addr(),
        dma.get_wstride() * bpp / 8,
        image.get_data(),
        image.w_stride * bpp / 8,
        image.height,
        image.width * bpp / 8);
    visiong::bufstate::mark_cpu_write(dma);
}

SourceDmaContext prepare_source_dma(const ImageBuffer& image) {
    SourceDmaContext ctx;
    if (image.is_zero_copy() && image.get_dma_fd() >= 0) {
        // Non-owning DMA wrapper over existing ImageBuffer storage. / 对现有 ImageBuffer 存储的非拥有型 DMA 封装。
        ctx.wrapped_dma = std::make_unique<RgaDmaBuffer>(
            image.get_dma_fd(),
            const_cast<void*>(image.get_data()),
            image.get_size(),
            image.width,
            image.height,
            static_cast<int>(image.format),
            image.w_stride,
            image.h_stride);
        ctx.current = ctx.wrapped_dma.get();
        return ctx;
    }

    ctx.uploaded_dma =
        std::make_unique<RgaDmaBuffer>(image.width, image.height, static_cast<int>(image.format));
    upload_image_to_dma(image, *ctx.uploaded_dma);
    ctx.current = ctx.uploaded_dma.get();
    return ctx;
}

bool ensure_dma_cache(std::unique_ptr<RgaDmaBuffer>* cache, int width, int height, int format) {
    if (cache == nullptr) {
        return false;
    }
    if (!*cache || (*cache)->get_width() != width || (*cache)->get_height() != height ||
        (*cache)->get_mpi_format() != format) {
        *cache = std::make_unique<RgaDmaBuffer>(width, height, format);
    }
    return true;
}

PIXEL_FORMAT_E parse_model_input_format(const std::string& model_format) {
    const std::string normalized = visiong::to_lower_copy(model_format);
    if (normalized.empty() || normalized == "bgr") {
        return RK_FMT_BGR888;
    }
    if (normalized == "rgb") {
        return RK_FMT_RGB888;
    }
    throw std::invalid_argument("model_format must be 'rgb' or 'bgr'.");
}

bool normalize_roi(const std::tuple<int, int, int, int>& roi,
                   int image_width,
                   int image_height,
                   int& roi_x,
                   int& roi_y,
                   int& roi_w,
                   int& roi_h) {
    roi_x = std::get<0>(roi);
    roi_y = std::get<1>(roi);
    roi_w = std::get<2>(roi);
    roi_h = std::get<3>(roi);

    if (roi_w <= 0 || roi_h <= 0) {
        roi_x = 0;
        roi_y = 0;
        roi_w = 0;
        roi_h = 0;
        return false;
    }

    roi_x = std::max(0, std::min(roi_x, image_width - 1));
    roi_y = std::max(0, std::min(roi_y, image_height - 1));
    roi_w = std::max(1, std::min(roi_w, image_width - roi_x));
    roi_h = std::max(1, std::min(roi_h, image_height - roi_y));
    return true;
}

int clamp_to_image_edge(int value, int max_edge) {
    return std::max(0, std::min(value, max_edge));
}

std::string detection_label(ModelType model_type, int class_id, const YoloV5PostProcessCtx* yolov5_ctx,
                            const Yolo11PostProcessCtx* yolo11_ctx,
                            const Yolo11SegPostProcessCtx* yolo11_seg_ctx,
                            const Yolo11PosePostProcessCtx* yolo11_pose_ctx) {
    switch (model_type) {
        case ModelType::YOLO11:
            return coco_cls_to_name_yolo11(yolo11_ctx, class_id);
        case ModelType::YOLO11_SEG:
            return coco_cls_to_name_yolo11_seg(yolo11_seg_ctx, class_id);
        case ModelType::YOLO11_POSE:
            return coco_cls_to_name_yolo11_pose(yolo11_pose_ctx, class_id);
        case ModelType::YOLOV5:
            return coco_cls_to_name(yolov5_ctx, class_id);
        case ModelType::RETINAFACE:
            return "face";
        default:
            return "unknown";
    }
}

int init_model_dispatch(ModelType model_type, const std::string& model_path, rknn_app_context_t* app_ctx) {
    switch (model_type) {
        case ModelType::YOLOV5:
            return init_yolov5_model(model_path.c_str(), app_ctx);
        case ModelType::RETINAFACE:
            return init_retinaface_model(model_path.c_str(), app_ctx);
        case ModelType::FACENET:
            return init_facenet_model(model_path.c_str(), app_ctx);
        case ModelType::YOLO11:
            return init_yolo11_model(model_path.c_str(), app_ctx);
        case ModelType::YOLO11_SEG:
            return init_yolo11_seg_model(model_path.c_str(), app_ctx);
        case ModelType::YOLO11_POSE:
            return init_yolo11_pose_model(model_path.c_str(), app_ctx);
        case ModelType::LPRNET:
            return init_lprnet_model(model_path.c_str(), app_ctx);
        default:
            throw std::invalid_argument("Unknown model type.");
    }
}

int init_postprocess_dispatch(ModelType model_type,
                              const std::string& label_path,
                              float box_thresh,
                              float nms_thresh,
                              rknn_app_context_t* app_ctx,
                              YoloV5PostProcessCtx** yolov5_ctx,
                              Yolo11PostProcessCtx** yolo11_ctx,
                              Yolo11SegPostProcessCtx** yolo11_seg_ctx,
                              Yolo11PosePostProcessCtx** yolo11_pose_ctx) {
    if (yolov5_ctx) {
        *yolov5_ctx = nullptr;
    }
    if (yolo11_ctx) {
        *yolo11_ctx = nullptr;
    }
    if (yolo11_seg_ctx) {
        *yolo11_seg_ctx = nullptr;
    }
    if (yolo11_pose_ctx) {
        *yolo11_pose_ctx = nullptr;
    }

    const char* label_cstr = label_path.empty() ? nullptr : label_path.c_str();
    switch (model_type) {
        case ModelType::YOLOV5: {
            const int num_classes = get_yolov5_model_num_classes(app_ctx);
            YoloV5PostProcessCtx* ctx =
                create_yolov5_post_process_ctx(label_cstr, box_thresh, nms_thresh, num_classes);
            if (!ctx) {
                return -1;
            }
            if (yolov5_ctx) {
                *yolov5_ctx = ctx;
            }
            return 0;
        }
        case ModelType::YOLO11: {
            const int num_classes = get_yolo11_model_num_classes(app_ctx);
            Yolo11PostProcessCtx* ctx =
                create_yolo11_post_process_ctx(label_cstr, box_thresh, nms_thresh, num_classes);
            if (!ctx) {
                return -1;
            }
            if (yolo11_ctx) {
                *yolo11_ctx = ctx;
            }
            return 0;
        }
        case ModelType::YOLO11_SEG: {
            // YOLO11-seg can export either decoded boxes or raw DFL logits. Class count is ambiguous / YOLO11-seg 既可能导出解码后的框，也可能导出原始 DFL logits。类别数在这种情况下会变得不明确
            // from tensor shape alone in raw mode, so do not enforce model-side class count here.
            const int num_classes = -1;
            Yolo11SegPostProcessCtx* ctx =
                create_yolo11_seg_post_process_ctx(label_cstr, box_thresh, nms_thresh, num_classes);
            if (!ctx) {
                return -1;
            }
            if (yolo11_seg_ctx) {
                *yolo11_seg_ctx = ctx;
            }
            return 0;
        }
        case ModelType::YOLO11_POSE: {
            const int num_classes = get_yolo11_pose_model_num_classes(app_ctx);
            Yolo11PosePostProcessCtx* ctx =
                create_yolo11_pose_post_process_ctx(label_cstr, box_thresh, nms_thresh, num_classes);
            if (!ctx) {
                return -1;
            }
            if (yolo11_pose_ctx) {
                *yolo11_pose_ctx = ctx;
            }
            return 0;
        }
        default:
            return 0;
    }
}

int release_model_dispatch(ModelType model_type, rknn_app_context_t* app_ctx) {
    switch (model_type) {
        case ModelType::YOLOV5:
            return release_yolov5_model(app_ctx);
        case ModelType::RETINAFACE:
            return release_retinaface_model(app_ctx);
        case ModelType::FACENET:
            return release_facenet_model(app_ctx);
        case ModelType::YOLO11:
            return release_yolo11_model(app_ctx);
        case ModelType::YOLO11_SEG:
            return release_yolo11_seg_model(app_ctx);
        case ModelType::YOLO11_POSE:
            return release_yolo11_pose_model(app_ctx);
        case ModelType::LPRNET:
            return release_lprnet_model(app_ctx);
        default:
            return -1;
    }
}

void deinit_postprocess_dispatch(ModelType model_type, YoloV5PostProcessCtx*& yolov5_ctx,
                                 Yolo11PostProcessCtx*& yolo11_ctx,
                                 Yolo11SegPostProcessCtx*& yolo11_seg_ctx,
                                 Yolo11PosePostProcessCtx*& yolo11_pose_ctx) {
    switch (model_type) {
        case ModelType::YOLOV5:
            destroy_yolov5_post_process_ctx(yolov5_ctx);
            yolov5_ctx = nullptr;
            break;
        case ModelType::YOLO11:
            destroy_yolo11_post_process_ctx(yolo11_ctx);
            yolo11_ctx = nullptr;
            break;
        case ModelType::YOLO11_SEG:
            destroy_yolo11_seg_post_process_ctx(yolo11_seg_ctx);
            yolo11_seg_ctx = nullptr;
            break;
        case ModelType::YOLO11_POSE:
            destroy_yolo11_pose_post_process_ctx(yolo11_pose_ctx);
            yolo11_pose_ctx = nullptr;
            break;
        default:
            break;
    }
}

void run_detection_dispatch(ModelType model_type,
                            rknn_app_context_t* app_ctx,
                            const YoloV5PostProcessCtx* yolov5_ctx,
                            const Yolo11PostProcessCtx* yolo11_ctx,
                            const Yolo11SegPostProcessCtx* yolo11_seg_ctx,
                            const Yolo11PosePostProcessCtx* yolo11_pose_ctx,
                            float scale,
                            int pad_x,
                            int pad_y,
                            object_detect_result_list* od_results) {
    switch (model_type) {
        case ModelType::YOLOV5:
            if (inference_yolov5_model(app_ctx, yolov5_ctx, od_results) != 0) {
                throw std::runtime_error("YOLOv5 inference failed.");
            }
            return;
        case ModelType::RETINAFACE:
            if (inference_retinaface_model(app_ctx, od_results) != 0) {
                throw std::runtime_error("RetinaFace inference failed.");
            }
            return;
        case ModelType::YOLO11:
            if (inference_yolo11_model(app_ctx, yolo11_ctx, scale, pad_x, pad_y, od_results) != 0) {
                throw std::runtime_error("YOLOv11 inference failed.");
            }
            return;
        case ModelType::YOLO11_SEG:
            if (inference_yolo11_seg_model(app_ctx, yolo11_seg_ctx, scale, pad_x, pad_y, od_results) != 0) {
                throw std::runtime_error("YOLO11-seg inference failed.");
            }
            return;
        case ModelType::YOLO11_POSE:
            if (inference_yolo11_pose_model(app_ctx, yolo11_pose_ctx, od_results) != 0) {
                throw std::runtime_error("YOLO11-pose inference failed.");
            }
            return;
        default:
            throw std::runtime_error("NPU::inference supports only YOLOV5/YOLO11/YOLO11_SEG/YOLO11_POSE/RETINAFACE.");
    }
}

std::pair<int, int> resolve_npu_input_strides(const rknn_app_context_t& app_ctx, int model_w, int model_h) {
    int w_stride = app_ctx.input_attrs[0].w_stride;
    int h_stride = app_ctx.input_attrs[0].h_stride;
    if (w_stride <= 0) {
        w_stride = model_w;
    }
    if (h_stride <= 0) {
        h_stride = model_h;
    }
    return {w_stride, h_stride};
}

}  // namespace

NPU::NPU(ModelType model_type, const std::string& model_path, const std::string& label_path, float box_thresh, float nms_thresh)
    : m_app_ctx(std::make_unique<rknn_app_context_t>()),
      m_model_type(model_type),
      m_model_path(model_path),
      m_label_path(label_path),
      m_box_thresh(box_thresh),
      m_nms_thresh(nms_thresh),
      m_initialized(false) {
    memset(m_app_ctx.get(), 0, sizeof(rknn_app_context_t));
    const int ret = initialize_runtime(true);
    if (ret != 0) {
        throw std::runtime_error("Failed to initialize NPU model. ret=" + std::to_string(ret));
    }
}

NPU::~NPU() {
    std::lock_guard<std::mutex> lock(m_runtime_mutex);
    release_runtime();
}

int NPU::initialize_runtime(bool print_log) {
    const int init_ret = init_model_dispatch(m_model_type, m_model_path, m_app_ctx.get());
    int ret = init_ret;
    if (ret == 0) {
        ret = init_postprocess_dispatch(m_model_type, m_label_path, m_box_thresh, m_nms_thresh, m_app_ctx.get(),
                                        &m_yolov5_post_ctx, &m_yolo11_post_ctx, &m_yolo11_seg_post_ctx,
                                        &m_yolo11_pose_post_ctx);
    }
    if (ret == 0) {
        m_initialized = true;
        if (print_log) {
            VISIONG_LOG_INFO("NPU", "Initialized. Model: " << m_model_path << " ["
                                                            << m_app_ctx->model_width << "x"
                                                            << m_app_ctx->model_height << "]");
        }
        return 0;
    }

    if (init_ret == 0) {
        release_model_dispatch(m_model_type, m_app_ctx.get());
        deinit_postprocess_dispatch(m_model_type, m_yolov5_post_ctx, m_yolo11_post_ctx, m_yolo11_seg_post_ctx,
                                    m_yolo11_pose_post_ctx);
    }
    m_initialized = false;
    return ret;
}

void NPU::release_runtime() {
    if (!m_initialized) {
        deinit_postprocess_dispatch(m_model_type, m_yolov5_post_ctx, m_yolo11_post_ctx, m_yolo11_seg_post_ctx,
                                m_yolo11_pose_post_ctx);
        return;
    }
    release_model_dispatch(m_model_type, m_app_ctx.get());
    deinit_postprocess_dispatch(m_model_type, m_yolov5_post_ctx, m_yolo11_post_ctx, m_yolo11_seg_post_ctx,
                                m_yolo11_pose_post_ctx);
    m_initialized = false;
}

bool NPU::try_recover_runtime(const char* reason) {
    if (reason != nullptr && reason[0] != '\0') {
        VISIONG_LOG_WARN("NPU", "Runtime failure (" << reason << "), trying one-time runtime reinit...");
    } else {
        VISIONG_LOG_WARN("NPU", "Runtime failure, trying one-time runtime reinit...");
    }

    release_runtime();
    std::memset(m_app_ctx.get(), 0, sizeof(rknn_app_context_t));

    const int ret = initialize_runtime(false);
    if (ret != 0) {
        VISIONG_LOG_ERROR("NPU", "Runtime reinit failed, ret=" << ret);
        return false;
    }
    VISIONG_LOG_INFO("NPU", "Runtime reinit succeeded.");
    return true;
}

std::vector<Detection> NPU::inference(const ImageBuffer& img_buf, const std::tuple<int, int, int, int>& roi, const std::string& model_format_str) {
    std::lock_guard<std::mutex> lock(m_runtime_mutex);
    if (!m_initialized || !img_buf.is_valid()) throw std::runtime_error("NPU invalid state or input.");

    // 1) Validate input format and infer model input format. / 1）校验输入格式，并推断模型输入格式。
    PIXEL_FORMAT_E target_format = parse_model_input_format(model_format_str);
    
    if (visiong::is_gray8_format(img_buf.format)) {
        throw std::invalid_argument("NPU does not support Grayscale input directly. Please provide color image or convert first.");
    }

    // 2) Prepare source DMA (zero-copy when possible). / 2）准备源 DMA（尽可能走零拷贝）。
    SourceDmaContext src_dma_ctx = prepare_source_dma(img_buf);
    const RgaDmaBuffer* current_dma_buf = src_dma_ctx.current;

    int current_width = img_buf.width;
    int current_height = img_buf.height;

    // 3) ROI handling. Crop via reusable DMA cache when ROI is provided. / 3）处理 ROI。若提供 ROI，则通过可复用 DMA 缓存完成裁切。
    int roi_x = 0;
    int roi_y = 0;
    int roi_w = 0;
    int roi_h = 0;
    const bool has_roi = normalize_roi(roi, img_buf.width, img_buf.height, roi_x, roi_y, roi_w, roi_h);
    bool roi_applied = false;

    if (has_roi && ensure_dma_cache(&m_cached_infer_roi_dma, roi_w, roi_h, static_cast<int>(img_buf.format))) {
        const im_rect crop_rect = {roi_x, roi_y, roi_w, roi_h};
        visiong::bufstate::prepare_device_read(*current_dma_buf, visiong::bufstate::BufferOwner::RGA);
        visiong::bufstate::prepare_device_write(*m_cached_infer_roi_dma,
                                                visiong::bufstate::BufferOwner::RGA,
                                                visiong::bufstate::AccessIntent::Overwrite);
        if (imcrop(current_dma_buf->get_buffer(), m_cached_infer_roi_dma->get_buffer(), crop_rect) == IM_STATUS_SUCCESS) {
            visiong::bufstate::mark_device_write(*m_cached_infer_roi_dma, visiong::bufstate::BufferOwner::RGA);
            current_dma_buf = m_cached_infer_roi_dma.get();
            current_width = roi_w;
            current_height = roi_h;
            roi_applied = true;
        }
    }
    const int roi_offset_x = roi_applied ? roi_x : 0;
    const int roi_offset_y = roi_applied ? roi_y : 0;

    // 4) Compute letterbox parameters. / 4）计算 letterbox 参数。
    const int model_w = m_app_ctx->model_width;
    const int model_h = m_app_ctx->model_height;
    float scale = std::min((float)model_w / current_width, (float)model_h / current_height);
    int scaled_w = static_cast<int>(current_width * scale) & ~1;
    int scaled_h = static_cast<int>(current_height * scale) & ~1;
    int pad_x = (model_w - scaled_w) / 2 & ~1;
    int pad_y = (model_h - scaled_h) / 2 & ~1;

    // 5) Write letterbox output directly into NPU input memory. / 5）将 letterbox 输出直接写入 NPU 输入内存。
    // Pull required strides from model input attrs.
    const auto [npu_w_stride, npu_h_stride] = resolve_npu_input_strides(*m_app_ctx, model_w, model_h);

    // Construct a non-owning wrapper over NPU input memory. / 在 NPU 输入内存之上构造一个非拥有型包装。
    // Pass NPU strides explicitly so RGA writes into the correct layout.
    RgaDmaBuffer npu_input_wrapper(
        m_app_ctx->input_mems[0]->fd, 
        m_app_ctx->input_mems[0]->virt_addr, 
        m_app_ctx->input_mems[0]->size,
        model_w, model_h, 
        static_cast<int>(target_format),
        npu_w_stride, npu_h_stride 
    );

    // 5b) Tightened RGA path: try direct YUV->RGB/BGR + resize + letterbox. / 5b）收紧后的 RGA 路径：尝试直接做 YUV->RGB/BGR + resize + letterbox。
    const bool src_is_yuv = (current_dma_buf->get_mpi_format() == static_cast<int>(RK_FMT_YUV420SP) ||
                             current_dma_buf->get_mpi_format() == static_cast<int>(RK_FMT_YUV420SP_VU));
    const bool model_expects_rgb_bgr = (target_format == RK_FMT_RGB888 || target_format == RK_FMT_BGR888);
    const bool src_uploaded_from_cpu = (src_dma_ctx.uploaded_dma != nullptr);

    bool letterbox_done = false;

    if (src_is_yuv && model_expects_rgb_bgr && m_cached_direct_yuv_to_rgb_letterbox != 0) {
        try {
            rga_letterbox_op(*current_dma_buf,
                             npu_input_wrapper,
                             std::make_tuple(128, 128, 128),
                             src_uploaded_from_cpu,
                             false);
            letterbox_done = true;
            m_cached_direct_yuv_to_rgb_letterbox = 1;
        } catch (const std::exception&) {
            if (m_cached_direct_yuv_to_rgb_letterbox < 0) {
                VISIONG_LOG_WARN("NPU",
                                 "Direct YUV->RGB/BGR letterbox unsupported, fallback to imcvtcolor+letterbox.");
            }
            m_cached_direct_yuv_to_rgb_letterbox = 0;
        }
    }

    if (!letterbox_done) {
        const bool need_cvtcolor = src_is_yuv && model_expects_rgb_bgr;
        const RgaDmaBuffer* letterbox_src = current_dma_buf;
        bool letterbox_src_cpu_dirty = src_uploaded_from_cpu;

        if (need_cvtcolor) {
            if (!ensure_dma_cache(&m_cached_infer_cvt_dma,
                                  current_width,
                                  current_height,
                                  static_cast<int>(target_format))) {
                throw std::runtime_error("NPU failed to allocate conversion DMA cache.");
            }
            const int src_rga_fmt = (current_dma_buf->get_mpi_format() == static_cast<int>(RK_FMT_YUV420SP_VU))
                                        ? RK_FORMAT_YCrCb_420_SP
                                        : RK_FORMAT_YCbCr_420_SP;
            const int dst_rga_fmt = (target_format == RK_FMT_RGB888) ? RK_FORMAT_RGB_888 : RK_FORMAT_BGR_888;
            visiong::bufstate::prepare_device_read(*current_dma_buf, visiong::bufstate::BufferOwner::RGA);
            visiong::bufstate::prepare_device_write(*m_cached_infer_cvt_dma,
                                                    visiong::bufstate::BufferOwner::RGA,
                                                    visiong::bufstate::AccessIntent::Overwrite);
            if (imcvtcolor(current_dma_buf->get_buffer(),
                           m_cached_infer_cvt_dma->get_buffer(),
                           src_rga_fmt,
                           dst_rga_fmt) != IM_STATUS_SUCCESS) {
                throw std::runtime_error("NPU RGA imcvtcolor (YUV->RGB) failed.");
            }
            visiong::bufstate::mark_device_write(*m_cached_infer_cvt_dma, visiong::bufstate::BufferOwner::RGA);
            letterbox_src = m_cached_infer_cvt_dma.get();
            letterbox_src_cpu_dirty = false;
        }

        rga_letterbox_op(*letterbox_src,
                         npu_input_wrapper,
                         std::make_tuple(128, 128, 128),
                         letterbox_src_cpu_dirty,
                         false);
    }

    // Hand input ownership back to NPU after RGA writes. / 在 RGA 写入完成后，把输入所有权交还给 NPU。
    visiong::bufstate::prepare_device_read(npu_input_wrapper, visiong::bufstate::BufferOwner::NPU);
    
    // 6) Run inference. / 6）执行推理。
    object_detect_result_list od_results;
    memset(&od_results, 0, sizeof(od_results));

    try {
        run_detection_dispatch(m_model_type, m_app_ctx.get(), m_yolov5_post_ctx, m_yolo11_post_ctx, m_yolo11_seg_post_ctx,
                               m_yolo11_pose_post_ctx, scale, pad_x, pad_y, &od_results);
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        if (msg.find("inference failed") != std::string::npos && try_recover_runtime(msg.c_str())) {
            // Current frame used old tensor buffers; after reinit they are invalid. / 当前帧使用的是旧 tensor 缓冲区；重新初始化后它们已失效。
            // Drop this frame and let next frame run on fresh runtime context.
            return {};
        }
        throw;
    }

    // 7) Pack model outputs into API-level detection objects. / 7）把模型输出封装成 API 层检测对象。
    std::vector<Detection> detections;
    detections.reserve(od_results.count);
    const bool has_native_xyxy = (m_model_type == ModelType::YOLO11 || m_model_type == ModelType::YOLO11_SEG);
    const bool has_pose_keypoints = (m_model_type == ModelType::YOLO11_POSE);
    const int max_x = std::max(0, static_cast<int>(img_buf.width) - 1);
    const int max_y = std::max(0, static_cast<int>(img_buf.height) - 1);
    const float inv_scale = (scale > 1e-6f) ? (1.0f / scale) : 1.0f;

    for (int i = 0; i < od_results.count; ++i) {
        object_detect_result* det = &od_results.results[i];
        int x1, y1, x2, y2;

        if (has_native_xyxy) {
            x1 = det->box.left + roi_offset_x;
            y1 = det->box.top + roi_offset_y;
            x2 = det->box.right + roi_offset_x;
            y2 = det->box.bottom + roi_offset_y;
        } else {
            x1 = static_cast<int>((det->box.left - pad_x) * inv_scale) + roi_offset_x;
            y1 = static_cast<int>((det->box.top - pad_y) * inv_scale) + roi_offset_y;
            x2 = static_cast<int>((det->box.right - pad_x) * inv_scale) + roi_offset_x;
            y2 = static_cast<int>((det->box.bottom - pad_y) * inv_scale) + roi_offset_y;
        }

        x1 = clamp_to_image_edge(x1, max_x);
        y1 = clamp_to_image_edge(y1, max_y);
        x2 = clamp_to_image_edge(x2, max_x);
        y2 = clamp_to_image_edge(y2, max_y);

        if (x2 > x1 && y2 > y1) {
            Detection d;
            d.box = std::make_tuple(x1, y1, x2 - x1, y2 - y1);
            d.score = std::max(0.f, std::min(1.f, det->prop));
            d.class_id = det->cls_id;
            d.label = detection_label(m_model_type, det->cls_id, m_yolov5_post_ctx, m_yolo11_post_ctx,
                                      m_yolo11_seg_post_ctx, m_yolo11_pose_post_ctx);

            if (m_model_type == ModelType::RETINAFACE) {
                d.landmarks.reserve(5);
                for (int j = 0; j < 5; ++j) {
                    float lx = (det->point[j].x - pad_x) * inv_scale + roi_offset_x;
                    float ly = (det->point[j].y - pad_y) * inv_scale + roi_offset_y;
                    lx = std::max(0.f, std::min(lx, static_cast<float>(max_x)));
                    ly = std::max(0.f, std::min(ly, static_cast<float>(max_y)));
                    d.landmarks.emplace_back(lx, ly);
                }
            }
            if (m_model_type == ModelType::YOLO11_SEG && det->mask_point_count > 0) {
                const int limit = std::max(0, std::min(det->mask_point_count, SEG_MASK_POINT_MAX_SIZE));
                d.mask_points.reserve(limit);
                for (int j = 0; j < limit; ++j) {
                    float mx = det->mask_points[j][0] + roi_offset_x;
                    float my = det->mask_points[j][1] + roi_offset_y;
                    mx = std::max(0.f, std::min(mx, static_cast<float>(max_x)));
                    my = std::max(0.f, std::min(my, static_cast<float>(max_y)));
                    d.mask_points.emplace_back(mx, my);
                }
            }

            if (has_pose_keypoints && det->keypoint_count > 0) {
                const int limit = std::max(0, std::min(det->keypoint_count, POSE_KEYPOINT_MAX_SIZE));
                d.keypoints.reserve(limit);
                for (int j = 0; j < limit; ++j) {
                    float kx = (det->keypoints[j][0] - pad_x) * inv_scale + roi_offset_x;
                    float ky = (det->keypoints[j][1] - pad_y) * inv_scale + roi_offset_y;
                    float ks = det->keypoints[j][2];
                    kx = std::max(0.f, std::min(kx, static_cast<float>(max_x)));
                    ky = std::max(0.f, std::min(ky, static_cast<float>(max_y)));
                    ks = std::max(0.f, std::min(ks, 1.f));
                    d.keypoints.emplace_back(kx, ky, ks);
                }
            }

            detections.push_back(d);
        }
    }
    return detections;
}

// --- FaceNet zero-copy optimized path --- / --- FaceNet 零拷贝优化路径 ---
std::vector<float> NPU::get_face_feature(const ImageBuffer& face_image) {
    std::lock_guard<std::mutex> lock(m_runtime_mutex);
    if (!m_initialized || m_model_type != ModelType::FACENET) throw std::runtime_error("Invalid model for feature extraction.");
    if (!face_image.is_valid()) throw std::invalid_argument("Invalid input image.");

    const int model_w = m_app_ctx->model_width;
    const int model_h = m_app_ctx->model_height;

    // 1) Prepare source DMA (zero-copy when possible). / 1) 准备 源 DMA (零拷贝 当 possible).
    SourceDmaContext src_dma_ctx = prepare_source_dma(face_image);
    const RgaDmaBuffer* current_dma_buf = src_dma_ctx.current;

    // Fetch NPU input stride. / 获取 NPU 输入 stride。
    const auto [npu_w_stride, npu_h_stride] = resolve_npu_input_strides(*m_app_ctx, model_w, model_h);

    // 2) Prepare NPU input wrapper. / 2）准备 NPU 输入包装。
    RgaDmaBuffer npu_input_wrapper(
        m_app_ctx->input_mems[0]->fd, 
        m_app_ctx->input_mems[0]->virt_addr, 
        m_app_ctx->input_mems[0]->size,
        model_w, model_h, 
        RK_FMT_RGB888,
        npu_w_stride, npu_h_stride
    );

    // 3) RGA letterbox. Skip destination readback because NPU consumes it directly. / 3）RGA letterbox。因为 NPU 会直接消费结果，所以跳过目标读回。
    rga_letterbox_op(*current_dma_buf,
                     npu_input_wrapper,
                     {128, 128, 128},
                     src_dma_ctx.uploaded_dma != nullptr,
                     false);
    visiong::bufstate::prepare_device_read(npu_input_wrapper, visiong::bufstate::BufferOwner::NPU);

    // 4) Run inference. / 4）执行推理。
    if (visiong::npu::rknn::run_and_sync_outputs(m_app_ctx.get(), "FaceNet") != 0) {
        throw std::runtime_error("FaceNet inference execution failed.");
    }
    
    std::vector<float> feature_vector;
    output_normalization(m_app_ctx.get(), (uint8_t*)m_app_ctx->output_mems[0]->virt_addr, feature_vector);
    return feature_vector;
}

std::string NPU::recognize_plate(const ImageBuffer& plate_image) {
    std::lock_guard<std::mutex> lock(m_runtime_mutex);
    // 1. Basic state checks. / 1）基础状态检查。
    if (!m_initialized || m_model_type != ModelType::LPRNET) {
        throw std::runtime_error("recognize_plate can only be called on an initialized LPRNET model.");
    }
    if (!plate_image.is_valid()) {
        throw std::invalid_argument("Input plate_image is not valid for recognition.");
    }

    // 2) Ensure BGR888 input as required by LPRNet. / 2）确保输入符合 LPRNet 所需的 BGR888。
    // Avoid extra copies: only convert when input isn't already BGR888.
    ImageBuffer temp_bgr_buffer;
    const ImageBuffer* processing_buf = &plate_image;

    if (plate_image.format != RK_FMT_BGR888) {
        try {
            // to_format internally selects CPU (NEON) or RGA path. / to_format 内部会自动选择 CPU（NEON）或 RGA 路径。
            temp_bgr_buffer = plate_image.to_format(RK_FMT_BGR888);
            processing_buf = &temp_bgr_buffer;
        } catch (const std::exception& e) {
            throw std::runtime_error("LPRNet pre-processing failed during color conversion to BGR: " + std::string(e.what()));
        }
    }

    // 3) Resize to LPRNet input shape (94x24). / 3）缩放到 LPRNet 输入尺寸（94x24）。
    // resize() auto-selects CPU for small sizes to avoid RGA restrictions.
    ImageBuffer resized_buffer;
    if (processing_buf->width != 94 || processing_buf->height != 24) {
        try {
            resized_buffer = processing_buf->resize(94, 24);
            processing_buf = &resized_buffer;
        } catch (const std::exception& e) {
             throw std::runtime_error("LPRNet pre-processing failed during resize: " + std::string(e.what()));
        }
    }

    // 4) Copy resized input into NPU buffer with stride handling. / 4）在处理 stride 的同时，把缩放后的输入拷贝到 NPU 缓冲区。
    int bpp = get_bpp_for_format(processing_buf->format);
    
    // Use model stride from input attrs so aligned memory layout is respected. / 使用输入属性中的模型 stride，保证对齐内存布局被正确遵守。
    const int npu_w_stride =
        resolve_npu_input_strides(*m_app_ctx, m_app_ctx->model_width, m_app_ctx->model_height).first;

    visiong::bufstate::prepare_cpu_read(*processing_buf);
    copy_data_with_stride(
        m_app_ctx->input_mems[0]->virt_addr,
        npu_w_stride * bpp / 8,
        processing_buf->get_data(),
        processing_buf->w_stride * bpp / 8,
        processing_buf->height,
        processing_buf->width * bpp / 8
    );

    // 5) Flush input cache before inference. / 5）在推理前刷新输入缓存。
    const auto npu_input_view = visiong::bufstate::make_dma_view(
        m_app_ctx->input_mems[0]->fd,
        m_app_ctx->input_mems[0]->virt_addr,
        m_app_ctx->input_mems[0]->size);
    visiong::bufstate::mark_cpu_write(npu_input_view);
    visiong::bufstate::prepare_device_read(npu_input_view, visiong::bufstate::BufferOwner::NPU);

    // 6) Inference and decoding. / 6）执行推理与解码。
    lprnet_result result;
    if (inference_lprnet_model(m_app_ctx.get(), &result) != 0) {
        throw std::runtime_error("LPRNet model inference failed.");
    }

    return result.plate_name;
}

float NPU::get_feature_distance(const std::vector<float>& f1, const std::vector<float>& f2) {
    if (f1.size() != 128 || f2.size() != 128) return 100.0f;
    float sum = 0;
    for(int i=0; i<128; ++i) { float d = f1[i] - f2[i]; sum += d*d; }
    return std::sqrt(sum);
}

bool NPU::is_initialized() const {
    std::lock_guard<std::mutex> lock(m_runtime_mutex);
    return m_initialized;
}

int NPU::get_model_width() const {
    std::lock_guard<std::mutex> lock(m_runtime_mutex);
    return m_initialized ? m_app_ctx->model_width : 0;
}

int NPU::get_model_height() const {
    std::lock_guard<std::mutex> lock(m_runtime_mutex);
    return m_initialized ? m_app_ctx->model_height : 0;
}

