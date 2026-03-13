// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_NPU_NPU_H
#define VISIONG_NPU_NPU_H

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <tuple>
struct rknn_app_context_t;

class ImageBuffer;
class RgaDmaBuffer;
struct YoloV5PostProcessCtx;
struct Yolo11PostProcessCtx;
struct Yolo11SegPostProcessCtx;
struct Yolo11PosePostProcessCtx;

enum class ModelType {
    YOLOV5,
    RETINAFACE,
    FACENET,
    YOLO11,
    YOLO11_SEG,
    YOLO11_POSE,
    LPRNET
};

struct Detection {
    std::tuple<int, int, int, int> box;
    float score;  // confidence in [0, 1], normalized from model output
    int class_id;
    std::string label;
    std::vector<std::tuple<float, float>> landmarks;  // RetinaFace only: 5 points (left_eye, right_eye, nose, left_mouth, right_mouth); empty for others
    std::vector<std::tuple<float, float, float>> keypoints; // YOLO11-pose only: (x, y, score)
    std::vector<std::tuple<float, float>> mask_points; // YOLO11-seg only: contour points
};

class NPU {
public:
    NPU(ModelType model_type,
        const std::string& model_path,
        const std::string& label_path = "",
        float box_thresh = 0.25f,
        float nms_thresh = 0.45f);

    ~NPU();
    NPU(const NPU&) = delete;
    NPU& operator=(const NPU&) = delete;
    NPU(NPU&&) = delete;
    NPU& operator=(NPU&&) = delete;

    std::vector<Detection> inference(const ImageBuffer& img_buf,
                                     const std::tuple<int, int, int, int>& roi = {0, 0, 0, 0},
                                     const std::string& model_format = "rgb");
    std::vector<Detection> infer(const ImageBuffer& img_buf,
                                 const std::tuple<int, int, int, int>& roi = {0, 0, 0, 0},
                                 const std::string& model_format = "rgb") {
        return inference(img_buf, roi, model_format);
    }

    std::vector<float> get_face_feature(const ImageBuffer& face_image);

    std::string recognize_plate(const ImageBuffer& plate_image);

    static float get_feature_distance(const std::vector<float>& feature1, const std::vector<float>& feature2);
    bool is_initialized() const;
    int get_model_width() const;
    int get_model_height() const;
    int model_width() const { return get_model_width(); }
    int model_height() const { return get_model_height(); }

private:
    int initialize_runtime(bool print_log = true);
    void release_runtime();
    bool try_recover_runtime(const char* reason);

    std::unique_ptr<rknn_app_context_t> m_app_ctx;
    ModelType m_model_type;
    std::string m_model_path;
    std::string m_label_path;
    float m_box_thresh;
    float m_nms_thresh;
    bool m_initialized;
    YoloV5PostProcessCtx* m_yolov5_post_ctx = nullptr;
    Yolo11PostProcessCtx* m_yolo11_post_ctx = nullptr;
    Yolo11SegPostProcessCtx* m_yolo11_seg_post_ctx = nullptr;
    Yolo11PosePostProcessCtx* m_yolo11_pose_post_ctx = nullptr;
    std::unique_ptr<RgaDmaBuffer> m_cached_infer_roi_dma;
    std::unique_ptr<RgaDmaBuffer> m_cached_infer_cvt_dma;
    int m_cached_direct_yuv_to_rgb_letterbox = -1;
    mutable std::mutex m_runtime_mutex;
};

#endif  // VISIONG_NPU_NPU_H

