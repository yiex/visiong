// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_NPU_INTERNAL_TRACKING_CORE_H
#define VISIONG_NPU_INTERNAL_TRACKING_CORE_H

#include "engine.h"

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

struct Config {
    std::string windowing = "cosine";
    std::vector<float> window;

    int stride = 16;
    float penalty_k = 0.138f;
    float window_influence = 0.455f;
    float lr = 0.348f;
    int exemplar_size = 127;
    int instance_size = 255;
    int total_stride = 16;
    int score_size = 15;
    float context_amount = 0.5f;
};

struct State {
    int im_h = 0;
    int im_w = 0;
    cv::Scalar channel_ave;
    cv::Point2f target_pos = {0.f, 0.f};
    cv::Point2f target_sz = {0.f, 0.f};
    float cls_score_max = 0.0f;
    cv::Rect bbox;
};

class NanoTrackCore {
public:
    NanoTrackCore();
    ~NanoTrackCore();

    void init(const cv::Mat& img, cv::Rect bbox);
    float track(const cv::Mat& img);
    void update(const cv::Mat& x_crops,
                cv::Point2f& target_pos,
                cv::Point2f& target_sz,
                float scale_z,
                float& cls_score_max);
    nn_error_e LoadModel(const char* modelTName, const char* modelXName, const char* modelHName);

    State state;
    Config cfg;

private:
    void create_grids();
    void create_window();
    cv::Mat get_subwindow_tracking(const cv::Mat& im,
                                   cv::Point2f pos,
                                   int model_sz,
                                   int original_sz,
                                   cv::Scalar channel_ave);

    std::vector<float> grid_to_search_x;
    std::vector<float> grid_to_search_y;
    std::vector<float> window;

    // T: template backbone, X: search backbone, H: head network. / 详见英文原注释。
    std::shared_ptr<NNEngine> t_engine_;
    std::shared_ptr<NNEngine> x_engine_;
    std::shared_ptr<NNEngine> h_engine_;

    tensor_data_s t_input_tensor_;
    tensor_data_s x_input_tensor_;
    tensor_data_s h_input_tensor_1;
    tensor_data_s h_input_tensor_2;

    std::vector<tensor_data_s> t_output_tensors_;
    std::vector<tensor_data_s> x_output_tensors_;
    std::vector<tensor_data_s> h_output_tensors_;

    bool t_want_float_ = false;
    bool x_want_float_ = false;
    bool h_want_float_ = false;

    std::vector<int32_t> t_out_zps_;
    std::vector<int32_t> x_out_zps_;
    std::vector<int32_t> h_out_zps_;
    std::vector<float> t_out_scales_;
    std::vector<float> x_out_scales_;
    std::vector<float> h_out_scales_;
};

#endif  // VISIONG_NPU_INTERNAL_TRACKING_CORE_H
