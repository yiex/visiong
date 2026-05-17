// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_COMMON_VISION_TYPES_H
#define VISIONG_COMMON_VISION_TYPES_H

#include <string>
#include <tuple>
#include <vector>

struct Blob {
    int x;
    int y;
    int w;
    int h;
    int cx;
    int cy;
    int pixels;
    unsigned int code;

    Blob(int x_value,
         int y_value,
         int width_value,
         int height_value,
         int center_x,
         int center_y,
         int pixel_count,
         unsigned int code_value = 0)
        : x(x_value),
          y(y_value),
          w(width_value),
          h(height_value),
          cx(center_x),
          cy(center_y),
          pixels(pixel_count),
          code(code_value) {}

    std::tuple<int, int, int, int> rect() const { return std::make_tuple(x, y, w, h); }
    int area() const { return w * h; }
};

struct Line {
    int x1;
    int y1;
    int x2;
    int y2;
    float rho;
    float theta;
    int magnitude;

    Line(int x1_value = 0, int y1_value = 0, int x2_value = 0, int y2_value = 0, int magnitude_value = 0)
        : x1(x1_value),
          y1(y1_value),
          x2(x2_value),
          y2(y2_value),
          rho(0.0f),
          theta(0.0f),
          magnitude(magnitude_value) {}

    Line(float rho_value, float theta_value, int magnitude_value = 0)
        : x1(0),
          y1(0),
          x2(0),
          y2(0),
          rho(rho_value),
          theta(theta_value),
          magnitude(magnitude_value) {}
};

struct Circle {
    int cx;
    int cy;
    int r;
    int magnitude;

    Circle(int center_x = 0, int center_y = 0, int radius = 0, int magnitude_value = 0)
        : cx(center_x), cy(center_y), r(radius), magnitude(magnitude_value) {}
};

struct Rect {
    int x;
    int y;
    int w;
    int h;

    Rect(int x_value = 0, int y_value = 0, int width_value = 0, int height_value = 0)
        : x(x_value), y(y_value), w(width_value), h(height_value) {}

    std::tuple<int, int, int, int> to_tuple() const { return std::make_tuple(x, y, w, h); }
};

using Polygon = std::vector<std::tuple<int, int>>;

struct QRCode {
    Polygon corners;
    std::string payload;

    QRCode(const Polygon& corner_points, const std::string& payload_text)
        : corners(corner_points), payload(payload_text) {}
};

struct AprilTagDetection {
    std::string family;
    int id;
    int hamming;
    float decision_margin;
    double cx;
    double cy;
    Polygon corners;
    std::vector<double> homography;
    bool has_pose;
    double pose_error;
    std::vector<double> pose_R;
    std::vector<double> pose_t;
    double tx;
    double ty;
    double tz;
    double distance;
    double roll;
    double pitch;
    double yaw;

    AprilTagDetection(const std::string& family_name,
                      int tag_id,
                      int hamming_value,
                      float margin,
                      double center_x,
                      double center_y,
                      const Polygon& corner_points,
                      const std::vector<double>& homography_values,
                      bool has_pose_value,
                      double pose_error_value,
                      const std::vector<double>& pose_R_values,
                      const std::vector<double>& pose_t_values,
                      double tx_value,
                      double ty_value,
                      double tz_value,
                      double distance_value,
                      double roll_value,
                      double pitch_value,
                      double yaw_value)
        : family(family_name),
          id(tag_id),
          hamming(hamming_value),
          decision_margin(margin),
          cx(center_x),
          cy(center_y),
          corners(corner_points),
          homography(homography_values),
          has_pose(has_pose_value),
          pose_error(pose_error_value),
          pose_R(pose_R_values),
          pose_t(pose_t_values),
          tx(tx_value),
          ty(ty_value),
          tz(tz_value),
          distance(distance_value),
          roll(roll_value),
          pitch(pitch_value),
          yaw(yaw_value) {}
};

#endif  // VISIONG_COMMON_VISION_TYPES_H
