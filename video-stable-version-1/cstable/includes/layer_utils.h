#ifndef LAYER_UTILS_H
#define LAYER_UTILS_H

namespace cv {class Mat;}


cv::Mat layer_overlay(
    const cv::Mat& foreground,
    const cv::Mat& background
);

cv::Mat layer_blend(
    const cv::Mat& foreground, 
    const cv::Mat& background, 
    double foreground_alpha = 0.6
);

Frame apply_layer_func(const Frame& cur_frame, const Frame& prev_frame, cv::Mat (*layer_func)(const cv::Mat&, const cv::Mat&));
#endif //LAYER_UTILS_H