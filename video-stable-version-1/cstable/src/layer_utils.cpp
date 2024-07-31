#include "../includes/layer_utils.h"
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include "../includes/frame.h"

// Helper function to put an image over the top of another
cv::Mat layer_overlay(const cv::Mat& foreground, const cv::Mat& background) {
    cv::Mat overlaid = foreground.clone();
    cv::Mat negative_space;
    cv::inRange(foreground, cv::Scalar(0, 0, 0, 0), cv::Scalar(255, 255, 255, 0), negative_space);

    background.copyTo(overlaid, negative_space);

    // Set the alpha channel to 255
    for (int y = 0; y < overlaid.rows; ++y) {
        for (int x = 0; x < overlaid.cols; ++x) {
            overlaid.at<cv::Vec4b>(y, x)[3] = 255;
        }
    }

    return overlaid;
}

// Helper function to blend a foreground image over background
cv::Mat layer_blend(const cv::Mat& foreground, const cv::Mat& background, double foreground_alpha = 0.6) {
    cv::Mat blended;
    cv::addWeighted(foreground, foreground_alpha, background, 1 - foreground_alpha, 0, blended);
    return blended;
}

// Helper method to apply layering function in vidstab process
Frame apply_layer_func(const Frame& cur_frame, const Frame& prev_frame, cv::Mat (*layer_func)(const cv::Mat&, const cv::Mat&)) {
    if (!prev_frame.empty()) {
        cv::Mat layered_image = layer_func(cur_frame.get_image(), prev_frame.get_image());
        return Frame(layered_image, cur_frame.get_color_format());
    }
    return cur_frame;
}

