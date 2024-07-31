
#include "../includes/vidstab_utils.h"
#include "../includes/border_utils.h"
#include "../includes/layer_utils.h"
#include "../includes/frame.h"
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
// Helper function to convert transform list to transformation matrix
cv::Mat build_transformation_matrix(const cv::Mat& transform) {
    if (transform.rows != 1 || transform.cols != 3) {
        throw std::invalid_argument("Transform matrix must be a 1x3 matrix");
    }

    cv::Mat transform_matrix = cv::Mat::zeros(2, 3, CV_64F);
    transform_matrix.at<double>(0, 0) = std::cos(transform.at<double>(0, 2));
    transform_matrix.at<double>(0, 1) = -std::sin(transform.at<double>(0, 2));
    transform_matrix.at<double>(1, 0) = std::sin(transform.at<double>(0, 2));
    transform_matrix.at<double>(1, 1) = std::cos(transform.at<double>(0, 2));
    transform_matrix.at<double>(0, 2) = transform.at<double>(0, 0);
    transform_matrix.at<double>(1, 2) = transform.at<double>(0, 1);

    return transform_matrix;
}
// Helper function to apply border to a frame
std::pair<cv::Mat, int> border_frame(const Frame& frame, int border_size, const std::string& border_type) {
    std::unordered_map<std::string, int> border_modes = {
        {"black", cv::BORDER_CONSTANT},
        {"reflect", cv::BORDER_REFLECT},
        {"replicate", cv::BORDER_REPLICATE}
    };

    int border_mode = border_modes[border_type];

    cv::Mat bordered_frame_image;
    cv::copyMakeBorder(frame.get_image(), bordered_frame_image, border_size, border_size, border_size, border_size, border_mode, cv::Scalar(0, 0, 0));

    cv::Mat alpha_bordered_frame = frame.get_bgra_image();
    alpha_bordered_frame.setTo(cv::Scalar(0, 0, 0, 0));
    int h = frame.get_image().rows;
    int w = frame.get_image().cols;

    for (int y = border_size; y < border_size + h; ++y) {
        for (int x = border_size; x < border_size + w; ++x) {
            alpha_bordered_frame.at<cv::Vec4b>(y, x)[3] = 255;
        }
    }

    return {alpha_bordered_frame, border_mode};
}
// Helper function to match optical flow keypoints
std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> match_keypoints(const std::vector<cv::Point2f>& cur_kps, const std::vector<uchar>& status, const std::vector<cv::Point2f>& prev_kps) {
    std::vector<cv::Point2f> cur_matched_kp;
    std::vector<cv::Point2f> prev_matched_kp;

    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            prev_matched_kp.push_back(prev_kps[i]);
            cur_matched_kp.push_back(cur_kps[i]);
        }
    }

    return {cur_matched_kp, prev_matched_kp};
}

// Helper function to estimate partial transform
std::vector<double> estimate_partial_transform(const std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>>& matched_keypoints) {
    const std::vector<cv::Point2f>& cur_matched_kp = matched_keypoints.first;
    const std::vector<cv::Point2f>& prev_matched_kp = matched_keypoints.second;

    cv::Mat transform = cv::estimateRigidTransform(prev_matched_kp, cur_matched_kp, false);
    if (!transform.empty()) {
        double dx = transform.at<double>(0, 2);
        double dy = transform.at<double>(1, 2);
        double da = std::atan2(transform.at<double>(1, 0), transform.at<double>(0, 0));
        return {dx, dy, da};
    } else {
        return {0, 0, 0};
    }
}

// Helper function to transform a frame
Frame transform_frame(const Frame& frame, const cv::Mat& transform, int border_size, const std::string& border_type) {
    if (border_type != "black" && border_type != "reflect" && border_type != "replicate") {
        throw std::invalid_argument("Invalid border type");
    }

    cv::Mat transform_matrix = build_transformation_matrix(transform);
    auto [bordered_frame_image, border_mode] = border_frame(frame, border_size, border_type);

    int h = bordered_frame_image.rows;
    int w = bordered_frame_image.cols;
    cv::Mat transformed_frame_image;
    cv::warpAffine(bordered_frame_image, transformed_frame_image, transform_matrix, cv::Size(w, h), border_mode);

    return Frame(transformed_frame_image, "BGRA");
}

// Helper function to post-process transformed frame
std::pair<Frame, std::map<std::string, Frame>> post_process_transformed_frame(
    const Frame& transformed_frame,
    std::map<std::string, int>& border_options,
    std::map<std::string, Frame>& layer_options,
    std::map<std::string, int>& extreme_frame_corners,
    int border_size) {
    Frame cropped_frame = crop_frame(transformed_frame, border_options, extreme_frame_corners, border_size);

    if (!layer_options["layer_func"].empty()) {
        cropped_frame = apply_layer_func(cropped_frame, layer_options["prev_frame"], layer_options["layer_func"]);
        layer_options["prev_frame"] = cropped_frame;
    }

    return {cropped_frame, layer_options};
}
