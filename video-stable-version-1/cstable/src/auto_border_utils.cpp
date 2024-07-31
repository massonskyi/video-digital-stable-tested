
#include <opencv4/opencv2/opencv.hpp>

#include "../includes/vidstab_utils.h"

std::map<std::string, int> extreme_corners(
    const cv::Mat& frame,
    const std::vector<cv::Mat>& transforms
){
    signed short int h = frame.rows;
    signed short int w = frame.cols;

    std::vector<cv::Point2f> frame_corners = {
        cv::Point2f(0, 0),         // top left
        cv::Point2f(0, h - 1),     // bottom left
        cv::Point2f(w - 1, 0),     // top right
        cv::Point2f(w - 1, h - 1)  // bottom right
    };


    auto min_x = 0, min_y = 0, max_x = 0, max_y = 0;

    for (const cv::Mat& transform : transforms){
        cv::Mat transformation_matrix = build_transformation_matrix(transform);
        std::vector<cv::Point2f> transformed_frame_corners(frame_corners.size());

        cv::transform(
            frame_corners,
            transformed_frame_corners,
            transformation_matrix
        );

        std::vector<float> delta_x_corners, delta_y_corners;

        for(size_t i = 0; i < frame_corners.size(); ++i){
            delta_x_corners.push_back(transformed_frame_corners[i].x - frame_corners[i].x);
            delta_y_corners.push_back(transformed_frame_corners[i].y - frame_corners[i].y);
        }

        min_x = *std::min_element(delta_x_corners.begin(), delta_x_corners.end());
        min_y = *std::min_element(delta_y_corners.begin(), delta_y_corners.end());
        max_x = *std::max_element(delta_x_corners.begin(), delta_x_corners.end());
        max_y = *std::max_element(delta_y_corners.begin(), delta_y_corners.end());
    }

    return{
        {"min_x", min_x},
        {"min_y", min_y},
        {"max_x", max_x},
        {"max_y", max_y}
    };
}


__always_inline int auto_border_start(
    int min_corner_point,
    int border_size
){
    return std::floor(border_size - std::abs(min_corner_point));
}

__always_inline int auto_border_length(
    int frame_dim,
    int extreme_corner,
    int border_size
){
    return std::ceil(frame_dim - (border_size - extreme_corner));
}

cv::Mat auto_border_crop(
    const cv::Mat& frame,
    std::map<std::string, int>& extreme_frame_corners,
    int border_size
){
    if (border_size == 0) return frame;

    signed short int frame_h = frame.rows;
    signed short int frame_w = frame.cols;

    int x = auto_border_start(extreme_frame_corners.at("mix_x"), border_size);
    int y = auto_border_start(extreme_frame_corners.at("min_y"), border_size);


    int w = auto_border_length(frame_w, extreme_frame_corners.at("max_x"), border_size);
    int h = auto_border_length(frame_h, extreme_frame_corners.at("max_y"), border_size);

    return frame(cv::Range(y, h), cv::Range(x, w));
}


int min_auto_border_size(
    const std::map<std::string, int>& extreme_frame_corners
){
    std::vector<int> abs_extreme_corners;
    
    for(const auto& pair: extreme_frame_corners){
        abs_extreme_corners.push_back(std::abs(pair.second));
    }

    return std::ceil(
        static_cast<double>(
            *std::max_element(abs_extreme_corners.begin(), abs_extreme_corners.end())
        )
    );
}