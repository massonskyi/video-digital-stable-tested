#include "../includes/general_utils.h"
#include "opencv4/opencv2/opencv.hpp"

std::string progress_message(bool gen_all) {
    return gen_all ? "Generating Transforms" : "Stabilizing";
}
bool playback_video(const cv::Mat& display_frame, bool playback_flag, int delay, int max_display_width = 750) {
    if (!playback_flag) {
        return false;
    }

    if (display_frame.cols > max_display_width) {
        float scale = static_cast<float>(max_display_width) / display_frame.cols;
        cv::Mat resized_frame;
        cv::resize(display_frame, resized_frame, cv::Size(), scale, scale);
        cv::imshow("VidStab Playback (" + std::to_string(delay) + " frame delay if using live video; press Q or ESC to quit)", resized_frame);
    } else {
        cv::imshow("VidStab Playback (" + std::to_string(delay) + " frame delay if using live video; press Q or ESC to quit)", display_frame);
    }

    int key = cv::waitKey(1);
    if (key == 'q' || key == 27) {
        return true;
    }

    return false;
}

cv::Mat bfill_rolling_mean(const cv::Mat& arr, int n = 30) {
    if (arr.rows < n) {
        throw std::invalid_argument("arr.rows cannot be less than n");
    }
    if (n == 1) {
        return arr;
    }

    cv::Mat pre_buffer = cv::Mat::zeros(1, 3, CV_64F);
    cv::Mat post_buffer = cv::Mat::zeros(n, 3, CV_64F);
    cv::Mat arr_cumsum;
    cv::vconcat(std::vector<cv::Mat>{pre_buffer, arr, post_buffer}, arr_cumsum);

    cv::Mat buffer_roll_mean = (arr_cumsum.rowRange(n, arr_cumsum.rows) - arr_cumsum.rowRange(0, arr_cumsum.rows - n)) / static_cast<double>(n);
    cv::Mat trunc_roll_mean = buffer_roll_mean.rowRange(0, buffer_roll_mean.rows - n);

    int bfill_size = arr.rows - trunc_roll_mean.rows;
    cv::Mat bfill = cv::repeat(trunc_roll_mean.row(0), bfill_size, 1);

    cv::Mat result;
    cv::vconcat(std::vector<cv::Mat>{bfill, trunc_roll_mean}, result);
    return result;
}