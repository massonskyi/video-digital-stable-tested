#include "../includes/border_utils.h"

#include "opencv4/opencv2/opencv.hpp"
#include "../includes/auto_border_utils.h"
#include "../includes/frame.h"

std::pair<int, int> functional_border_sizes(
    int border_size
){
    int neg_border_size = 0;

    if(border_size < 0){
        neg_border_size = 100 + std::abs(border_size);
        border_size = 100;
    }
    return {border_size, neg_border_size};
}


// Функция для обрезки кадра
Frame crop_frame(const Frame& frame, std::map<std::string, int>& border_options, std::map<std::string, int>& extreme_frame_corners, int border_size) {
    if (!border_options["auto_border_flag"] && border_options["neg_border_size"] == 0) {
        return frame;
    }

    cv::Mat cropped_frame_image;
    if (border_options["auto_border_flag"]) {
        cropped_frame_image = auto_border_crop(frame.get_image(),
                                                extreme_frame_corners, 
                                                border_size
        );
    } else {
        int frame_h = frame.get_image().rows;
        int frame_w = frame.get_image().cols;
        int neg_border_size = border_options["neg_border_size"];
        cropped_frame_image = frame.get_image()(cv::Range(neg_border_size, frame_h - neg_border_size),
                                                cv::Range(neg_border_size, frame_w - neg_border_size));
    }

    return Frame(cropped_frame_image, frame.get_color_format());
}
