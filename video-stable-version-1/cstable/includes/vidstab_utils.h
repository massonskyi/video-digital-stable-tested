#ifndef VIDSTAB_UTILS_H
#define VIDSTAB_UTILS_H


namespace cv{class Mat;}
class Frame;

cv::Mat build_transformation_matrix(
    const cv::Mat&
);

std::pair<cv::Mat, int> border_frame(
    const Frame& frame, 
    int border_size, 
    const std::string& border_type
);
std::pair<Frame, std::map<std::string, Frame>> post_process_transformed_frame(
    const Frame& transformed_frame,
    std::map<std::string, int>& border_options,
    std::map<std::string, Frame>& layer_options,
    std::map<std::string, int>& extreme_frame_corners,
    int border_size) ;
#endif // VIDSTAB_UTILS_H
