#ifndef MAIN_UTILS_H
#define MAIN_UTILS_H

#include <iostream>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <stdexcept>
#include <limits>
#include <algorithm>

namespace cv {class Mat;}

int str_int(
    const std::string& v
);


bool str_2_bool(
    const std::string& v
);

double process_max_frames_arg(
    int max_frames_arg
);

cv::Mat (*process_layer_frames_arg(bool layer_frames_arg))(
    const cv::Mat&, const cv::Mat&
);
std::string process_border_size_arg(
    const std::string& border_size_arg
);

void cli_stabilizer(const std::unordered_map<std::string, std::string>& args);
#endif // MAIN_UTILS_H