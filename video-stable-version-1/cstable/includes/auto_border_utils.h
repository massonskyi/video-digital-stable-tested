#ifndef AUTO_BORDER_UTILS_H
#define AUTO_BORDER_UTILS_H



#include <iostream>
#include <vector>
#include <algorithm>
#include <map>

namespace cv{class Mat;}


std::map<std::string, int> extreme_corners(
    const cv::Mat&,
    const std::vector<cv::Mat>&
);


__always_inline int auto_border_start(
    int,
    int
);

__always_inline int auto_border_length(
    int,
    int,
    int
);

cv::Mat auto_border_crop(
    const cv::Mat&,
    std::map<std::string, int>&,
    int
);

int min_auto_border_size(
    const std::map<std::string, int>&
);




#endif // AUTO_BORDER_UTILS_H
