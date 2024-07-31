#ifndef BORDER_UTILS_H
#define BORDER_UTILS_H

#include <iostream>
#include <map>
#include <string>
#include <algorithm>
#include <cmath>

namespace cv{class Mat;}

class Frame;

std::pair<int, int> functional_border_sizes(
    int
);


Frame crop_frame(const Frame& frame, std::map<std::string, int>& border_options, std::map<std::string, int>& extreme_frame_corners, int border_size);


#endif // BORDER_UTILS_H