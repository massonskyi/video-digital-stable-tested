#ifndef CV2_UTILS_H
#define CV2_UTILS_H

#include <vector>

namespace cv {
    class Mat;
    class Point2f;
}

void safe_import_cv2();

cv::Mat cv2_estimateRigidTransform(
    const std::vector<cv::Point2f>& from_pts,
    const std::vector<cv::Point2f>& to_pts,
    bool full = false
);


#endif // CV2_UTILS_H