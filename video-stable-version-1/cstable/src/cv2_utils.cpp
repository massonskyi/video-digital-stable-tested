#include "../includes/cv2_utils.h"
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/video/tracking.hpp>


void safe_import_cv2(){
    try{
        cv::getBuildInformation();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cerr << "No OpenCV found. Please install OpenCV." << std::endl;
        std::cerr << "You can install OpenCV from source. See the docs here:" << std::endl;
        std::cerr << "https://docs.opencv.org/3.4.1/da/df6/tutorial_py_table_of_contents_setup.html" << std::endl;
        exit(1);
    }
}

cv::Mat cv2_estimateRigidTransform(
    const std::vector<cv::Point2f>& from_pts,
    const std::vector<cv::Point2f>& to_pts,
    bool full = false
){
    if(from_pts.empty() || to_pts.empty()){
        return cv::Mat();
    }

    cv::Mat transform;

    if (CV_VERSION_MAJOR >= 4) {
        transform = cv::estimateAffinePartial2D(from_pts, to_pts);
    } else {
        transform = cv::estimateRigidTransform(from_pts, to_pts, full);
    }
    return transform;
}