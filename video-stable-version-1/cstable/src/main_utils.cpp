#include "../includes/main_utils.h"
#include "../includes/layer_utils.h"
#include "../includes/vidstable.h"
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>


int str_int(const std::string& v) {
    try {
        return std::stoi(v);
    } catch (const std::invalid_argument&) {
        return 0; // or handle the error as needed
    }
}

// Helper function to convert string to bool from different possible strings
bool str_2_bool(const std::string& v) {
    static const std::unordered_set<std::string> true_set = {"yes", "true", "t", "y", "1"};
    static const std::unordered_set<std::string> false_set = {"no", "false", "f", "n", "0"};

    std::string lower_v = v;
    std::transform(lower_v.begin(), lower_v.end(), lower_v.begin(), ::tolower);

    if (true_set.count(lower_v)) {
        return true;
    } else if (false_set.count(lower_v)) {
        return false;
    } else {
        throw std::invalid_argument("Boolean value expected.");
    }
}

// Helper function to handle maxFrames arg in vidstab.__main__
double process_max_frames_arg(int max_frames_arg) {
    if (max_frames_arg > 0) {
        return max_frames_arg;
    }
    return std::numeric_limits<double>::infinity();
}

// Helper function to handle layerFrames arg in vidstab.__main__
cv::Mat (*process_layer_frames_arg(bool layer_frames_arg))(const cv::Mat&, const cv::Mat&) {
    if (layer_frames_arg) {
        return layer_overlay;
    }
    return nullptr;
}

// Helper function to handle borderSize arg in vidstab.__main__
std::string process_border_size_arg(const std::string& border_size_arg) {
    if (border_size_arg != "auto") {
        std::cerr << "Warning: Invalid borderSize provided; converting to 0." << std::endl;
        return "0";
    }
    return border_size_arg;
}

// Helper function to handle CLI vidstab processing
void cli_stabilizer(const std::unordered_map<std::string, std::string>& args) {
    double max_frames = process_max_frames_arg(std::stoi(args.at("maxFrames")));
    std::string border_size = process_border_size_arg(args.at("borderSize"));
    cv::Mat (*layer_func)(const cv::Mat&, const cv::Mat&) = process_layer_frames_arg(str_2_bool(args.at("layerFrames")));

    // Initialize stabilizer with user-specified keypoint detector
    VidStab stabilizer(args.at("keyPointMethod"));

    // Stabilize input video and write to specified output file
    stabilizer.stabilize(args.at("input"), args.at("output"),
                         std::stoi(args.at("smoothWindow")), max_frames,
                         args.at("borderType"), border_size, layer_func,
                         str_2_bool(args.at("playback")));
}