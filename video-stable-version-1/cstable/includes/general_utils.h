#ifndef GENERAL_UTILS_H
#define GENERAL_UTILS_H


#include <iostream>
#include <climits>

namespace cv{
    class Mat;
}


cv::Mat bfill_rolling_mean(
    const cv::Mat& arr, 
    int n = 30
);

// Helper function to create progress bar for stabilizing processes
class IncrementalBar {
public:
    IncrementalBar(const std::string& message, int max, const std::string& suffix)
        : message(message), max(max), suffix(suffix), current(0) {}

    void update() {
        current++;
        int percent = static_cast<int>((current / static_cast<double>(max)) * 100);
        std::cout << "\r" << message << " |" << std::string(percent / 2, '█') << std::string(50 - percent / 2, ' ') << "| " << percent << "% " << suffix << std::flush;
    }

private:
    std::string message;
    int max;
    std::string suffix;
    int current;
};

IncrementalBar* init_progress_bar(int frame_count, int max_frames, bool show_progress = true, bool gen_all = false) {
    if (!show_progress) {
        return nullptr;
    }

    bool bad_frame_count = frame_count <= 0;
    bool use_max_frames = bad_frame_count || frame_count > max_frames;

    if (bad_frame_count && max_frames == INT_MAX) {
        std::cerr << "Warning: No progress bar will be shown. (Unable to grab frame count & no max_frames provided.)\n";
        return nullptr;
    }

    int max_bar = use_max_frames ? max_frames : frame_count;
    std::string message = progress_message(gen_all);

    return new IncrementalBar(message, max_bar, "%(percent)d%%");
}


std::string progress_message(bool gen_all);

bool playback_video(
    const cv::Mat& display_frame, 
    bool playback_flag, 
    int delay, 
    int max_display_width = 750
);
#endif // GENERAL_UTILS_H