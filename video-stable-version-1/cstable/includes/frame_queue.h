#ifndef FRAME_QUEUE_H
#define FRAME_QUEUE_H

#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <deque>
#include <vector>
#include <stdexcept>
#include <limits>
#include "frame.h"
#include "pop_deque.h"


// Класс FrameQueue для управления очередью кадров
class FrameQueue {
public:
    FrameQueue(size_t max_len = std::numeric_limits<size_t>::max(), size_t max_frames = std::numeric_limits<size_t>::max())
        : max_len(max_len), max_frames(max_frames), _max_frames(max_frames),
          frames(max_len), inds(max_len), i(0), source_fps(30), grabbed_frame(false) {}

    void reset_queue(size_t max_len = std::numeric_limits<size_t>::max(), size_t max_frames = std::numeric_limits<size_t>::max()) {
        max_len = max_len != std::numeric_limits<size_t>::max() ? max_len : this->max_len;
        max_frames = max_frames != std::numeric_limits<size_t>::max() ? max_frames : this->max_frames;

        if (max_frames != std::numeric_limits<size_t>::max()) {
            _max_frames = max_frames + 1;
        }

        frames.clear();
        inds.clear();
        i = 0;
    }

    void set_frame_source(cv::VideoCapture& source) {
        this->source = &source;
        source_frame_count = static_cast<size_t>(source.get(cv::CAP_PROP_FRAME_COUNT));
        source_fps = static_cast<size_t>(source.get(cv::CAP_PROP_FPS));

        if (source_frame_count > 0 && max_frames == std::numeric_limits<size_t>::max()) {
            _max_frames = source_frame_count;
        } else if (max_frames != std::numeric_limits<size_t>::max() && source_frame_count < max_frames) {
            _max_frames = source_frame_count;
        }
    }

    std::tuple<size_t, Frame, bool> read_frame(bool pop_ind = true, const cv::Mat* array = nullptr) {
        cv::Mat frame;
        if (source) {
            grabbed_frame = source->read(frame);
        } else if (array) {
            frame = *array;
        }

        return _append_frame(frame, pop_ind);
    }

private:
    std::tuple<size_t, Frame, bool> _append_frame(const cv::Mat& frame, bool pop_ind = true) {
        Frame popped_frame;
        if (!frame.empty()) {
            frames.pop_append(Frame(frame));
            inds.pop_append(i);
            i++;
        }

        if (pop_ind && i == 0) {
            i = inds.pop_front();
        }

        bool break_flag = false;
        if (pop_ind && i != 0 && max_frames != std::numeric_limits<size_t>::max()) {
            break_flag = i >= max_frames - 1;
        }

        return std::make_tuple(i, popped_frame, break_flag);
    }

    void populate_queue(size_t smoothing_window) {
        size_t n = std::min(smoothing_window, max_frames);

        for (size_t i = 0; i < n; ++i) {
            auto [index, frame, break_flag] = read_frame(false);
            if (!grabbed_frame) {
                break;
            }
        }
    }

    bool frames_to_process() const {
        return !frames.size() == 0 || grabbed_frame;
    }

    size_t max_len;
    size_t max_frames;
    size_t _max_frames;

    PopDeque<Frame> frames;
    PopDeque<size_t> inds;
    size_t i;

    cv::VideoCapture* source;
    size_t source_frame_count;
    size_t source_fps;

    bool grabbed_frame;
};

#endif // FRAME_QUEUE_H
