#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <deque>
#include <vector>

using namespace cv;
using namespace std;

class Stabilizer {
public:
    Stabilizer(int smoothing_radius = 25, string border_type = "black", int border_size = 0, bool crop_n_zoom = false, bool logging = false)
        : smoothing_radius(smoothing_radius), border_size(border_size), crop_n_zoom(crop_n_zoom), logging(logging) {
        
        if (border_type == "black") border_mode = BORDER_CONSTANT;
        else if (border_type == "reflect") border_mode = BORDER_REFLECT;
        else if (border_type == "reflect_101") border_mode = BORDER_REFLECT_101;
        else if (border_type == "replicate") border_mode = BORDER_REPLICATE;
        else if (border_type == "wrap") border_mode = BORDER_WRAP;
        else border_mode = BORDER_CONSTANT;

        clahe = createCLAHE(2.0, Size(8, 8));
        box_filter = Mat::ones(1, smoothing_radius, CV_32F) / smoothing_radius;
    }

    Mat stabilize(const Mat& frame) {
        if (frame.empty()) return Mat();

        if (frame_queue.empty()) {
            initialize(frame);
            return frame;
        } else if (frame_queue.size() < smoothing_radius) {
            frame_queue.push_back(frame);
            frame_queue_indexes.push_back(frame_queue_indexes.back() + 1);
            generate_transformations(frame);
            return frame;
        } else {
            frame_queue.push_back(frame);
            frame_queue_indexes.push_back(frame_queue_indexes.back() + 1);
            generate_transformations(frame);
            apply_transformations();
            return stabilized_frame;
        }
    }

private:
    void initialize(const Mat& frame) {
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        clahe->apply(gray, gray);
        goodFeaturesToTrack(gray, previous_keypoints, 200, 0.05, 30.0, Mat(), 3, false, 0.04);
        frame_height = frame.rows;
        frame_width = frame.cols;
        frame_queue.push_back(frame);
        frame_queue_indexes.push_back(0);
        previous_gray = gray.clone();
    }

    void generate_transformations(const Mat& frame) {
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        clahe->apply(gray, gray);

        vector<Point2f> curr_kps;
        vector<uchar> status;
        vector<float> err;
        TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
        calcOpticalFlowPyrLK(previous_gray, gray, previous_keypoints, curr_kps, status, err, Size(31, 31), 3, termcrit, 0, 0.001);

        vector<Point2f> valid_curr_kps, valid_previous_keypoints;
        for (size_t i = 0; i < status.size(); i++) {
            if (status[i]) {
                valid_curr_kps.push_back(curr_kps[i]);
                valid_previous_keypoints.push_back(previous_keypoints[i]);
            }
        }

        Mat transformation;
        if (valid_curr_kps.size() >= 4 && valid_previous_keypoints.size() >= 4) {
            transformation = estimateAffinePartial2D(valid_previous_keypoints, valid_curr_kps);
        } else {
            transformation = Mat::eye(2, 3, CV_64F);
        }

        if (transforms.empty()) {
            transforms = Mat::zeros(1, smoothing_radius, CV_64FC3);
        }

        double dx = transformation.at<double>(0, 2);
        double dy = transformation.at<double>(1, 2);
        double da = atan2(transformation.at<double>(1, 0), transformation.at<double>(0, 0));

        transforms.at<Vec3d>(0, frame_queue_indexes.back() % smoothing_radius) = Vec3d(dx, dy, da);

        frame_transform = Mat(transforms).reshape(1);
        path = cumsum(frame_transform);
        smoothed_path = path.clone();

        goodFeaturesToTrack(gray, previous_keypoints, 200, 0.05, 30.0, Mat(), 3, false, 0.04);
        previous_gray = gray.clone();
    }

    void apply_transformations() {
        Mat frame = frame_queue.front();
        frame_queue.pop_front();
        frame_queue_indexes.pop_front();

        Mat bordered_frame;
        copyMakeBorder(frame, bordered_frame, border_size, border_size, border_size, border_size, border_mode, Scalar(0, 0, 0));

        Vec3d transform_smoothed = transforms.at<Vec3d>(0, frame_queue_indexes.front() % smoothing_radius);
        double dx = transform_smoothed[0];
        double dy = transform_smoothed[1];
        double da = transform_smoothed[2];

        Mat transform = (Mat_<double>(2, 3) << cos(da), -sin(da), dx, sin(da), cos(da), dy);

        Mat frame_wrapped;
        warpAffine(bordered_frame, frame_wrapped, transform, bordered_frame.size(), INTER_LINEAR, border_mode, Scalar(0, 0, 0));

        stabilized_frame = frame_wrapped(Rect(border_size, border_size, frame_width, frame_height)).clone();

        if (crop_n_zoom) {
            Rect roi(border_size, border_size, frame_width - 2 * border_size, frame_height - 2 * border_size);
            Mat frame_cropped = stabilized_frame(roi);
            resize(frame_cropped, stabilized_frame, Size(frame_width, frame_height), 0, 0, INTER_LINEAR);
        }
    }

    Mat cumsum(const Mat& src) {
        Mat dst = src.clone();
        for (int i = 1; i < dst.rows; ++i) {
            dst.row(i) += dst.row(i - 1);
        }
        return dst;
    }

    int smoothing_radius;
    int border_size;
    bool crop_n_zoom;
    bool logging;
    int border_mode;
    Ptr<CLAHE> clahe;
    Mat box_filter;
    deque<Mat> frame_queue;
    deque<int> frame_queue_indexes;
    Mat transforms;
    Mat frame_transform;
    Mat path;
    Mat smoothed_path;
    Mat frame_transforms_smoothed;
    Mat previous_gray;
    vector<Point2f> previous_keypoints;
    int frame_height, frame_width;
    Mat stabilized_frame;
};
