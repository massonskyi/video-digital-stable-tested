#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <deque>
#include <vector>
#include <numeric>
#include <thread>
#include <mutex>
#include <fstream>
#include <future>

using namespace cv;
using namespace std;

class Stabilizer {
public:
    Stabilizer(int smoothing_radius = 25, string border_type = "black", int border_size = 0, bool crop_n_zoom = false, 
               bool logging = false, double process_noise_cov = 1e-3, double measurement_noise_cov = 1e-1)
        : smoothing_radius(smoothing_radius), border_size(border_size), crop_n_zoom(crop_n_zoom), logging(logging) {

        if (border_type == "black") border_mode = BORDER_CONSTANT;
        else if (border_type == "reflect") border_mode = BORDER_REFLECT;
        else if (border_type == "reflect_101") border_mode = BORDER_REFLECT_101;
        else if (border_type == "replicate") border_mode = BORDER_REPLICATE;
        else if (border_type == "wrap") border_mode = BORDER_WRAP;
        else border_mode = BORDER_CONSTANT;

        clahe = createCLAHE(2.0, Size(8, 8));

        // Initialize Kalman filter
        kalman.init(6, 3, 0);
        kalman.transitionMatrix = (Mat_<float>(6, 6) <<
            1, 0, 0, 1, 0, 0,
            0, 1, 0, 0, 1, 0,
            0, 0, 1, 0, 0, 1,
            0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 1);

        kalman.measurementMatrix = Mat::eye(3, 6, CV_32F);

        // Increase Kalman filter coefficients for stronger smoothing
        setIdentity(kalman.processNoiseCov, Scalar::all(process_noise_cov)); 
        setIdentity(kalman.measurementNoiseCov, Scalar::all(measurement_noise_cov)); 
        setIdentity(kalman.errorCovPost, Scalar::all(1)); // Initial error covariance

        if (logging) {
            log_file.open("stabilizer_log.txt");
        }
    }

    ~Stabilizer() {
        if (logging) {
            log_file.close();
        }
    }

    Mat stabilize(const Mat& frame) {
        if (frame.empty()) return Mat();

        lock_guard<mutex> lock(frame_queue_mutex);
        if (frame_queue.empty()) {
            initialize(frame);
            return frame;
        } else if (frame_queue.size() < smoothing_radius) {
            frame_queue.push_back(frame);
            generate_transformations(frame);
            return frame;
        } else {
            frame_queue.push_back(frame);
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
        goodFeaturesToTrack(gray, previous_keypoints, 750, 0.05, 30.0, Mat(), 3, false, 0.04);
        frame_height = frame.rows;
        clahe->apply(gray, gray);
        frame_width = frame.cols;
        frame_queue.push_back(frame);
        previous_gray = gray.clone();

        // Initialize Kalman state
        kalman.statePost = (Mat_<float>(6, 1) << 0, 0, 0, 0, 0, 0);
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
            transformation = findHomography(valid_previous_keypoints, valid_curr_kps, RANSAC);
        } else {
            transformation = Mat::eye(3, 3, CV_64F);
        }

        double dx = transformation.at<double>(0, 2);
        double dy = transformation.at<double>(1, 2);
        double da = atan2(transformation.at<double>(1, 0), transformation.at<double>(0, 0));

        Vec3d frame_transform(dx, dy, da);
        transforms.push_back(frame_transform);

        if (transforms.size() > smoothing_radius) {
            transforms.pop_front();
        }

        goodFeaturesToTrack(gray, previous_keypoints, 750, 0.05, 30.0, Mat(), 3, false, 0.04);
        previous_gray = gray.clone();

        // Update Kalman filter
        Mat measurement = (Mat_<float>(3, 1) << dx, dy, da);
        kalman.correct(measurement);

        if (logging) {
            log_file << "Transformation: dx = " << dx << ", dy = " << dy << ", da = " << da << endl;
        }
    }

    void apply_transformations() {
        Mat frame = frame_queue.front();
        frame_queue.pop_front();

        // Predict using Kalman filter
        Mat prediction = kalman.predict();
        double dx = prediction.at<float>(0);
        double dy = prediction.at<float>(1);
        double da = prediction.at<float>(2);

        // Apply average of previous transformations
        Vec3d avg_transform = average_transform();
        dx += avg_transform[0];
        dy += avg_transform[1];
        da += avg_transform[2];

        Mat transform = (Mat_<double>(3, 3) << cos(da), -sin(da), dx, sin(da), cos(da), dy, 0, 0, 1);

        Mat bordered_frame;
        copyMakeBorder(frame, bordered_frame, border_size, border_size, border_size, border_size, border_mode, Scalar(0, 0, 0));

        Mat frame_wrapped;
        warpPerspective(bordered_frame, frame_wrapped, transform, bordered_frame.size(), INTER_LINEAR, border_mode, Scalar(0, 0, 0));

        stabilized_frame = frame_wrapped(Rect(border_size, border_size, frame_width, frame_height)).clone();

        if (crop_n_zoom) {
            Rect roi(border_size, border_size, frame_width - 2 * border_size, frame_height - 2 * border_size);
            Mat frame_cropped = stabilized_frame(roi);
            resize(frame_cropped, stabilized_frame, Size(frame_width, frame_height), 0, 0, INTER_LINEAR);
        }

        if (logging) {
            log_file << "Applied transformation: dx = " << dx << ", dy = " << dy << ", da = " << da << endl;
        }
    }

    Vec3d average_transform() {
        double sum_dx = 0, sum_dy = 0, sum_da = 0;
        for (const Vec3d& trans : transforms) {
            sum_dx += trans[0];
            sum_dy += trans[1];
            sum_da += trans[2];
        }
        int n = transforms.size();
        return Vec3d(sum_dx / n, sum_dy / n, sum_da / n);
    }

    int smoothing_radius;
    int border_size;
    bool crop_n_zoom;
    bool logging;
    int border_mode;
    Ptr<CLAHE> clahe;
    deque<Mat> frame_queue;
    deque<Vec3d> transforms;
    Mat previous_gray;
    vector<Point2f> previous_keypoints;
    int frame_height, frame_width;
    Mat stabilized_frame;

    // Kalman filter
    KalmanFilter kalman;

    // Mutex for thread safety
    mutex frame_queue_mutex;

    // Logging
    ofstream log_file;
};

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <video_file>" << endl;
        return -1;
    }

    string source = argv[1];
    VideoCapture cap;
    if (isdigit(source[0])) {
        cap.open(stoi(source));  // Open camera
    } else {
        cap.open(source);  // Open video file
    }

    if (!cap.isOpened()) {
        cerr << "Error opening video source" << endl;
        return -1;
    }

    Stabilizer stabilizer(25, "black", 0, false, false, 1e-3, 1e-1);

    namedWindow("Stabilized Video", WINDOW_NORMAL);
    Mat frame, stabilized_frame, combinedFrame;

    auto start_time = chrono::high_resolution_clock::now();
    int frame_count = 0;

    while (cap.read(frame)) {
        auto frame_time = chrono::high_resolution_clock::now();
        stabilized_frame = stabilizer.stabilize(frame);

        if (!stabilized_frame.empty()) {
            hconcat(frame, stabilized_frame, combinedFrame);

            // Measure FPS
            auto end_time = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = end_time - frame_time;
            double fps = 1.0 / elapsed.count();

            // Display FPS on the frame
            putText(combinedFrame, "FPS: " + to_string(static_cast<int>(fps)), Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);

            // Display the combined frame
            imshow("Original and Stabilized Frames", combinedFrame);

            // Exit on any key press
            if (waitKey(1) >= 0) break; 

        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}