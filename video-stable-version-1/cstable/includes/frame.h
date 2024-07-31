#ifndef FRAME_H
#define FRAME_H

#include "string"
#include "iostream"
#include "stdexcept"

#include "opencv4/opencv2/opencv.hpp"


class Frame final : public cv::Mat{
public:
    Frame(
        const cv::Mat& image,
        const std::string& color_format = ""
    ) : image(image), color_format(color_format.empty() ? guess_color_format() : color_format) {} 

    Frame() = default;

    __always_inline std::string get_color_format() const {
        return color_format;
    }

    cv::Mat cvt_color(const std::string& to_format) const {
        if (color_format != to_format) {
            int color_conversion = lookup_color_conversion(color_format, to_format);
            cv::Mat converted_image;
            cv::cvtColor(image, converted_image, color_conversion);
            return converted_image;
        } else {
            return image;
        }
    }

    __always_inline cv::Mat get_gray_image() const {
        return cvt_color("GRAY");
    }

    __always_inline cv::Mat get_bgr_image() const {
        return cvt_color("BGR");
    }

    __always_inline cv::Mat get_bgra_image() const {
        return cvt_color("BGRA");
    }

    __always_inline cv::Mat get_image() const {
        return image;
    }

    __always_inline std::string get_format() const {
        return color_format;
    }
private:
    cv::Mat image;
    std::string color_format;


    __always_inline std::string guess_color_format() const {
        if (image.channels() == 1) {
            return "GRAY";
        } else if (image.channels() == 3) {
            return "BGR";
        } else if (image.channels() == 4) {
            return "BGRA";
        } else {
            throw std::runtime_error("Unexpected frame image shape");
        }
    }

    __always_inline static int lookup_color_conversion(const std::string& from_format, const std::string& to_format) {
        if (from_format == "GRAY" && to_format == "BGR") {
            return cv::COLOR_GRAY2BGR;
        } else if (from_format == "BGR" && to_format == "GRAY") {
            return cv::COLOR_BGR2GRAY;
        } else if (from_format == "BGR" && to_format == "BGRA") {
            return cv::COLOR_BGR2BGRA;
        } else if (from_format == "BGRA" && to_format == "BGR") {
            return cv::COLOR_BGRA2BGR;
        } else if (from_format == "BGRA" && to_format == "GRAY") {
            return cv::COLOR_BGRA2GRAY;
        } else if (from_format == "GRAY" && to_format == "BGRA") {
            return cv::COLOR_GRAY2BGRA;
        } else {
            throw std::runtime_error("Unsupported color conversion");
        }
    }
};


#endif // FRAME_H