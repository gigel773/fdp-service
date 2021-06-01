#ifndef FDP_UTIL_HPP
#define FDP_UTIL_HPP

#include <opencv2/opencv.hpp>
#include <vector>

namespace nntu::test {
    auto load_images(const char* path, size_t image_count) -> std::vector<cv::Mat>;
}

#endif //FDP_UTIL_HPP
