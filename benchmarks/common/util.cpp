#include <filesystem>

#include "util.hpp"

namespace nntu::test {
    auto load_images(const char* path, size_t image_count) -> std::vector<cv::Mat>
    {
        std::vector<cv::Mat> result;

        while (result.size()<image_count) {
            for (const auto& it: std::filesystem::directory_iterator(path)) {
                result.push_back(cv::imread(it.path().string()));

                if (result.size()>=image_count) {
                    break;
                }
            }
        }

        return result;
    }
}