#include "sharpen_stage.hpp"

#include <tbb/parallel_for.h>

nntu::img::sharpen_stage::sharpen_stage()
{
    kernel_ = cv::Mat(3, 3, CV_32S);

    for (int i = 0; i<3; i++) {
        for (int j = 0; j<3; j++) {
            kernel_.at<int>(i, j) = -1;
        }
    }

    kernel_.at<int>(1, 1) = 9;
}

void nntu::img::sharpen_stage::submit(cv::Mat* begin, cv::Mat* end)
{
#ifdef WITH_TBB
    tbb::parallel_for(tbb::blocked_range<cv::Mat*>(begin, end),
            [this](tbb::blocked_range<cv::Mat*> range) {
                for (auto it = range.begin(); it<range.end(); it++) {
                    cv::filter2D(*it, *it, -1, kernel_);
                }
            });
#else
    for (auto* cur = begin; cur<end; cur++) {
        cv::filter2D(*cur, *cur, -1, kernel_);
    }
#endif
}

void nntu::img::sharpen_stage::wait()
{
    // No async yet
}
