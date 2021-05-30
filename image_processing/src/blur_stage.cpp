#include "blur_stage.hpp"

#include <tbb/parallel_for.h>

void nntu::img::blur_stage::submit(cv::Mat* begin, cv::Mat* end)
{
#ifdef WITH_TBB
    tbb::parallel_for(tbb::blocked_range<cv::Mat*>(begin, end),
            [](tbb::blocked_range<cv::Mat*> range) {
                for (auto it = range.begin(); it<range.end(); it++) {
                    cv::Mat result;

                    cv::medianBlur(*it, *it, 3);
                    cv::bilateralFilter(*it, result, 5, 50, 50);

                    *it = result;
                }
            });
#else
    for (auto* cur = begin; cur<end; cur++) {
        cv::Mat result;

        cv::medianBlur(*cur, *cur, 3);
        cv::bilateralFilter(*cur, result, 5, 50, 50);

        *cur = result;
    }
#endif
}

void nntu::img::blur_stage::wait()
{
    // No async yet
}
