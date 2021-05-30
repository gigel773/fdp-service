#include "equalize_stage.hpp"

#include <tbb/parallel_for.h>

void nntu::img::equalize_stage::submit(cv::Mat* begin, cv::Mat* end)
{
#ifdef WITH_TBB
    tbb::parallel_for(tbb::blocked_range<cv::Mat*>(begin, end),
            [](tbb::blocked_range<cv::Mat*> range) {
                for (auto it = range.begin(); it<range.end(); it++) {
                    cv::Mat result;

                    cv::cvtColor(*it, *it, cv::COLOR_BGR2GRAY);
                    cv::equalizeHist(*it, result);

                    *it = result;
                }
            });
#else
    for (auto* cur = begin; cur<end; cur++) {
        cv::Mat result;

        cv::cvtColor(*cur, *cur, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(*cur, result);

        *cur = result;
    }
#endif
}

void nntu::img::equalize_stage::wait()
{
    // No async yet
}
