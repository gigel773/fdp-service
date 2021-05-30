#include "scale_stage.hpp"

#include <tbb/parallel_for.h>

namespace nntu::img {

    void scale_stage<scale_type::scale>::submit(cv::Mat* begin, cv::Mat* end)
    {
#ifdef WITH_TBB
        tbb::parallel_for(tbb::blocked_range<cv::Mat*>(begin, end),
                [this](tbb::blocked_range<cv::Mat*> range) {
                    for (auto it = range.begin(); it<range.end(); it++) {
                        cv::resize(*it, *it, cv::Size(), target_factor_, target_factor_);
                    }
                });
#else
        for (auto* cur = begin; cur<end; cur++) {
            cv::resize(*cur, *cur, cv::Size(), target_factor_, target_factor_);
        }
#endif
    }

    void scale_stage<scale_type::resize>::submit(cv::Mat* begin, cv::Mat* end)
    {
#if 1
        tbb::parallel_for(tbb::blocked_range<cv::Mat*>(begin, end),
                [this](tbb::blocked_range<cv::Mat*> range) {
                    for (auto it = range.begin(); it<range.end(); it++) {
                        const float factor = static_cast<float>(target_cols_)/static_cast<float>(it->cols);

                        cv::resize(*it, *it, cv::Size(), factor, factor);
                    }
                });
#else
        for (auto* cur = begin; cur<end; cur++) {
            const float factor = static_cast<float>(target_cols_)/static_cast<float>(cur->cols);

            cv::resize(*cur, *cur, cv::Size(), factor, factor);
        }
#endif
    }

    void scale_stage<scale_type::resize>::wait()
    {
        // No async yet
    }

    void scale_stage<scale_type::scale>::wait()
    {
        // No async yet
    }
}