#include "postprocess_cpu_stage.hpp"

#include <tbb/parallel_for.h>

nntu::img::postprocess_cpu_stage::postprocess_cpu_stage(int required_size)
        :required_size_(required_size)
{

}

void nntu::img::postprocess_cpu_stage::submit(cv::Mat* begin, cv::Mat* end)
{
    tbb::parallel_for(tbb::blocked_range<cv::Mat*>(begin, end),
            [this](tbb::blocked_range<cv::Mat*> range) {
                for (auto it = range.begin(); it<range.end(); it++) {
                    cv::Mat result;

                    cv::cvtColor(*it, *it, cv::COLOR_BGR2GRAY);
                    cv::equalizeHist(*it, result);
                    cv::resize(result, result, cv::Size(required_size_, required_size_));

                    *it = result;
                }
            });
}

void nntu::img::postprocess_cpu_stage::wait()
{

}
