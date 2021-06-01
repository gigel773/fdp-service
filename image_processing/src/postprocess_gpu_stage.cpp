#include "postprocess_gpu_stage.hpp"

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

nntu::img::postprocess_gpu_stage::postprocess_gpu_stage(int required_size)
        :required_size_(required_size)
{

}

void nntu::img::postprocess_gpu_stage::submit(cv::Mat* begin, cv::Mat* end)
{
    std::vector<cv::cuda::GpuMat> gpu_in_frames(std::distance(begin, end));
    std::vector<cv::cuda::GpuMat> gpu_out_frames(std::distance(begin, end));

    for (size_t i = 0; i<std::distance(begin, end); i++) {
        gpu_in_frames[i].upload(begin[i]);
    }

    for (size_t i = 0; i<std::distance(begin, end); i++) {
        cv::cuda::cvtColor(gpu_in_frames[i], gpu_in_frames[i], cv::COLOR_BGR2GRAY);
        cv::cuda::equalizeHist(gpu_in_frames[i], gpu_in_frames[i]);
        cv::cuda::resize(gpu_in_frames[i], gpu_out_frames[i], cv::Size(required_size_, required_size_));
    }

    for (size_t i = 0; i<std::distance(begin, end); i++) {
        gpu_out_frames[i].download(begin[i]);
    }
}

void nntu::img::postprocess_gpu_stage::wait()
{

}
