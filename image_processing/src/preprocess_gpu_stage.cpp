#include "preprocess_gpu_stage.hpp"

#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaarithm.hpp>

void nntu::img::preprocess_gpu_stage::submit(cv::Mat* begin, cv::Mat* end)
{
    std::vector<cv::cuda::GpuMat> gpu_in_frames(std::distance(begin, end));
    std::vector<cv::cuda::GpuMat> gpu_out_frames(std::distance(begin, end));
    auto cuda_classifier = cv::cuda::CascadeClassifier::create(haar_cascade_face_path);
    const auto filter_gaussian = cv::cuda::createGaussianFilter(CV_8UC3, CV_8UC3, cv::Size(0, 0), 3);

    for (size_t i = 0; i<std::distance(begin, end); i++) {
        gpu_in_frames[i].upload(begin[i]);
    }

    for (size_t i = 0; i<std::distance(begin, end); i++) {
        auto* cur = begin+i;

        cv::cuda::GpuMat obj_buffer;
        auto faces = std::vector<cv::Rect>();

        // Convert to gray
        cv::cuda::resize(gpu_in_frames[i], gpu_in_frames[i], cv::Size(1024, 1024));
        cv::cuda::cvtColor(gpu_in_frames[i], gpu_out_frames[i], cv::COLOR_BGR2GRAY);
        cv::cuda::equalizeHist(gpu_out_frames[i], gpu_out_frames[i]);

        // Find faces
        cuda_classifier->detectMultiScale(gpu_out_frames[i], obj_buffer);
        cuda_classifier->convert(obj_buffer, faces);

        if (faces.empty()) throw std::runtime_error("No faces were found");

        // Cut the face
        auto face_coordinates = faces[0];
        int max_side = std::max(face_coordinates.height, face_coordinates.width);
        int half_side = std::max(max_side, 60)/2;

        cv::Point center = (face_coordinates.br()+face_coordinates.tl())*0.5;
        cv::Rect face_rect(center.x-half_side, center.y-half_side, half_side*2, half_side*2);

        gpu_in_frames[i] = gpu_in_frames[i](face_rect);

        // Resize + sharpen + blur
        cv::cuda::resize(gpu_in_frames[i], gpu_in_frames[i], cv::Size(512, 512));

        filter_gaussian->apply(gpu_in_frames[i], gpu_out_frames[i]);
        cv::cuda::addWeighted(gpu_in_frames[i], 1.5, gpu_out_frames[i], -0.5, 0, gpu_out_frames[i]);
        cv::cuda::bilateralFilter(gpu_out_frames[i], gpu_out_frames[i], 5, 50, 50);
    }

    for (size_t i = 0; i<std::distance(begin, end); i++) {
        gpu_out_frames[i].upload(begin[i]);
    }
}

void nntu::img::preprocess_gpu_stage::wait()
{

}
