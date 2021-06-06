#include "preprocess_cpu_stage.hpp"

#include <execution>
#include <tbb/parallel_for.h>

void nntu::img::preprocess_cpu_stage::submit(cv::Mat* begin, cv::Mat* end)
{
    tbb::parallel_for(tbb::blocked_range<cv::Mat*>(begin, end),
            [this](tbb::blocked_range<cv::Mat*> range) {
                thread_local auto local_classifier = cv::CascadeClassifier(haar_cascade_face_path);
                thread_local auto kernel = cv::Mat(3, 3, CV_32S);

                for (int i = 0; i<3; i++) {
                    for (int j = 0; j<3; j++) {
                        kernel.at<int>(i, j) = -1;
                    }
                }

                kernel.at<int>(1, 1) = 9;

                for (auto it = range.begin(); it<range.end(); it++) {
                    auto frame_gray = cv::Mat();
                    auto blur_result = cv::Mat();
                    auto faces = std::vector<cv::Rect>();

                    cv::resize(*it, *it, cv::Size(1024, 1024));
                    cv::cvtColor(*it, frame_gray, cv::COLOR_BGR2GRAY);
                    cv::equalizeHist(frame_gray, frame_gray);

                    local_classifier.detectMultiScale(frame_gray, faces);

                    if (faces.empty()) throw std::runtime_error("No faces were found");

                    auto face_coordinates = *std::max_element(std::execution::par_unseq,
                            faces.begin(),
                            faces.end(),
                            [](const cv::Rect& a, const cv::Rect& b) -> bool {
                                return a.area()<b.area();
                            });
                    int max_side = std::max(face_coordinates.height, face_coordinates.width);
                    int half_side = std::max(max_side, 60)/2;

                    cv::Point center = (face_coordinates.br()+face_coordinates.tl())*0.5;
                    cv::Rect face_rect(center.x-half_side, center.y-half_side, half_side*2, half_side*2);

                    *it = (*it)(face_rect);

                    cv::resize(*it, *it, cv::Size(512, 512));
                    cv::filter2D(*it, *it, -1, kernel);

                    cv::medianBlur(*it, *it, 3);
                    cv::bilateralFilter(*it, blur_result, 5, 50, 50);

                    *it = blur_result;
                }
            });
}

void nntu::img::preprocess_cpu_stage::wait()
{

}
