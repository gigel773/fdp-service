#include "facial_cut_stage.hpp"

#include <tbb/parallel_for.h>

nntu::img::facial_cut_stage::facial_cut_stage(size_t batch_size)
{
    classifier_ = cv::CascadeClassifier(haar_cascade_face_path);
}

void nntu::img::facial_cut_stage::submit(cv::Mat* begin, cv::Mat* end)
{
#ifdef WITH_TBB
    tbb::parallel_for(tbb::blocked_range<cv::Mat*>(begin, end),
            [this](tbb::blocked_range<cv::Mat*> range) {
                thread_local auto local_classifier = cv::CascadeClassifier(haar_cascade_face_path);

                for (auto it = range.begin(); it<range.end(); it++) {
                    auto frame_gray = cv::Mat();
                    auto faces = std::vector<cv::Rect>();

                    cv::cvtColor(*it, frame_gray, cv::COLOR_BGR2GRAY);
                    cv::equalizeHist(frame_gray, frame_gray);

                    local_classifier.detectMultiScale(frame_gray, faces);

                    if (faces.empty()) throw std::runtime_error("No faces were found");

                    auto face_coordinates = faces[0];
                    int max_side = std::max(face_coordinates.height, face_coordinates.width);
                    int half_side = std::max(max_side, 60)/2;

                    cv::Point center = (face_coordinates.br()+face_coordinates.tl())*0.5;
                    cv::Rect face_rect(center.x-half_side, center.y-half_side, half_side*2, half_side*2);

                    *it = (*it)(face_rect);
                }
            });
#else
    for (auto* cur = begin; cur<end; cur++) {
        auto frame_gray = cv::Mat();
        auto faces = std::vector<cv::Rect>();

        cv::cvtColor(*cur, frame_gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(frame_gray, frame_gray);

        classifier_.detectMultiScale(frame_gray, faces);

        if (faces.empty()) throw std::runtime_error("No faces were found");

        auto face_coordinates = faces[0];
        int max_side = std::max(face_coordinates.height, face_coordinates.width);
        int half_side = std::max(max_side, 60)/2;

        cv::Point center = (face_coordinates.br()+face_coordinates.tl())*0.5;
        cv::Rect face_rect(center.x-half_side, center.y-half_side, half_side*2, half_side*2);

        *cur = (*cur)(face_rect);
    }
#endif
}

void nntu::img::facial_cut_stage::wait()
{
    // No async yet
}
