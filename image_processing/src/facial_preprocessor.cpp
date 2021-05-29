#include "facial_preprocessor.hpp"

nntu::img::facial_preprocessor::facial_preprocessor(size_t batch_size)
{
	classifier_ = cv::CascadeClassifier(haar_cascade_face_path);
}

void nntu::img::facial_preprocessor::submit(cv::Mat* begin, cv::Mat* end)
{
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
}

void nntu::img::facial_preprocessor::wait()
{
	// No async yet
}
