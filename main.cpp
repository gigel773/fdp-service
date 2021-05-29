#include <iostream>
#include <opencv2/opencv.hpp>

#include <common_processor.hpp>

static constexpr const char face_path[] = "../photos/maks_face.jpg";
static constexpr const char haar_cascade_face_path[] = "../models/haarcascade_frontalface_default.xml";

int main()
{
	// Cut the face
	auto face_classifier = cv::CascadeClassifier(haar_cascade_face_path);
	auto frame = cv::imread(face_path);
	auto frame_gray = cv::Mat();
	auto faces = std::vector<cv::Rect>();

	auto processor = nntu::img::queue::default_impl();

	cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
	cv::equalizeHist(frame_gray, frame_gray);

	face_classifier.detectMultiScale(frame_gray, faces);

	if (faces.empty()) return 1;

	for (auto& face_coordinates: faces) {
		if (face_coordinates.height<60 && face_coordinates.width<60) continue;

		int max_side = std::max(face_coordinates.height, face_coordinates.width);
		int half_side = std::max(max_side, 60)/2;

		cv::Point center = (face_coordinates.br()+face_coordinates.tl())*0.5;
		cv::Rect face_rect(center.x-half_side, center.y-half_side, half_side*2, half_side*2);

		auto face_frame = frame(face_rect);

		processor->submit(face_frame);

		cv::imshow("Face", processor->get_result(face_frame));
		cv::waitKey(0);
	}

	return 0;
}
