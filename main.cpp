#include <iostream>
#include <opencv2/opencv.hpp>

#include <filesystem>

#include <work_queue.hpp>
#include <stage.hpp>
#include <pipeline.hpp>

constexpr const char face_path[] = R"(../photos/)";
constexpr const char haar_cascade_face_path[] = R"(../models/haarcascade_frontalface_default.xml)";

/*
 * Hierarchy:
 *  - Async queue
 *  - Pipeline consists of stages
 *  - Each stage works with pre-defined number of images
 *  - Queue is being attached to pipeline
 * */

int main()
{
	// Cut the face
	auto face_classifier = cv::CascadeClassifier(haar_cascade_face_path);
	auto pipeline = nntu::img::default_pipeline_impl<64>();

	std::vector<cv::Mat> gray_faces;

	for (const auto& path: std::filesystem::directory_iterator(face_path)) {

		std::cout << "Processing: " << path.path().c_str() << std::endl;

		auto frame = cv::imread(path.path().c_str());

		auto frame_gray = cv::Mat();
		auto faces = std::vector<cv::Rect>();

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

//			processor->submit(face_frame);

			gray_faces.push_back(face_frame);
		}
	}

	pipeline.process(gray_faces.begin(), gray_faces.end());

	for (auto it: gray_faces) {
		cv::imshow("Face", it);
		cv::waitKey(0);
	}

	return 0;
}
