#include <iostream>
#include <opencv2/opencv.hpp>

#include <filesystem>

#include <work_queue.hpp>
#include <stage.hpp>
#include <pipeline.hpp>

constexpr const char face_path[] = R"(../photos/)";

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
	auto pipeline = nntu::img::default_pipeline_impl<64>();

	std::vector<cv::Mat> faces;

	for (const auto& path: std::filesystem::directory_iterator(face_path)) {

		std::cout << "Processing: " << path.path().c_str() << std::endl;

		auto frame = cv::imread(path.path().c_str());

		faces.push_back(frame);
	}

	pipeline.process(faces.begin(), faces.end());

	for (const auto& it: faces) {
		cv::imshow("Face", it);
		cv::waitKey(0);
	}

	return 0;
}
