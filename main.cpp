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

constexpr const size_t batch_size = 64;

int main()
{
	auto work_queue = nntu::img::work_queue<batch_size>(1);
	auto pipeline = nntu::img::default_pipeline_impl<batch_size>(120);

	work_queue.attach_to(pipeline);

	for (const auto& path: std::filesystem::directory_iterator(face_path)) {

		std::cout << "Processing: " << path.path().c_str() << std::endl;

		work_queue.submit(cv::imread(path.path().c_str()));
	}

	work_queue.force_processing();
	work_queue.wait();

	for (const auto& it: work_queue) {
		cv::imshow("Face", it);
		cv::waitKey(0);
	}

	return 0;
}
