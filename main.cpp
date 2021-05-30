#include <iostream>
#include <nips.hpp>
#include <opencv2/opencv.hpp>

#include <filesystem>

#include <work_queue.hpp>
#include <stage.hpp>
#include <pipeline.hpp>

/*
 * Hierarchy:
 *  - Async queue
 *  - Pipeline consists of stages
 *  - Each stage works with pre-defined number of images
 *  - Queue is being attached to pipeline
 * */
static constexpr const size_t batch_size = 128;
constexpr const char face_path[] = R"(../photos/)";

int main()
{
	auto work_queue = nntu::img::work_queue<batch_size>(1);
	auto pipeline = nntu::img::default_pipeline_impl<batch_size>(120);

	work_queue.attach_to(pipeline);

//	std::vector<cv::Mat> images;
//	std::atomic<size_t> images_processed = 0;
//	size_t images_submitted = 0;
//
//	for (const auto& path: std::filesystem::directory_iterator(face_path)) {
//
//		std::cout << "Processing: " << path.path().c_str() << std::endl;
//
//		work_queue.submit(cv::imread(path.path().c_str()),
//				[&images_processed, &images](cv::Mat&& img) {
//					++images_processed;
//					images.push_back(img);
//				});
//		++images_submitted;
//	}
//
//	while (images_submitted!=images_processed) {
//		work_queue.wait();
//	}
//
//	for (const auto& it: images) {
//		cv::imshow("Face", it);
//		cv::waitKey(0);
//	}

//	nntu::net::run_server("http://localhost:8081/",
//			"http://localhost:8761/");

	return 0;
}
