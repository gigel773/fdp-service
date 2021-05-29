#include "facial_preprocessor.hpp"

nntu::img::facial_preprocessor::facial_preprocessor(size_t batch_size)
{
	image_pool_.reserve(batch_size);
}

void nntu::img::facial_preprocessor::submit(const cv::Mat& frame)
{
	image_pool_.push_back(frame);
	image_count_++;
}

auto nntu::img::facial_preprocessor::get_result(const std::vector<cv::Mat>& input) -> std::vector<cv::Mat>
{
	std::vector<cv::Mat> results;

	results.reserve(image_count_);

	for (const auto& it: image_pool_) {
		// Processing
		results.push_back(it.clone());
	}

	image_pool_.clear();
	image_count_ = 0;

	return results;
}

auto nntu::img::preprocessor_impl(size_t batch_size) -> std::shared_ptr<queue>
{
	return std::make_shared<facial_preprocessor>(batch_size);
}

