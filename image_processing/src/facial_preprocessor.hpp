
#ifndef FDP_FACIAL_PREPROCESSOR_HPP
#define FDP_FACIAL_PREPROCESSOR_HPP

#include <common_processor.hpp>
#include <atomic>

namespace nntu::img {

	class facial_preprocessor final : public queue {
	public:
		explicit facial_preprocessor(size_t batch_size);

		void submit(const cv::Mat& frame) override;

		auto get_result(const std::vector<cv::Mat>& input) -> std::vector<cv::Mat> override;

	private:
		std::vector<cv::Mat> image_pool_;
		std::atomic<size_t> image_count_;
	};

}
#endif //FDP_FACIAL_PREPROCESSOR_HPP
