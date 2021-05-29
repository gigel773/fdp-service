#ifndef FDP_COMMON_PROCESSOR_HPP
#define FDP_COMMON_PROCESSOR_HPP

#include <cstdint>
#include <cstddef>

namespace nntu::img {

	class event {

	};

	class queue {
	public:
		virtual void submit(const cv::Mat& frame) = 0;

		virtual auto get_result(const std::vector<cv::Mat>& input) -> std::vector<cv::Mat> = 0;

		static auto default_impl(size_t batch_size) -> std::shared_ptr<queue>;
	};
}

#endif //FDP_COMMON_PROCESSOR_HPP
