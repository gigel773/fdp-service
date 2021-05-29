#ifndef FDP_COMMON_PROCESSOR_HPP
#define FDP_COMMON_PROCESSOR_HPP

#include <cstdint>
#include <cstddef>

namespace nntu::img {

	class event {

	};

	class queue {
	public:
		virtual void set_queue_size(size_t value) = 0;

		virtual void submit(const cv::Mat& frame) = 0;

		virtual auto get_result(const cv::Mat& input) -> cv::Mat = 0;

		static auto default_impl() -> std::shared_ptr<queue>;
	};
}

#endif //FDP_COMMON_PROCESSOR_HPP
