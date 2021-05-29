#ifndef FDP_WORK_QUEUE_HPP
#define FDP_WORK_QUEUE_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>

#include "pipeline.hpp"

namespace nntu::img {

	template<size_t batch_size>
	class work_queue {
	public:

		explicit work_queue(size_t pool_size)
				:pool_size_(pool_size)
		{
		}

		work_queue(work_queue& other) = delete;

		auto operator=(work_queue& other) -> work_queue& = delete;

		auto attach_to(pipeline<batch_size>& value)
		{
			attached_pipeline_ = &value;
		}

	private:
		using image_pool_t = std::array<cv::Mat, batch_size>;
		using pools_t = std::vector<image_pool_t>;

		pools_t overall_pools_{};
		size_t pool_size_ = 0;
		pipeline<batch_size>* attached_pipeline_ = nullptr;
	};
}

#endif //FDP_WORK_QUEUE_HPP
