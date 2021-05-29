#ifndef FDP_WORK_QUEUE_HPP
#define FDP_WORK_QUEUE_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <thread>
#include <future>

#include "pipeline.hpp"

namespace nntu::img {

	template<size_t batch_size>
	class work_queue {
		using image_pool_t = std::array<cv::Mat, batch_size>;
		using pools_t = std::vector<image_pool_t>;

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

		// TODO: Should be thread safe
		void submit(cv::Mat&& img)
		{
			work_queue::wait();

			image_pool_[pool_position_idx_] = std::move(img);
			++pool_position_idx_;

			if (pool_position_idx_<batch_size) return;

			work_queue::force_processing();
		}

		void force_processing()
		{
			last_processing_ = std::async(std::launch::async, [&]() -> void {
				attached_pipeline_->template process(work_queue::begin(), work_queue::end());
			}).share();

			last_img_idx_ = pool_position_idx_;
			pool_position_idx_ = 0;
		}

		auto begin() -> typename image_pool_t::iterator
		{
			return image_pool_.begin();
		}

		auto end() -> typename image_pool_t::iterator
		{
			return image_pool_.begin()+last_img_idx_;
		}

		void wait()
		{
			if (last_processing_.valid()) last_processing_.wait();
		}

	private:
		image_pool_t image_pool_{};
		std::shared_future<void> last_processing_{};
		size_t pool_size_ = 0;
		size_t last_img_idx_ = 0;
		pipeline<batch_size>* attached_pipeline_ = nullptr;
		size_t pool_position_idx_ = 0;
	};
}

#endif //FDP_WORK_QUEUE_HPP
