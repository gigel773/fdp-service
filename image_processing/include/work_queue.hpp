#ifndef FDP_WORK_QUEUE_HPP
#define FDP_WORK_QUEUE_HPP

#include <opencv2/opencv.hpp>
#include <boost/log/trivial.hpp>

#include <vector>
#include <memory>
#include <thread>
#include <future>
#include <chrono>
#include <condition_variable>
#include <functional>

#include "pipeline.hpp"

namespace nntu::img {

	template<size_t batch_size>
	class work_queue {
		using image_handler_t = std::function<void(cv::Mat&&)>;

		struct daemon_meta {
			std::condition_variable cv{};
			std::mutex mutex{};
			std::atomic<bool> condition{ true };
		};

	public:
		explicit work_queue(size_t pool_size)
		{
			work_queue::run_flusher();
		}

		work_queue(work_queue& other) = delete;

		auto operator=(work_queue& other) -> work_queue& = delete;

		auto attach_to(pipeline<batch_size>& value)
		{
			attached_pipeline_ = &value;
		}

		void submit(cv::Mat&& img, image_handler_t handler)
		{
			work_queue::wait();
			std::lock_guard lock(submission_mutex_);

			image_pool_.push_back(std::move(img));
			image_handlers_.push_back(std::move(handler));

			if (image_pool_.size()<batch_size) return;

			work_queue::force_processing();
		}

		void wait()
		{
			if (last_processing_.valid()) last_processing_.wait();
		}

		virtual ~work_queue()
		{
			flushing_meta_.condition = false;
			flushing_meta_.cv.notify_all();

			if (last_processing_.valid()) last_processing_.wait();
			if (flusher_process_.valid()) flusher_process_.wait();
		}

	protected:
		void run_flusher()
		{
			flusher_process_ = std::async(std::launch::async, [this]() {
				while (flushing_meta_.condition) {
					using namespace std::chrono_literals;

					work_queue::wait();
					work_queue::force_processing();

					std::unique_lock<std::mutex> lock(flushing_meta_.mutex);
					flushing_meta_.cv.wait_for(lock, 10000ns);
				}
			}).share();
		}

		void force_processing()
		{
			if (attached_pipeline_==nullptr) return;
			if (image_pool_.empty()) return;

			last_processing_ = std::async(std::launch::async, [&]() -> void {
				auto begin = std::begin(image_pool_);
				auto end = image_pool_.size()>batch_size ? begin+batch_size : std::end(image_pool_);

				attached_pipeline_->template process(begin, end);

				for (size_t i = 0; i<std::distance(begin, end); i++) {
					image_handlers_[i](std::move(image_pool_[i]));
				}

				image_pool_.clear();
				image_handlers_.clear();

				image_pool_.reserve(batch_size);
				image_handlers_.reserve(batch_size);
			}).share();
		}

	private:
		std::vector<cv::Mat> image_pool_{};
		std::vector<image_handler_t> image_handlers_{};
		daemon_meta flushing_meta_{};
		std::shared_future<void> last_processing_{};
		std::shared_future<void> flusher_process_{};
		std::mutex submission_mutex_{};

		pipeline<batch_size>* attached_pipeline_ = nullptr;
	};
}

#endif //FDP_WORK_QUEUE_HPP
