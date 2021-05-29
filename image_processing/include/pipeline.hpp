#ifndef FDP_PIPELINE_HPP
#define FDP_PIPELINE_HPP

#include <opencv2/opencv.hpp>
#include <vector>

#include "stage.hpp"

namespace nntu::img {

	template<size_t batch_size>
	class pipeline {
	public:
		pipeline(std::initializer_list<stage*> stages)
		{
			for (auto& ptr : stages) {
				stages_.emplace_back(ptr);
			}
		}

		template<class iterator_t>
		void process(iterator_t begin, iterator_t end)
		{
			static_assert(std::is_same<typename std::iterator_traits<iterator_t>::value_type, cv::Mat>::value,
					"Incorrect iterator type");
			static_assert(std::is_same<typename std::iterator_traits<iterator_t>::iterator_category,
							std::random_access_iterator_tag>::value,
					"Incorrect iterator type");

			if (std::distance(begin, end)>batch_size) {
				throw std::runtime_error("Incorrect number of images to process");
			}

			auto first_stage = stages_.begin();
			auto second_stage = first_stage+1;

			(*first_stage)->submit(&*begin, &*end);

			while (second_stage<stages_.end()) {
				(*first_stage)->wait();
				(*second_stage)->submit(&*begin, &*end);

				first_stage = second_stage;
				++second_stage;
			}

			(*first_stage)->wait();
		}

	private:
		using stages_pool_t = std::vector<std::unique_ptr<stage>>;

		stages_pool_t stages_{};
	};

	template<size_t batch_size>
	auto default_pipeline_impl(size_t required_size) -> pipeline<batch_size>;
}

#endif //FDP_PIPELINE_HPP
