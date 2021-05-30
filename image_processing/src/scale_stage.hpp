#ifndef FDP_SCALE_STAGE_HPP
#define FDP_SCALE_STAGE_HPP

#include <stage.hpp>

#include "defs.hpp"

namespace nntu::img {

	enum class scale_type {
		resize,
		scale
	};

	template<scale_type scale_t>
	class scale_stage;

	template<>
	class scale_stage<scale_type::resize> : public stage {
	public:
		scale_stage(int required_size)
				:target_cols_(required_size),
				 target_rows_(required_size)
		{
		}

		void submit(cv::Mat* begin, cv::Mat* end) override;

		void wait() override;

	private:
		int target_cols_ = 0;
		int target_rows_ = 0;
	};

	template<>
	class scale_stage<scale_type::scale> : public stage {
	public:
		explicit scale_stage(float factor)
				:target_factor_(factor)
		{
		}

		void submit(cv::Mat* begin, cv::Mat* end) override;

		void wait() override;

	private:
		float target_factor_ = 0.0f;
	};
}
#endif //FDP_SCALE_STAGE_HPP
