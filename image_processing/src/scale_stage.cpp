#include "scale_stage.hpp"

namespace nntu::img {

	void scale_stage<scale_type::scale>::submit(cv::Mat* begin, cv::Mat* end)
	{
		for (auto* cur = begin; cur<end; cur++) {
			cv::resize(*cur, *cur, cv::Size(), target_factor_, target_factor_);
		}
	}

	void scale_stage<scale_type::resize>::submit(cv::Mat* begin, cv::Mat* end)
	{
		for (auto* cur = begin; cur<end; cur++) {
			const float factor = static_cast<float>(target_cols_)/static_cast<float>(cur->cols);

			cv::resize(*cur, *cur, cv::Size(), factor, factor);
		}
	}

	void scale_stage<scale_type::resize>::wait()
	{
		// No async yet
	}

	void scale_stage<scale_type::scale>::wait()
	{
		// No async yet
	}
}