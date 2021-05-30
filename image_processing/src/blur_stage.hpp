#ifndef FDP_BLUR_STAGE_HPP
#define FDP_BLUR_STAGE_HPP

#include <stage.hpp>

#include "defs.hpp"

namespace nntu::img {

	class blur_stage final : public stage {
	public:
		void submit(cv::Mat* begin, cv::Mat* end) override;

		void wait() override;
	};

}
#endif //FDP_BLUR_STAGE_HPP
