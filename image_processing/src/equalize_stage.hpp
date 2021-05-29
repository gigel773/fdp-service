#ifndef FDP_EQUALIZE_STAGE_HPP
#define FDP_EQUALIZE_STAGE_HPP

#include <stage.hpp>

namespace nntu::img {

	class equalize_stage final : public stage {
	public:
		void submit(cv::Mat* begin, cv::Mat* end) override;

		void wait() override;
	};

}
#endif //FDP_EQUALIZE_STAGE_HPP
