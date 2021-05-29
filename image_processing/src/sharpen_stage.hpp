#ifndef FDP_SHARPEN_STAGE_HPP
#define FDP_SHARPEN_STAGE_HPP

#include <stage.hpp>

namespace nntu::img {

	class sharpen_stage final : public stage {
	public:
		sharpen_stage();

		void submit(cv::Mat* begin, cv::Mat* end) override;

		void wait() override;

	private:
		cv::Mat kernel_;
	};

}
#endif //FDP_SHARPEN_STAGE_HPP
