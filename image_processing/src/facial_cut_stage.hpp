
#ifndef FDP_FACIAL_CUT_STAGE_HPP
#define FDP_FACIAL_CUT_STAGE_HPP

#include <atomic>
#include <opencv2/opencv.hpp>

#include <stage.hpp>

#include "defs.hpp"

namespace nntu::img {

	class facial_cut_stage final : public stage {
		static constexpr const char haar_cascade_face_path[] = R"(haarcascade_frontalface_default.xml)";

	public:
		explicit facial_cut_stage(size_t batch_size);

		void submit(cv::Mat* begin, cv::Mat* end) override;

		void wait() override;

	private:
		cv::CascadeClassifier classifier_;
	};

}
#endif //FDP_FACIAL_CUT_STAGE_HPP
