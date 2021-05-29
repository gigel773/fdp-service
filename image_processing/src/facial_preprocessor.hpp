
#ifndef FDP_FACIAL_PREPROCESSOR_HPP
#define FDP_FACIAL_PREPROCESSOR_HPP

#include <atomic>
#include <opencv2/opencv.hpp>

#include <stage.hpp>

namespace nntu::img {

	class facial_preprocessor final : public stage {
	public:
		explicit facial_preprocessor(size_t batch_size);

		void submit(cv::Mat* begin, cv::Mat* end) override;

		void wait() override;
	};

}
#endif //FDP_FACIAL_PREPROCESSOR_HPP
