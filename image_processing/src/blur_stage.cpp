#include "blur_stage.hpp"

void nntu::img::blur_stage::submit(cv::Mat* begin, cv::Mat* end)
{
	for (auto* cur = begin; cur<end; cur++) {
		cv::Mat result;

		cv::medianBlur(*cur, *cur, 3);
		cv::bilateralFilter(*cur, result, 5, 50, 50);

		*cur = result;
	}
}

void nntu::img::blur_stage::wait()
{
	// No async yet
}
