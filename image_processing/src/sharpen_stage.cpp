#include "sharpen_stage.hpp"

nntu::img::sharpen_stage::sharpen_stage()
{
	kernel_ = cv::Mat(3, 3, CV_32S);

	for (int i = 0; i<3; i++) {
		for (int j = 0; j<3; j++) {
			kernel_.at<int>(i, j) = -1;
		}
	}

	kernel_.at<int>(1, 1) = 9;
}

void nntu::img::sharpen_stage::submit(cv::Mat* begin, cv::Mat* end)
{
	for (auto* cur = begin; cur<end; cur++) {
		cv::filter2D(*cur, *cur, -1, kernel_);
	}
}

void nntu::img::sharpen_stage::wait()
{
	// No async yet
}
