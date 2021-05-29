#include "equalize_stage.hpp"

void nntu::img::equalize_stage::submit(cv::Mat* begin, cv::Mat* end)
{
	for (auto* cur = begin; cur<end; cur++) {
		cv::Mat result;

		cv::cvtColor(*cur, *cur, cv::COLOR_BGR2GRAY);
		cv::equalizeHist(*cur, result);

		*cur = result;
	}
}

void nntu::img::equalize_stage::wait()
{
	// No async yet
}
