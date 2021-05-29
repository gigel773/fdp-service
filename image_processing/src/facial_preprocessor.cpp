#include "facial_preprocessor.hpp"

nntu::img::facial_preprocessor::facial_preprocessor(size_t batch_size)
{
}

void nntu::img::facial_preprocessor::submit(cv::Mat* begin, cv::Mat* end)
{
	for (auto* cur = begin; cur<end; cur++) {
		*cur = *cur;
	}
}

void nntu::img::facial_preprocessor::wait()
{
	// No async yet
}
