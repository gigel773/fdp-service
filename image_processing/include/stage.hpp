#ifndef FDP_STAGE_HPP
#define FDP_STAGE_HPP

//#define DEBUG_STAGES
#ifdef DEBUG_STAGES

#include <iostream>

#define LOG() std::cout
#else
#define LOG()
#endif

#include <opencv2/opencv.hpp>

namespace nntu::img {

	class stage {
	public:
		virtual void submit(cv::Mat* begin, cv::Mat* end) = 0;

		virtual void wait() = 0;

		virtual ~stage() = default;
	};
}

#endif //FDP_STAGE_HPP
