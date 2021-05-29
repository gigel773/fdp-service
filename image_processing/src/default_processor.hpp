#ifndef FDP_DEFAULT_PROCESSOR_HPP
#define FDP_DEFAULT_PROCESSOR_HPP

#include <inference_engine.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <common_processor.hpp>

namespace nntu::img {

	class default_processor : public queue {
		static constexpr const char cnn_path[] = "../models/facial-landmarks-35-adas-0002.xml";
		static constexpr const char cnn_weights_path[] = "../models/facial-landmarks-35-adas-0002.bin";
		static constexpr const char output_layer_name[] = "align_fc3";

	public:
		default_processor();

		void set_queue_size(size_t value) override;

		void submit(const cv::Mat& frame) override;

		auto get_result(const cv::Mat& input) -> cv::Mat override;

	private:
		InferenceEngine::Core core_{};
		InferenceEngine::CNNNetwork network_;
		InferenceEngine::ExecutableNetwork executable_network_;
		InferenceEngine::InferRequest request_;
		InferenceEngine::InputsDataMap input_info_;
	};

}
#endif //FDP_DEFAULT_PROCESSOR_HPP
