#ifndef FDP_LANDMARKS_PROCESSOR_HPP
#define FDP_LANDMARKS_PROCESSOR_HPP

#include <inference_engine.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <atomic>

#include <stage.hpp>

namespace nntu::img {

	class landmarks_processor : public stage {
		static constexpr const char cnn_path[] = "../models/facial-landmarks-35-adas-0002.xml";
		static constexpr const char cnn_weights_path[] = "../models/facial-landmarks-35-adas-0002.bin";
		static constexpr const char output_layer_name[] = "align_fc3";
		static constexpr const size_t coordinates_pair_count = 68;

	public:
		explicit landmarks_processor(size_t batch_size);

		void submit(cv::Mat* begin, cv::Mat* end) override;

		void wait() override;

		~landmarks_processor() override = default;

	private:
		InferenceEngine::Core core_{};
		InferenceEngine::CNNNetwork network_;
		InferenceEngine::ExecutableNetwork executable_network_;
		InferenceEngine::InferRequest request_;
		InferenceEngine::InputsDataMap input_info_;

		const size_t wq_size_;
		std::atomic<size_t> image_count_ = 0;
		cv::Mat* begin_ = nullptr;
		cv::Mat* end_ = nullptr;
	};

}
#endif //FDP_LANDMARKS_PROCESSOR_HPP
