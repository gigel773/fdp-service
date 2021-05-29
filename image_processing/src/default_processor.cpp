#include "default_processor.hpp"

namespace nntu::img::detail {

	void fill_blob(const cv::Mat& orig_image, InferenceEngine::Blob::Ptr& blob, uint32_t batchIndex)
	{
		InferenceEngine::SizeVector blobSize = blob->getTensorDesc().getDims();
		const size_t width = blobSize[3];
		const size_t height = blobSize[2];
		const size_t channels = blobSize[1];

		if (static_cast<size_t>(orig_image.channels())!=channels) {
			THROW_IE_EXCEPTION << "The number of channels for net input and image must match";
		}
		InferenceEngine::LockedMemory<void> blobMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob)->wmap();
		auto* blob_data = blobMapped.as<uint8_t*>();

		cv::Mat resized_image(orig_image);
		if (static_cast<int>(width)!=orig_image.size().width ||
				static_cast<int>(height)!=orig_image.size().height) {
			cv::resize(orig_image, resized_image, cv::Size(width, height));
		}

		int batchOffset = batchIndex*width*height*channels;

		if (channels==1) {
			for (size_t h = 0; h<height; h++) {
				for (size_t w = 0; w<width; w++) {
					blob_data[batchOffset+h*width+w] = resized_image.at<uchar>(h, w);
				}
			}
		}
		else if (channels==3) {
			for (size_t c = 0; c<channels; c++) {
				for (size_t h = 0; h<height; h++) {
					for (size_t w = 0; w<width; w++) {
						blob_data[batchOffset+c*width*height+h*width+w] =
								resized_image.at<cv::Vec3b>(h, w)[c];
					}
				}
			}
		}
		else {
			THROW_IE_EXCEPTION << "Unsupported number of channels";
		}
	}
}

nntu::img::default_processor::default_processor()
{
	network_ = core_.ReadNetwork(cnn_path, cnn_weights_path);
	input_info_ = network_.getInputsInfo();

	auto input = input_info_.begin();
	input->second->setPrecision(InferenceEngine::Precision::U8);

	executable_network_ = core_.LoadNetwork(network_, "CPU");
	request_ = executable_network_.CreateInferRequest();
}

void nntu::img::default_processor::set_queue_size(size_t value)
{

}

void nntu::img::default_processor::submit(const cv::Mat& frame)
{
	auto input = input_info_.begin();
	auto blob_ptr = request_.GetBlob(input->first);

	detail::fill_blob(frame, blob_ptr, 0);

	request_.Infer();
}

auto nntu::img::default_processor::get_result(const cv::Mat& input) -> cv::Mat
{
	request_.Wait(1000);

	auto landmarks_blob = request_.GetBlob(output_layer_name);

	InferenceEngine::LockedMemory<const void> landmarks_blob_mapped =
			InferenceEngine::as<InferenceEngine::MemoryBlob>(request_.GetBlob(output_layer_name))->rmap();
	const float* coordinates_ptr = landmarks_blob_mapped.as<float*>();

	std::vector<cv::Point> face_boundary;
	face_boundary.reserve(22);

	int max_y = std::numeric_limits<int>::min();
	int max_x = std::numeric_limits<int>::min();
	int min_y = std::numeric_limits<int>::max();
	int min_x = std::numeric_limits<int>::max();

	for (int i = 12; i<34; i++) {
		int x = static_cast<int>(input.size().width*coordinates_ptr[i*2]);
		int y = static_cast<int>(input.size().height*coordinates_ptr[i*2+1]);

		max_x = std::max(max_x, x);
		max_y = std::max(max_y, y);
		min_x = std::min(min_x, x);
		min_y = std::min(min_y, y);

		face_boundary.emplace_back(x, y);
	}

	cv::Rect required_image(min_x, min_y, max_x-min_x, max_y-min_y);

	// Cut the face
	auto mask = cv::Mat(input.rows, input.cols, CV_8UC3, cv::Scalar(0, 0, 0));
	auto resulted_face = cv::Mat();
	auto ordered_landmarks = std::vector<cv::Point>();

	cv::convexHull(face_boundary, ordered_landmarks);
	cv::fillConvexPoly(mask, ordered_landmarks, cv::Scalar(255, 255, 255));

	input.copyTo(resulted_face, mask);
	return resulted_face(required_image);
}

auto nntu::img::queue::default_impl() -> std::shared_ptr<queue>
{
	return std::make_shared<default_processor>();
}
