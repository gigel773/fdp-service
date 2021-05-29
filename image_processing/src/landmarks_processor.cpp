#include "landmarks_processor.hpp"

namespace nntu::img::detail {

	static inline void fill_blob(const cv::Mat& orig_image, InferenceEngine::Blob::Ptr& blob, size_t batchIndex)
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

		size_t batchOffset = batchIndex*width*height*channels;

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

nntu::img::landmarks_processor::landmarks_processor(size_t batch_size)
		:wq_size_(batch_size)
{
	const std::map<std::string, std::string> dyn_config =
			{{ InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED, InferenceEngine::PluginConfigParams::YES }};

	network_ = core_.ReadNetwork(cnn_path, cnn_weights_path);
	network_.setBatchSize(batch_size);
	input_info_ = network_.getInputsInfo();

	auto input = input_info_.begin();
	input->second->setPrecision(InferenceEngine::Precision::U8);

	executable_network_ = core_.LoadNetwork(network_, "CPU", dyn_config);
	request_ = executable_network_.CreateInferRequest();
}

void nntu::img::landmarks_processor::submit(cv::Mat* begin, cv::Mat* end)
{
	begin_ = begin;
	end_ = end;

	auto input = input_info_.begin();
	auto blob_ptr = request_.GetBlob(input->first);
	size_t image_count = 0;

	for (auto* cur = begin_; cur<end_; cur++) {
		detail::fill_blob(*cur, blob_ptr, image_count);
		++image_count;
	}

	request_.SetBatch(image_count);
	request_.StartAsync();
}

void nntu::img::landmarks_processor::wait()
{
	request_.Wait(100);

	auto landmarks_blob = request_.GetBlob(output_layer_name);

	InferenceEngine::LockedMemory<const void> landmarks_blob_mapped =
			InferenceEngine::as<InferenceEngine::MemoryBlob>(request_.GetBlob(output_layer_name))->rmap();

	auto* cur_img = begin_;

	for (size_t batch_idx = 0; cur_img<end_; batch_idx++, cur_img++) {
		auto& img = *cur_img;
		const float* coordinates_ptr = landmarks_blob_mapped.as<float*>()+(coordinates_pair_count*batch_idx);

		std::vector<float> coords(coordinates_ptr, coordinates_ptr+coordinates_pair_count);
		std::vector<cv::Point> face_boundary;
		face_boundary.reserve(22);

		int max_y = std::numeric_limits<int>::min();
		int max_x = std::numeric_limits<int>::min();
		int min_y = std::numeric_limits<int>::max();
		int min_x = std::numeric_limits<int>::max();

		for (int i = 12; i<34; i++) {
			int x = static_cast<int>(img.size().width*coordinates_ptr[i*2]);
			int y = static_cast<int>(img.size().height*coordinates_ptr[i*2+1]);

			max_x = std::max(max_x, x);
			max_y = std::max(max_y, y);
			min_x = std::min(min_x, x);
			min_y = std::min(min_y, y);

			face_boundary.emplace_back(x, y);
		}

		cv::Rect required_image(min_x, min_y, max_x-min_x, max_y-min_y);

		// Cut the face
		auto mask = cv::Mat(img.rows, img.cols, CV_8UC3, cv::Scalar(0, 0, 0));
		auto resulted_face = cv::Mat();
		auto ordered_landmarks = std::vector<cv::Point>();

		cv::convexHull(face_boundary, ordered_landmarks);
		cv::fillConvexPoly(mask, ordered_landmarks, cv::Scalar(255, 255, 255));

		img.copyTo(resulted_face, mask);
		*cur_img = resulted_face(required_image);
	}
}

//auto nntu::img::landmarks_processor::get_result(const std::vector<cv::Mat>& input) -> std::vector<cv::Mat>
//{
//	std::vector<cv::Mat> results;
//
//	auto landmarks_blob = request_.GetBlob(output_layer_name);
//
//	InferenceEngine::LockedMemory<const void> landmarks_blob_mapped =
//			InferenceEngine::as<InferenceEngine::MemoryBlob>(request_.GetBlob(output_layer_name))->rmap();
//
//	for (size_t batch_idx = 0; batch_idx<image_count_; batch_idx++) {
//		auto& img = input[batch_idx];
//		const float* coordinates_ptr = landmarks_blob_mapped.as<float*>()+(coordinates_pair_count*batch_idx);
//
//		std::vector<float> coords(coordinates_ptr, coordinates_ptr+coordinates_pair_count);
//		std::vector<cv::Point> face_boundary;
//		face_boundary.reserve(22);
//
//		int max_y = std::numeric_limits<int>::min();
//		int max_x = std::numeric_limits<int>::min();
//		int min_y = std::numeric_limits<int>::max();
//		int min_x = std::numeric_limits<int>::max();
//
//		for (int i = 12; i<34; i++) {
//			int x = static_cast<int>(img.size().width*coordinates_ptr[i*2]);
//			int y = static_cast<int>(img.size().height*coordinates_ptr[i*2+1]);
//
//			max_x = std::max(max_x, x);
//			max_y = std::max(max_y, y);
//			min_x = std::min(min_x, x);
//			min_y = std::min(min_y, y);
//
//			face_boundary.emplace_back(x, y);
//		}
//
//		cv::Rect required_image(min_x, min_y, max_x-min_x, max_y-min_y);
//
//		// Cut the face
//		auto mask = cv::Mat(img.rows, img.cols, CV_8UC3, cv::Scalar(0, 0, 0));
//		auto resulted_face = cv::Mat();
//		auto ordered_landmarks = std::vector<cv::Point>();
//
//		cv::convexHull(face_boundary, ordered_landmarks);
//		cv::fillConvexPoly(mask, ordered_landmarks, cv::Scalar(255, 255, 255));
//
//		img.copyTo(resulted_face, mask);
//		results.push_back(resulted_face(required_image));
//	}
//
//	image_count_ = 0;
//
//	return results;
//}
