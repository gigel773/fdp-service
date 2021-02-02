#include <iostream>
#include <inference_engine.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

constexpr const char cnn_path[] = "../models/facial-landmarks-35-adas-0002.xml";
constexpr const char cnn_weights_path[] = "../models/facial-landmarks-35-adas-0002.bin";
constexpr const char face_path[] = "../photos/face.jpg";
constexpr const char haar_cascade_face_path[] = "../models/haarcascade_frontalface_default.xml";
constexpr const char output_layer_name[] = "align_fc3";

template<typename T>
void matU8ToBlob(const cv::Mat& orig_image, InferenceEngine::Blob::Ptr& blob, int batchIndex = 0);

int main()
{
	// Cut the face
	auto face_classifier = cv::CascadeClassifier(haar_cascade_face_path);
	auto frame = cv::imread(face_path);
	auto frame_gray = cv::Mat();
	auto faces = std::vector<cv::Rect>();

	cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
	cv::equalizeHist(frame_gray, frame_gray);

	face_classifier.detectMultiScale(frame_gray, faces);

	if (faces.empty()) return 1;

	for (auto& face_coordinates: faces) {
		if (face_coordinates.height<60 && face_coordinates.width<60) continue;

		auto face_frame = frame(face_coordinates);

		// Find landmarks
		InferenceEngine::Core core{};
		InferenceEngine::CNNNetwork network = core.ReadNetwork(cnn_path, cnn_weights_path);

		auto input_info = network.getInputsInfo();
		auto input = input_info.begin();

		input->second->setPrecision(InferenceEngine::Precision::U8);

		InferenceEngine::ExecutableNetwork executable_network = core.LoadNetwork(network, "CPU");
		InferenceEngine::InferRequest request = executable_network.CreateInferRequest();

		auto blob_ptr = request.GetBlob(input->first);
		matU8ToBlob<uint8_t>(face_frame, blob_ptr);

		request.Infer();

		auto landmarks_blob = request.GetBlob(output_layer_name);

		InferenceEngine::LockedMemory<const void> landmarks_blob_mapped =
				InferenceEngine::as<InferenceEngine::MemoryBlob>(request.GetBlob(output_layer_name))->rmap();
		const float* coordinates_ptr = landmarks_blob_mapped.as<float*>();

		std::vector<cv::Point> face_boundary;
		face_boundary.reserve(22);

		int max_y = std::numeric_limits<int>::min();
		int max_x = std::numeric_limits<int>::min();
		int min_y = std::numeric_limits<int>::max();
		int min_x = std::numeric_limits<int>::max();

		for (int i = 12; i<34; i++) {
			int x = static_cast<int>(face_frame.size().width*coordinates_ptr[i*2]);
			int y = static_cast<int>(face_frame.size().height*coordinates_ptr[i*2+1]);

			max_x = std::max(max_x, x);
			max_y = std::max(max_y, y);
			min_x = std::min(min_x, x);
			min_y = std::min(min_y, y);

			face_boundary.emplace_back(x, y);
		}

		cv::Rect required_image(min_x, min_y, max_x - min_x, max_y - min_y);

		// Cut the face
		auto mask = cv::Mat(face_frame.rows, face_frame.cols, CV_8UC3, cv::Scalar(0, 0, 0));
		auto resulted_face = cv::Mat();
		auto ordered_landmarks = std::vector<cv::Point>();

		cv::convexHull(face_boundary, ordered_landmarks);
		cv::fillConvexPoly(mask, ordered_landmarks, cv::Scalar(255, 255, 255));

		face_frame.copyTo(resulted_face, mask);

		cv::imshow("Face", resulted_face(required_image));
		cv::waitKey(0);
	}

	return 0;
}

template<typename T>
void matU8ToBlob(const cv::Mat& orig_image, InferenceEngine::Blob::Ptr& blob, int batchIndex)
{
	InferenceEngine::SizeVector blobSize = blob->getTensorDesc().getDims();
	const size_t width = blobSize[3];
	const size_t height = blobSize[2];
	const size_t channels = blobSize[1];

	if (static_cast<size_t>(orig_image.channels())!=channels) {
		THROW_IE_EXCEPTION << "The number of channels for net input and image must match";
	}
	InferenceEngine::LockedMemory<void> blobMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob)->wmap();
	T* blob_data = blobMapped.as<T*>();

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
