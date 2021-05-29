#include <benchmark/benchmark.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <filesystem>

#include <work_queue.hpp>

constexpr const char face_path[] = R"(../../photos/)";

static inline auto load_images(const char* path, size_t image_count) -> std::vector<cv::Mat>
{
	std::vector<cv::Mat> result;

	while (result.size()<image_count) {
		for (const auto& it: std::filesystem::directory_iterator(path)) {
			result.push_back(cv::imread(it.path().c_str()));

			if (result.size()>=image_count) {
				break;
			}
		}
	}

	return result;
}

template<size_t batch_size>
void BM_impl(benchmark::State& state)
{
	auto work_queue = nntu::img::work_queue<batch_size>(1);
	auto pipeline = nntu::img::default_pipeline_impl<batch_size>(120);

	work_queue.attach_to(pipeline);

	for (auto _: state) {
		state.PauseTiming();

		auto images = load_images(face_path, batch_size-1);

		for (auto& img :images) {
			work_queue.submit(std::move(img));
		}

		state.ResumeTiming();

		work_queue.force_processing();
		work_queue.wait();
	}

	size_t total_size = 0;
	for (const auto& img : load_images(face_path, batch_size-1)) {
		total_size += img.total()*img.elemSize();
	}

	state.SetBytesProcessed(total_size*state.iterations());
}

void BM_default_pipeline_64_test(benchmark::State& state)
{
	constexpr const size_t batch_size = 64;

	BM_impl<batch_size>(state);
}

void BM_default_pipeline_128_test(benchmark::State& state)
{
	constexpr const size_t batch_size = 128;

	BM_impl<batch_size>(state);
}

BENCHMARK(BM_default_pipeline_64_test)
		->Unit(benchmark::kMicrosecond)
		->MeasureProcessCPUTime()
		->UseRealTime();

BENCHMARK(BM_default_pipeline_128_test)
		->Unit(benchmark::kMicrosecond)
		->MeasureProcessCPUTime()
		->UseRealTime();
