#include <benchmark/benchmark.h>
#include <vector>
#include <filesystem>

#include <work_queue.hpp>

#include "common/util.hpp"

constexpr const char face_path[] = R"(../../photos/)";

template<size_t batch_size>
static inline void BM_impl(benchmark::State& state)
{
    auto work_queue = nntu::img::work_queue<batch_size>(1);
    auto pipeline = nntu::img::gpu_pipeline_impl<batch_size>(120);

    work_queue.attach_to(pipeline);

    for (auto _: state) {
        state.PauseTiming();
        auto images = nntu::test::load_images(face_path, batch_size);
        std::atomic<size_t> images_processed = 0;
        size_t images_submitted = 0;

        state.ResumeTiming();

        for (auto& img :images) {
            work_queue.submit(std::move(img), [&images_processed](cv::Mat&& img) {
                ++images_processed;
            });

            ++images_submitted;
        }

        while (images_submitted!=images_processed) {
            work_queue.wait();
        }
    }

    size_t total_size = 0;
    for (const auto& img : nntu::test::load_images(face_path, batch_size)) {
        total_size += img.total()*img.elemSize();
    }

    state.SetBytesProcessed(total_size*state.iterations());
    state.SetItemsProcessed(batch_size);
}

void BM_gpu_pipeline_64_test(benchmark::State& state)
{
    constexpr const size_t batch_size = 64;

    BM_impl<batch_size>(state);
}

void BM_gpu_pipeline_128_test(benchmark::State& state)
{
    constexpr const size_t batch_size = 128;

    BM_impl<batch_size>(state);
}

void BM_gpu_pipeline_512_test(benchmark::State& state)
{
    constexpr const size_t batch_size = 512;

    BM_impl<batch_size>(state);
}

BENCHMARK(BM_gpu_pipeline_64_test)
        ->Unit(benchmark::kMicrosecond)
        ->MeasureProcessCPUTime()
        ->UseRealTime();

BENCHMARK(BM_gpu_pipeline_128_test)
        ->Unit(benchmark::kMicrosecond)
        ->MeasureProcessCPUTime()
        ->UseRealTime();

BENCHMARK(BM_gpu_pipeline_512_test)
        ->Unit(benchmark::kMicrosecond)
        ->MeasureProcessCPUTime()
        ->UseRealTime();
