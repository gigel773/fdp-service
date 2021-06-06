#include <benchmark/benchmark.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <filesystem>

#include <work_queue.hpp>
#include "common/util.hpp"

constexpr const char face_path[] = R"(../../photos/)";

template<size_t cpu_batch_size, size_t gpu_batch_size>
static inline void BM_impl(benchmark::State& state)
{
    auto work_queue_cpu = nntu::img::work_queue<cpu_batch_size>(1);
    auto work_queue_gpu = nntu::img::work_queue<gpu_batch_size>(1);

    auto cpu_pipeline = nntu::img::default_pipeline_impl<cpu_batch_size>(120);
    auto gpu_pipeline = nntu::img::gpu_pipeline_impl<gpu_batch_size>(120);

    work_queue_cpu.attach_to(cpu_pipeline);
    work_queue_gpu.attach_to(gpu_pipeline);

    for (auto _: state) {
        state.PauseTiming();

        std::atomic<size_t> images_processed = 0;
        size_t images_submitted = 0;

        auto cpu_images = nntu::test::load_images(face_path, cpu_batch_size);
        auto gpu_images = nntu::test::load_images(face_path, gpu_batch_size);

        state.ResumeTiming();

        for (auto& img :gpu_images) {
            work_queue_gpu.submit(std::move(img), [&images_processed](cv::Mat&& img) {
                ++images_processed;
            });

            ++images_submitted;
        }

        for (auto& img :cpu_images) {
            work_queue_cpu.submit(std::move(img), [&images_processed](cv::Mat&& img) {
                ++images_processed;
            });

            ++images_submitted;
        }

        while (images_submitted!=images_processed) {
            work_queue_gpu.wait();
            work_queue_cpu.wait();
        }
    }

    uint64_t total_size = 0;
    for (const auto& img : nntu::test::load_images(face_path, cpu_batch_size)) {
        total_size += img.total()*img.elemSize();
    }

    for (const auto& img : nntu::test::load_images(face_path, gpu_batch_size)) {
        total_size += img.total()*img.elemSize();
    }

    state.SetBytesProcessed(total_size*state.iterations());
    state.SetItemsProcessed(cpu_batch_size+gpu_batch_size);
}

void BM_combined_pipeline_64_512_test(benchmark::State& state)
{
    constexpr const size_t cpu_batch_size = 64;
    constexpr const size_t gpu_batch_size = 512;

    BM_impl<cpu_batch_size, gpu_batch_size>(state);
}

BENCHMARK(BM_combined_pipeline_64_512_test)
        ->Unit(benchmark::kMicrosecond)
        ->MeasureProcessCPUTime()
        ->UseRealTime();
