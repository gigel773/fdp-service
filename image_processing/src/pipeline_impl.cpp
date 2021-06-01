#include "pipeline.hpp"
#include "landmarks_stage.hpp"
#include "facial_cut_stage.hpp"
#include "scale_stage.hpp"
#include "sharpen_stage.hpp"
#include "blur_stage.hpp"
#include "equalize_stage.hpp"
#include "preprocess_gpu_stage.hpp"
#include "postprocess_gpu_stage.hpp"

#include <tbb/tbb.h>

namespace nntu::img {
    template<size_t batch_size>
    auto default_pipeline_impl(size_t required_size) -> pipeline<batch_size>
    {
#ifdef WITH_TBB
        tbb::task_scheduler_init init(tbb::task_scheduler_init::automatic);
#endif
        pipeline<batch_size> result({
                new scale_stage<scale_type::scale>(5.0f),
                new facial_cut_stage(batch_size),
                new scale_stage<scale_type::resize>(128),
                new sharpen_stage(),
                new blur_stage(),
                new landmarks_stage(batch_size),
                new equalize_stage(),
                new scale_stage<scale_type::resize>(required_size)
        });

        return result;
    }

    template<size_t batch_size>
    auto gpu_pipeline_impl(size_t required_size) -> pipeline<batch_size>
    {
        pipeline<batch_size> result({
                new preprocess_gpu_stage(),
                new landmarks_stage(batch_size),
                new postprocess_gpu_stage(required_size)
        });

        return result;
    }

    template
    auto default_pipeline_impl(size_t required_size) -> pipeline<64>;

    template
    auto default_pipeline_impl(size_t required_size) -> pipeline<128>;

    template
    auto default_pipeline_impl(size_t required_size) -> pipeline<512>;

    template
    auto gpu_pipeline_impl(size_t required_size) -> pipeline<64>;

    template
    auto gpu_pipeline_impl(size_t required_size) -> pipeline<128>;

    template
    auto gpu_pipeline_impl(size_t required_size) -> pipeline<512>;
}
