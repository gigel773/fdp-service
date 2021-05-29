#include "pipeline.hpp"
#include "landmarks_stage.hpp"
#include "facial_cut_stage.hpp"
#include "scale_stage.hpp"
#include "sharpen_stage.hpp"

namespace nntu::img {
	template<size_t batch_size>
	auto default_pipeline_impl() -> pipeline<batch_size>
	{
		pipeline<batch_size> result({
				new facial_cut_stage(batch_size),
				new scale_stage<scale_type::resize>(256),
				new scale_stage<scale_type::scale>(2.0f),
				new sharpen_stage(),
				new landmarks_stage(batch_size)
		});

		return result;
	}

	template
	auto default_pipeline_impl() -> pipeline<64>;

	template
	auto default_pipeline_impl() -> pipeline<128>;
}
