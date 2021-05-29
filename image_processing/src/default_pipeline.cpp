#include "pipeline.hpp"
#include "landmarks_processor.hpp"
#include "facial_preprocessor.hpp"

namespace nntu::img {
	template<size_t batch_size>
	auto default_pipeline_impl() -> pipeline<batch_size>
	{
		pipeline<batch_size> result({
				new facial_preprocessor(batch_size),
				new landmarks_processor(batch_size)
		});

		return result;
	}

	template
	auto default_pipeline_impl() -> pipeline<64>;

	template
	auto default_pipeline_impl() -> pipeline<128>;
}
