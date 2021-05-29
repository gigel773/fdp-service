#ifndef FDP_IMAGE_PROCESSOR_HPP
#define FDP_IMAGE_PROCESSOR_HPP

#include <cpprest/http_listener.h>
#include <work_queue.hpp>

namespace nntu::net {

	class image_processor final {
		static constexpr const size_t batch_size = 128;
	public:
		image_processor();

		void process_image(web::http::http_request request);

	private:
		nntu::img::work_queue<batch_size> work_queue_;
		nntu::img::pipeline<batch_size> default_pipeline_;
	};

}
#endif //FDP_IMAGE_PROCESSOR_HPP
