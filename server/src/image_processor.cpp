#include "image_processor.hpp"

nntu::net::image_processor::image_processor()
		:default_pipeline_(nntu::img::default_pipeline_impl<batch_size>(120)),
		 work_queue_(1)
{
	work_queue_.attach_to(default_pipeline_);
}

void nntu::net::image_processor::process_image(web::http::http_request request)
{
	request.extract_json()
			.then([=](web::json::value json) {
				auto encoded_img = json["array"].as_array()[0].as_string();
				auto decoded_img = cv::imdecode(utility::conversions::from_base64(encoded_img),
						CV_LOAD_IMAGE_COLOR);

				work_queue_.submit(std::move(decoded_img), [request](cv::Mat&& img) {
					request.reply(web::http::status_codes::OK);
				});
			})
			.then([=]() {
				request.reply(web::http::status_codes::OK);
			});
}
