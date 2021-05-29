#include <nips.hpp>
#include <cpprest/http_listener.h>
#include <opencv2/opencv.hpp>

#include "eureka_client.hpp"
#include "image_processor.hpp"

void nntu::net::run_server(const char* server_path, const char* eureka_url)
{
	auto client = eureka_client();
	auto processor = image_processor();
	auto listener = web::http::experimental::listener::http_listener(std::string(server_path)+"/process");

	listener.support(web::http::methods::POST,
			[&processor](auto&& request) {
				processor.process_image(std::forward<decltype(request)>(request));
			});

	client.enroll(eureka_url);

	std::string line;
	std::getline(std::cin, line);
}
