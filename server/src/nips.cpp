#include <nips.hpp>

#include "eureka_client.hpp"

void nntu::net::run_server(const char* eureka_url)
{
	auto client = eureka_client();

	client.enroll(eureka_url);

	std::string line;
	std::getline(std::cin, line);
}
