#include <boost/log/trivial.hpp>

#include "eureka_client.hpp"

nntu::net::eureka_client::eureka_client()
		:eureka_path_("/eureka/apps/")
{
	auto instance = web::json::value::object();
	auto port = web::json::value::object();
	auto data_center_info = web::json::value::object();
	auto meta = web::json::value::object();

	port["$"] = web::json::value::string("8081");
	port["@enabled"] = web::json::value::boolean(true);

	data_center_info["name"] = web::json::value::string("MyOwn");
	data_center_info["@class"] = web::json::value::string(
			"com.netflix.appinfo.InstanceInfo$DefaultDataCenterInfo");

	meta["instanceId"] = web::json::value::string("nip-service:8081");

	instance["hostName"] = web::json::value::string("localhost");
	instance["app"] = web::json::value::string("nip-service");
	instance["ipAddr"] = web::json::value::string("127.0.0.1");
	instance["vipAddress"] = web::json::value::string("nip-service");
	instance["port"] = port;
	instance["dataCenterInfo"] = data_center_info;
	instance["metadata"] = meta;

	config_["instance"] = instance;
}

nntu::net::eureka_client::~eureka_client()
{
	continue_heart_beat_ = false;
	eureka_client::unregister();
}

void nntu::net::eureka_client::enroll(const char* url)
{
	eureka_url_ = url;

	web::http::client::http_client_config configs;
	web::uri_builder builder = eureka_path_;

	builder.append("nip-service");
	configs.set_validate_certificates(false);

	web::http::client::http_client(eureka_url_, configs)
			.request(web::http::methods::POST,
					builder.to_string(),
					config_.serialize(),
					"application/json")
			.then([=](web::http::http_response response) {
				if (response.status_code()!=204) {
					BOOST_LOG_TRIVIAL(error) << "Failed to register nip-service in Eureka with code: "
											 << response.status_code();
				}
				else {
					BOOST_LOG_TRIVIAL(info) << "Successfully registered in Eureka";
				}
			})
			.wait();

	eureka_client::run_heart_beating();
}

void nntu::net::eureka_client::run_heart_beating()
{
	continue_heart_beat_ = true;

	hear_beat_event_ = std::async(std::launch::async, [this]() {
		using namespace std::chrono_literals;
		while (continue_heart_beat_) {
			std::this_thread::sleep_for(5s);

			web::http::client::http_client_config configs;
			web::uri_builder builder = eureka_path_;

			builder.append("nip-service")
					.append_path_raw("localhost:nip-service:8081");
			configs.set_validate_certificates(false);

			web::http::client::http_client(eureka_url_, configs)
					.request(web::http::methods::PUT, builder.to_string())
					.then([=](web::http::http_response response) {
						if (response.status_code()!=200) {
							BOOST_LOG_TRIVIAL(error) << "Failed to heart-beat nip-service with code: "
													 << response.status_code();
						}
						else {
							BOOST_LOG_TRIVIAL(info) << "Heartbeat sent to Eureka";
						}
					})
					.wait();
		}
	}).share();
}

void nntu::net::eureka_client::unregister()
{
	web::http::client::http_client_config configs;
	web::uri_builder builder = eureka_path_;

	builder.append("nip-service")
			.append_path_raw("localhost:nip-service:8081");
	configs.set_validate_certificates(false);

	web::http::client::http_client(eureka_url_, configs)
			.request(web::http::methods::DEL, builder.to_string())
			.then([=](web::http::http_response response) {
				if (response.status_code()!=200) {
					BOOST_LOG_TRIVIAL(error) << "Failed to de-register nip-service with code: "
											 << response.status_code();
				}
				else {
					BOOST_LOG_TRIVIAL(info) << "Successfully de-registered in Eureka";
				}
			})
			.wait();
}
