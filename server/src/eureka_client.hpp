#ifndef FDP_EUREKA_CLIENT_HPP
#define FDP_EUREKA_CLIENT_HPP

#include <cpprest/http_client.h>
#include <cpprest/http_listener.h>

#include <atomic>
#include <unordered_map>
#include <string>
#include <thread>
#include <future>
#include <chrono>

namespace nntu::net {

	class eureka_client final {
	public:
		eureka_client();

		void enroll(const char* url);

		virtual ~eureka_client();

	protected:
		void run_heart_beating();

		void unregister();

	private:
		web::json::value config_;
		web::uri_builder eureka_path_;
		std::atomic<bool> continue_heart_beat_ = false;
		std::string eureka_url_;
		std::shared_future<void> hear_beat_event_;
	};

}
#endif //FDP_EUREKA_CLIENT_HPP
