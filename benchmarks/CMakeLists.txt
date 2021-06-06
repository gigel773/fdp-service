project(nipb CXX)

find_package(OpenCV REQUIRED)

add_executable(nipb main.cpp default_pipeline_test.cpp gpu_pipeline_test.cpp common/util.cpp combined_pipeline_test.cpp cpu_pipeline_test.cpp)
target_include_directories(nipb PRIVATE $<TARGET_PROPERTY:nip,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(nipb PUBLIC benchmark::benchmark nip ${OpenCV_LIBRARIES})
