cmake_minimum_required(VERSION 3.17)
project(fdp)

set(CMAKE_CXX_STANDARD 17)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF)

find_package(OpenCV REQUIRED)

add_subdirectory(image_processing)
add_subdirectory(server)
add_subdirectory(third-party/google-benchmark)
add_subdirectory(benchmarks)

add_executable(fdp main.cpp)
target_include_directories(fdp PRIVATE image_processing/include server/include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(fdp PUBLIC nip nips ${OpenCV_LIBRARIES})
#target_compile_options(fdp PUBLIC -g PUBLIC -O0)
