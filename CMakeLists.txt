cmake_minimum_required(VERSION 3.17)
project(fdp)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)

add_subdirectory(image_processing)

add_executable(fdp main.cpp)
target_include_directories(fdp PRIVATE image_processing/include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(fdp PUBLIC nip ${OpenCV_LIBRARIES})
#target_compile_options(fdp PUBLIC -g PUBLIC -O0)
