cmake_minimum_required(VERSION 3.17)
project(fdp)

set(CMAKE_CXX_STANDARD 17)

find_package(InferenceEngine REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TBB REQUIRED)
find_package(ngraph REQUIRED)

add_executable(fdp main.cpp)
target_link_libraries(fdp PRIVATE ${InferenceEngine_LIBRARIES} ${OpenCV_LIBRARIES} ${NGRAPH_LIBRARIES})
target_include_directories(fdp PRIVATE ${InferenceEngine_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
