project(nip)

find_package(InferenceEngine REQUIRED)
find_package(ngraph REQUIRED)

add_library(nip
        src/landmarks_processor.cpp
        src/facial_preprocessor.cpp
        src/default_pipeline.cpp)
target_include_directories(nip PUBLIC include
        PRIVATE ${InferenceEngine_INCLUDE_DIRS}
        PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(nip PRIVATE optimized ${InferenceEngine_LIBRARIES} ${OpenCV_LIBRARIES} ${NGRAPH_LIBRARIES})
