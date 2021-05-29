project(nip)

find_package(InferenceEngine REQUIRED)
find_package(ngraph REQUIRED)

add_library(nip
        src/landmarks_stage.cpp
        src/facial_cut_stage.cpp
        src/default_pipeline.cpp
        src/scale_stage.cpp src/sharpen_stage.cpp src/sharpen_stage.hpp)

target_include_directories(nip PUBLIC include
        PRIVATE ${InferenceEngine_INCLUDE_DIRS}
        PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(nip PRIVATE optimized IE::inference_engine opencv_core ${NGRAPH_LIBRARIES})

#target_compile_options(nip PUBLIC -g PUBLIC -O0)
