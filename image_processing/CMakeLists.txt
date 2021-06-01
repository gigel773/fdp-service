project(nip)

find_package(InferenceEngine REQUIRED)
find_package(ngraph REQUIRED)
find_package(TBB REQUIRED COMPONENTS tbb)
find_package(Boost COMPONENTS thread log REQUIRED)

add_library(nip
        src/landmarks_stage.cpp
        src/facial_cut_stage.cpp
        src/pipeline_impl.cpp
        src/scale_stage.cpp
        src/sharpen_stage.cpp
        src/sharpen_stage.hpp
        src/blur_stage.cpp
        src/blur_stage.hpp
        src/equalize_stage.cpp
        src/equalize_stage.hpp
        src/preprocess_gpu_stage.cpp src/postprocess_gpu_stage.cpp src/postprocess_gpu_stage.hpp)

target_include_directories(nip PUBLIC include
        PRIVATE ${InferenceEngine_INCLUDE_DIRS}
        PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(nip PUBLIC IE::inference_engine ${OpenCV_LIBRARIES} ${NGRAPH_LIBRARIES} ${Boost_LIBRARIES} TBB::tbb)

#target_compile_options(nip PUBLIC -g PUBLIC -O0)
