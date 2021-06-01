#ifndef FDP_PREPROCESS_GPU_STAGE_HPP
#define FDP_PREPROCESS_GPU_STAGE_HPP

#include <stage.hpp>

namespace nntu::img {

    class preprocess_gpu_stage final : public stage {
        static constexpr const char haar_cascade_face_path[] = R"(haarcascade_frontalface_default_cuda.xml)";
    public:
        void submit(cv::Mat* begin, cv::Mat* end) override;

        void wait() override;
    };

}

#endif //FDP_PREPROCESS_GPU_STAGE_HPP
