#ifndef FDP_POSTPROCESS_GPU_STAGE_HPP
#define FDP_POSTPROCESS_GPU_STAGE_HPP

#include <stage.hpp>

namespace nntu::img {

    class postprocess_gpu_stage final : public stage {
    public:
        explicit postprocess_gpu_stage(int required_size);

        void submit(cv::Mat* begin, cv::Mat* end) override;

        void wait() override;

    private:
        int required_size_;
    };

}
#endif //FDP_POSTPROCESS_GPU_STAGE_HPP
