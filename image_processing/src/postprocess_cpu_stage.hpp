#ifndef FDP_POSTPROCESS_CPU_STAGE_HPP
#define FDP_POSTPROCESS_CPU_STAGE_HPP

#include <stage.hpp>

namespace nntu::img {

    class postprocess_cpu_stage final : public stage {
    public:
        explicit postprocess_cpu_stage(int required_size);

        void submit(cv::Mat* begin, cv::Mat* end) override;

        void wait() override;

    private:
        int required_size_ = 0;
    };

}
#endif //FDP_POSTPROCESS_CPU_STAGE_HPP
