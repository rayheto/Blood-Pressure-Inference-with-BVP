#include "../main/ppg_base_model.hpp"
#include "../main/ppg_fixture.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--stdin") == 0) {
        float raw[bp_config::kWindowSamples];
        for (float& sample : raw)
            if (std::scanf("%f", &sample) != 1) return 2;
        float result[2];
        bp::PpgBaseModel::infer(raw, result);
        std::printf("%.6f %.6f\n", result[0], result[1]);
        return 0;
    }
    float prediction[2];
    bp::PpgBaseModel::infer(bp_fixture::kRawPpg, prediction);
    std::printf("%s fixture: %.5f / %.5f mmHg\n", bp::PpgBaseModel::tag(),
                prediction[0], prediction[1]);
    const float sbp_error = std::fabs(prediction[0] - bp_fixture::kExpectedFloatSbp);
    const float dbp_error = std::fabs(prediction[1] - bp_fixture::kExpectedFloatDbp);
    if (sbp_error > 0.05f || dbp_error > 0.05f) {
        std::printf("PC reference mismatch: %.5f / %.5f mmHg\n", sbp_error, dbp_error);
        return 1;
    }
    return 0;
}
