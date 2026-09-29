#include "esp_timer.h"
#include "ppg_base_model.hpp"
#include "ppg_fixture.hpp"
#include "ppg_input.hpp"
#include <cstdint>
#include <cstdio>

namespace {
struct FixtureCursor { std::size_t index = 0; };
bp::ReadResult read_fixture(void* context, float* sample) {
    auto& cursor = *static_cast<FixtureCursor*>(context);
    if (cursor.index == bp_config::kWindowSamples) return bp::ReadResult::End;
    *sample = bp_fixture::kRawPpg[cursor.index++];
    return bp::ReadResult::Sample;
}

// Keep the 10 s buffer and scratch window off app_main's small task stack.
bp::WindowBuffer window_buffer;
float raw_window[bp_config::kWindowSamples];
}  // namespace

extern "C" void app_main(void) {
    static_assert(sizeof(bp_fixture::kRawPpg) / sizeof(float) ==
                  bp_config::kWindowSamples, "Fixture length mismatch");
    std::printf("Model %s: replaying real VitalDB PPG, case %s, t=%d s\n",
                bp::PpgBaseModel::tag(), bp_fixture::kCaseId, bp_fixture::kTimeSeconds);

    FixtureCursor fixture;
    bp::SampleSource source{read_fixture, &fixture};
    float sample = 0.0f;
    while (true) {
        const bp::ReadResult result = source.read(source.context, &sample);
        if (result == bp::ReadResult::End) break;
        if (result == bp::ReadResult::Gap) {
            window_buffer.reset();
            continue;
        }
        if (!window_buffer.push(sample)) {
            std::printf("PPG invalid sample; collecting a fresh 10 s window\n");
            continue;
        }
        if (!window_buffer.take_window(raw_window)) continue;

        float prediction[2];
        const int64_t start_us = esp_timer_get_time();
        bp::PpgBaseModel::infer(raw_window, prediction);
        const int64_t elapsed_us = esp_timer_get_time() - start_us;
        std::printf("VitalDB replay: estimate %.2f / %.2f mmHg, "
                    "reference %.2f / %.2f, float PC %.2f / %.2f, "
                    "inference %lld us\n",
                    prediction[0], prediction[1], bp_fixture::kReferenceSbp,
                    bp_fixture::kReferenceDbp, bp_fixture::kExpectedFloatSbp,
                    bp_fixture::kExpectedFloatDbp,
                    static_cast<long long>(elapsed_us));
    }
}
