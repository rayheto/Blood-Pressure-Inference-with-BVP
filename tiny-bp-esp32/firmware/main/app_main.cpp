#include "dl_model_base.hpp"
#include "esp_timer.h"
#include "ppg_fixture.hpp"
#include "ppg_input.hpp"
#include <cstdint>
#include <cstdio>

extern const uint8_t model_espdl[] asm("_binary_model_espdl_start");

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
    std::printf("TinyBP: replaying real VitalDB PPG, case %s, t=%d s\n",
                bp_fixture::kCaseId, bp_fixture::kTimeSeconds);
    dl::Model model((const char *)model_espdl, fbs::MODEL_LOCATION_IN_FLASH_RODATA);
    auto *input = model.get_inputs().begin()->second;
    auto *output = model.get_outputs().begin()->second;
    auto *input_data = static_cast<int8_t *>(input->data);

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

        // Model input is NWC: 1 x 1250 x 1, in chronological order.
        for (std::size_t i = 0; i < bp_config::kWindowSamples; ++i) {
            const float normalized =
                (raw_window[i] - bp_config::kPpgMean) / bp_config::kPpgStd;
            input_data[i] = dl::quantize<int8_t>(normalized,
                                                  DL_RESCALE(input->exponent));
        }
        const int64_t start_us = esp_timer_get_time();
        model.run();
        const int64_t elapsed_us = esp_timer_get_time() - start_us;
        auto *output_data = static_cast<int8_t *>(output->data);
        const float sbp = bp_config::kOutputScaleMmhg *
            dl::dequantize(output_data[0], DL_SCALE(output->exponent));
        const float dbp = bp_config::kOutputScaleMmhg *
            dl::dequantize(output_data[1], DL_SCALE(output->exponent));
        std::printf("VitalDB replay: estimate %.2f / %.2f mmHg, "
                    "reference %.2f / %.2f, float PC %.2f / %.2f, "
                    "inference %lld us\n",
                    sbp, dbp, bp_fixture::kReferenceSbp,
                    bp_fixture::kReferenceDbp, bp_fixture::kExpectedFloatSbp,
                    bp_fixture::kExpectedFloatDbp,
                    static_cast<long long>(elapsed_us));
    }
}
