#pragma once

#include "ppg_base_weights.hpp"
#include "ppg_preprocess.hpp"
#include <cmath>
#include <cstddef>

namespace bp {

// FP32 inference for the selected PPG-only CNN + current-window correction.
// Only one task may call infer() at a time; scratch memory is shared to keep
// the ESP32-S3 task stack small.
class PpgBaseModel {
public:
    static constexpr const char* tag() { return bp_weights::kModelTag; }

    static void infer(const float* raw_ppg, float* bp_mmhg) {
        // Conv1d: kernel 9, stride 2, pad 4, fused BatchNorm and ReLU.
        conv_first(raw_ppg, scratch_a_);
        conv<625, 8, 313, 16>(scratch_a_, scratch_b_,
                               bp_weights::kConv1Weight, bp_weights::kConv1Bias);
        conv<313, 16, 157, 24>(scratch_b_, scratch_a_,
                                bp_weights::kConv2Weight, bp_weights::kConv2Bias);
        conv<157, 24, 79, 32>(scratch_a_, scratch_b_,
                               bp_weights::kConv3Weight, bp_weights::kConv3Bias);

        float latent[32] = {};
        for (int t = 0; t < 79; ++t)
            for (int c = 0; c < 32; ++c)
                latent[c] += scratch_b_[t * 32 + c] / 79.0f;

        float tiny[2] = {};
        for (int o = 0; o < 2; ++o) {
            float value = bp_weights::kTinyHeadBias[o];
            for (int c = 0; c < 32; ++c)
                value += bp_weights::kTinyHeadWeight[o * 32 + c] * latent[c];
            tiny[o] = value * 100.0f;
        }

        float features[35];
        for (int c = 0; c < 32; ++c)
            features[c] = (latent[c] - bp_weights::kFeatureMean[c]) /
                          bp_weights::kFeatureStd[c];
        for (int o = 0; o < 2; ++o)
            features[32 + o] = (tiny[o] / 100.0f - bp_weights::kFeatureMean[32 + o]) /
                               bp_weights::kFeatureStd[32 + o];
        features[34] = 1.0f;  // Valid current PPG window.

        float hidden[32];
        for (int o = 0; o < 32; ++o) {
            float value = bp_weights::kCorrection0Bias[o];
            for (int i = 0; i < 35; ++i)
                value += bp_weights::kCorrection0Weight[o * 35 + i] * features[i];
            hidden[o] = value > 0.0f ? value : 0.0f;
        }
        for (int o = 0; o < 2; ++o) {
            float value = bp_weights::kCorrection1Bias[o];
            for (int i = 0; i < 32; ++i)
                value += bp_weights::kCorrection1Weight[o * 32 + i] * hidden[i];
            bp_mmhg[o] = tiny[o] + std::tanh(value) * (o == 0 ? 40.0f : 25.0f);
        }
    }

private:
    static void conv_first(const float* raw, float* out) {
        for (int t = 0; t < 625; ++t) {
            for (int o = 0; o < 8; ++o) {
                float sum = bp_weights::kConv0Bias[o];
                for (int k = 0; k < 9; ++k) {
                    const int position = t * 2 + k - 4;
                    if (position >= 0 && position < 1250) {
                        const float normalized =
                            (raw[position] - bp_config::kPpgMean) / bp_config::kPpgStd;
                        sum += normalized * bp_weights::kConv0Weight[o * 9 + k];
                    }
                }
                out[t * 8 + o] = sum > 0.0f ? sum : 0.0f;
            }
        }
    }

    template <int InLength, int InChannels, int OutLength, int OutChannels>
    static void conv(const float* input, float* output,
                     const float* weight, const float* bias) {
        for (int t = 0; t < OutLength; ++t) {
            for (int o = 0; o < OutChannels; ++o) {
                float sum = bias[o];
                for (int i = 0; i < InChannels; ++i) {
                    for (int k = 0; k < 9; ++k) {
                        const int position = t * 2 + k - 4;
                        if (position >= 0 && position < InLength)
                            sum += input[position * InChannels + i] *
                                   weight[(o * InChannels + i) * 9 + k];
                    }
                }
                output[t * OutChannels + o] = sum > 0.0f ? sum : 0.0f;
            }
        }
    }

    inline static float scratch_a_[625 * 8] = {};
    inline static float scratch_b_[313 * 16] = {};
};

}  // namespace bp
