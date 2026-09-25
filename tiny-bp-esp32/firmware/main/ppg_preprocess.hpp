#pragma once
#include <cstddef>
namespace bp_config {
inline constexpr std::size_t kSampleRateHz = 125;
inline constexpr std::size_t kWindowSamples = 1250;
inline constexpr std::size_t kStrideSamples = kSampleRateHz;
inline constexpr float kPpgMean = 38.7217608f;
inline constexpr float kPpgStd = 10.7599537f;
inline constexpr float kOutputScaleMmhg = 2.0f;
}
