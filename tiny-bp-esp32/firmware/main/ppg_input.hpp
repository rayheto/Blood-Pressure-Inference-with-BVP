#pragma once
#include "ppg_preprocess.hpp"
#include <cmath>
#include <cstddef>

namespace bp {
enum class ReadResult { Sample, Gap, End };

// read() supplies one raw PPG sample at 125 Hz. Gap means a dropped sample;
// End means the source has finished.
struct SampleSource {
    ReadResult (*read)(void* context, float* raw_sample);
    void* context;
};

class WindowBuffer {
public:
    void reset() {
        next_ = 0;
        count_ = 0;
        since_window_ = 0;
        emitted_ = false;
    }
    bool push(float sample) {
        if (!std::isfinite(sample)) {
            reset();
            return false;
        }
        samples_[next_] = sample;
        next_ = (next_ + 1) % bp_config::kWindowSamples;
        if (count_ < bp_config::kWindowSamples) ++count_;
        if (emitted_) ++since_window_;
        return true;
    }
    // First window needs 10 s; subsequent windows arrive every 1 s.
    bool take_window(float* out) {
        if (count_ < bp_config::kWindowSamples ||
            (emitted_ && since_window_ < bp_config::kStrideSamples)) return false;
        for (std::size_t i = 0; i < bp_config::kWindowSamples; ++i)
            out[i] = samples_[(next_ + i) % bp_config::kWindowSamples];
        emitted_ = true;
        since_window_ = 0;
        return true;
    }
private:
    float samples_[bp_config::kWindowSamples] = {};
    std::size_t next_ = 0;
    std::size_t count_ = 0;
    std::size_t since_window_ = 0;
    bool emitted_ = false;
};
}  // namespace bp
