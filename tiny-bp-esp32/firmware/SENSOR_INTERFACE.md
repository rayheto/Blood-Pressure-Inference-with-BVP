# PPG input interface

`main/ppg_input.hpp` defines `bp::SampleSource`, a callback returning one **raw** PPG sample per call. The callback blocks until the next sample at 125 Hz, writes it to `*raw_sample`, then returns `Sample`. Return `Gap` for a lost or invalid sample, a FIFO overflow, sensor disconnect, or timing break; the 10-second buffer then resets. Return `End` only when acquisition stops. Only one task calls the callback and window buffer.

`WindowBuffer` first emits 1,250 chronological samples after 10 seconds, then emits another overlapping 10-second window every 125 new samples (1 second). `app_main.cpp` normalizes the window, quantizes it to INT16, and runs the embedded ESP-DL model. Its two outputs are scaled by 4 to SBP/DBP in mmHg. For a live sensor, replace `FixtureCursor fixture; bp::SampleSource source{read_fixture, &fixture};` in `app_main.cpp` with a sensor context and its `read` callback. For example:

```cpp
struct SensorContext { /* device handle, FIFO state, timing state */ };
bp::ReadResult read_sensor(void* opaque, float* raw_sample) {
    auto& sensor = *static_cast<SensorContext*>(opaque);
    // Wait for the next 125 Hz PPG sample, convert it to the raw signal scale
    // expected by the trained model, and write *raw_sample.
    // Return Gap if any sample was dropped. This callback is device-specific.
}
SensorContext sensor;
bp::SampleSource source{read_sensor, &sensor};
```

The currently embedded fixture is one **real VitalDB SNUADC/PLETH training window**, case 2158, subject 3312, starting at 1856 seconds. The raw values and their SHA-256 are in `main/ppg_fixture.hpp` and `main/ppg_fixture.json`. Reference pressure is about 119.90/79.99 mmHg; the selected PPG base float model predicts about 114.39/62.03 mmHg, and ESP-PPQ's INT16 simulation predicts about 115.30/62.10 mmHg. This visible error is part of the fixture, not a passing accuracy claim. `../ppg_base/export_firmware.py` regenerates the FP32 reference weight header, preprocessing constants, and float fixture expectation from tracked checkpoints. `tools/generate_fixture.py` is the earlier TinyBP-only fixture generator and would overwrite the selected model's expectation if run afterward.

The training PPG is from an operating-room finger probe. A wrist sensor can have different wavelength, geometry, units, gain, offset, and motion artifacts. Feeding its ADC values into the current mean/std is only an interface check. Record synchronized wrist PPG and reference BP, establish signal conversion and quality checks, then train and evaluate for that sensor before interpreting its BP output.
