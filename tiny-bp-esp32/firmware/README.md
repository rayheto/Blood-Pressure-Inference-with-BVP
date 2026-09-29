# ESP32-S3 PPG base firmware

The default image runs `ppg-current-seed20260929-fp32`: the tracked TinyBP CNN followed by the selected current-PPG correction head. The model is self-contained in `main/ppg_base_weights.hpp`; the previous `main/models/s3/model.espdl` is an inactive historical artifact. This build does not require ESP-DL or ESP-PPQ.

## Build on another machine

Install ESP-IDF v5.5 or a compatible ESP32-S3 release, clone this branch, and run:

```bash
cd tiny-bp-esp32/firmware
idf.py set-target esp32s3
idf.py build
idf.py -p YOUR_PORT flash monitor
```

The default source replays one real 10-second VitalDB PPG fixture. A successful functional smoke test prints model tag `ppg-current-seed20260929-fp32` and a prediction close to the `float_model_sbp_dbp_mmhg` values in `main/ppg_fixture.json` (allow 0.05 mmHg for FP32 operation order). It also prints board inference time in microseconds. Reference BP in that JSON is for waveform provenance; one window cannot measure model accuracy.

The firmware accepts one raw PPG sample at 125 Hz through `bp::SampleSource`. A missing, invalid, or late sample should return `Gap`, which clears the 10-second buffer. The first result needs 1,250 samples; subsequent results use a sliding 10-second window every 125 samples. [SENSOR_INTERFACE.md](SENSOR_INTERFACE.md) describes the callback for a live sensor. Use the same raw signal scale as the training data for inference; wrist sensors require their own measured conversion and validation.

`main/ppg_base_model.hpp` implements deterministic FP32 convolution, global average pooling, and correction. It uses about 40 KB static scratch memory plus flash-resident weights. The old INT8 timing and quantization results do not apply to this build. This project has no automatic hardware build gate on the current development machine; `tools/host_smoke.cpp` can check the FP32 inference against the PC fixture without ESP-IDF.
