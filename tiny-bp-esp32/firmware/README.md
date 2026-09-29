# ESP32-S3 PPG base firmware

The default image runs the complete current-window PPG base as an **ESP-DL W16A16** model: TinyBP CNN plus the selected PPG correction head. The embedded file is `main/models/s3/ppg_base.espdl`. `main/ppg_base_model.hpp` and `main/ppg_base_weights.hpp` remain as a host FP32 reference; the earlier `main/models/s3/model.espdl` is inactive.

## Build on another machine

Install ESP-IDF v5.5 or a compatible ESP32-S3 release, clone this branch, and run. The component manager obtains ESP-DL 3.3.11 or a compatible 3.x version:

```bash
cd tiny-bp-esp32/firmware
idf.py set-target esp32s3
idf.py build
idf.py -p YOUR_PORT flash monitor
```

The default source replays one real 10-second VitalDB PPG fixture. On an ESP32-S3 (rev v0.2, 8 MB flash, 8 MB PSRAM) the console printed `ppg-current-seed20260929-espdl-w16a16` and **115.30 / 62.10 mmHg**, matching the ESP-PPQ simulation recorded in `main/ppg_fixture.json` (115.296875 / 62.1015625); the float reference is 114.39/62.03. A single fixture checks execution and scaling, not model accuracy. Investigate a large mismatch before connecting a live sensor.

The firmware accepts one raw PPG sample at 125 Hz through `bp::SampleSource`. A missing, invalid, or late sample should return `Gap`, which clears the 10-second buffer. The first result needs 1,250 samples; subsequent results use a sliding 10-second window every 125 samples. [SENSOR_INTERFACE.md](SENSOR_INTERFACE.md) describes the callback for a live sensor. Use the same raw signal scale as the training data for inference; wrist sensors require their own measured conversion and validation.

The complete graph was exported to ONNX and converted with ESP-PPQ. On 193 windows from five held-out cases, the PC W16A16 simulation differed from ONNX float inference by mean absolute **0.77 / 0.33 mmHg** (SBP/DBP), with maxima **4.42 / 2.73 mmHg**. This measures conversion error only.

Replaying the fixture 191 times on the board gave a median `model.run()` time of **5.97 ms** (range 5.95–6.41 ms, the maximum on the first cold inference) at 160 MHz, timed by `esp_timer` around inference only, so the 1,250-sample int16 input quantization is excluded. After the first window a new one is inferred every second, so inference uses roughly 0.6% of that budget. Board model accuracy has not been measured; the 193-window figure above is PC conversion error, not board accuracy. `tools/host_smoke.cpp` checks the retained FP32 reference without ESP-IDF.
