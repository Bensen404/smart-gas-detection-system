# ESP32 firmware

## Files

| File | Purpose |
|---|---|
| `esp32_gas_detection.ino` | Deployed sketch: ADC read, inference, LEDs, serial output |
| `gas_model.h` | Generated header-only C classifier; do not hand-edit |

## Pinout

| Function | ESP32 pin | Firmware constant |
|---|---|---|
| MQ-5 analog output | **GPIO34** | `MQ5_PIN` |
| Green LED (`Normal`) | **GPIO25** | `GREEN_LED` |
| Yellow LED (`Warning`) | **GPIO26** | `YELLOW_LED` |
| Red LED (`Danger`) | **GPIO27** | `RED_LED` |

There is no buzzer. GPIO34 is input-only and uses ADC1 with 12-bit resolution and
`ADC_11db` attenuation. Do not exceed 3.3 V on GPIO34.

## Build

1. Install ESP32 board support in the Arduino IDE.
2. Open `esp32_gas_detection.ino`.
3. Keep `gas_model.h` in the same folder.
4. Select a compatible ESP32 board and upload.
5. Open the Serial Monitor at **115200 baud**.

Expected startup output:

```text
Smart Gas AI - Prototype Classifier
raw_adc,class,prob_normal,prob_warning,prob_danger
```

One CSV row is printed per second:

```text
1800,Warning,0.0000,0.9839,0.0161
```

## Regenerate the header

From the repository root:

```bash
python -m src.export_embedded
```

This rewrites `gas_model.h`, `models/gas_model.json`, `models/model_card.json`,
`models/test_vectors.csv` and `results/metrics/embedded_parity.txt`.

## Verification status

- Host C/Python parity: **PASS**, 165/165 cases, max probability difference
  `2.38e-07` (GCC 6.3.0).
- ESP32 cross-compilation: **not performed**.
- On-device test: **not performed**.
- Real-sensor validation: **not performed**.

See [docs/hardware.md](../../docs/hardware.md) and
[docs/limitations.md](../../docs/limitations.md).
