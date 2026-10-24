# Limitations

## Scope of the evidence

The reported metrics describe the supplied synthetic dataset. They are not
measurements of a physical MQ-5 installation. A 93.94% held-out accuracy on this
data does not imply 93.94% accuracy in a kitchen, laboratory, enclosure or
industrial setting.

## Synthetic data

The dataset is described as synthetic/enclosed. It does not contain:

- real MQ-5 warm-up transients;
- sensor-to-sensor manufacturing variation;
- ageing, contamination or drift;
- supply-voltage variation;
- temperature/humidity cross-sensitivity of the physical sensor;
- gas mixtures, interfering gases or airflow effects;
- background odours or cleaning products;
- the long-tail noise of a real ADC and power supply.

The balanced class design is also unrealistic for a leak-detection prior: real
events are rare, while this dataset gives each class exactly one third of the
rows.

## Class overlap and errors

Adjacent classes overlap in ADC space. On the held-out test split:

- Warning recall is 0.8990, the lowest of the three classes;
- 13 Warning rows are predicted Normal;
- 7 Warning rows are predicted Danger;
- 10 Danger rows are predicted Warning;
- there are **0 Danger → Normal** errors.

The absence of Danger→Normal errors in three test sessions is encouraging but is
not a safety guarantee. A different session, sensor or environment could produce
that error. The confidence interval for accuracy is also wide: [0.9172, 0.9559]
by Wilson interval, and [0.9192, 0.9646] by session-group bootstrap.

## Small session-level sample

Only three sessions are held out for test and three for validation. Session-level
uncertainty is therefore estimated from very few groups. The bootstrap intervals
resample whole sessions, which is more honest than resampling rows, but they
remain coarse.

## No advantage over a threshold rule

A two-threshold rule fitted on the training split reaches 0.9411 test accuracy
and 0.9409 macro F1, slightly above the model's 0.9394 / 0.9393. McNemar's exact
test gives p = 1.0 (one discordant pair). The ML model has not demonstrated a
statistically significant accuracy advantage on this dataset.

The model is still useful as a compact end-to-end example: it produces smooth
probabilities, has a 56-byte parameter footprint and demonstrates a complete
Python-to-C export. That is an engineering demonstration, not evidence of
superior gas detection.

## Calibration

Top-label ECE is 0.0117 and MCE is 0.2574 on the test split. The average
calibration looks good, but the maximum calibration error is large. Probabilities
were not recalibrated. Do not use a probability threshold for alarms without
validating it on real data.

## Hardware status

- The C header has been compiled and parity-tested on the host with GCC 6.3.0.
- It has **not** been cross-compiled with the ESP32 toolchain.
- It has **not** been flashed to an ESP32 or run on device.
- Flash, RAM, latency and thermal behaviour have not been measured.
- The firmware has not been tested with a real MQ-5.
- The ADC path, supply, divider (if any) and LED wiring are hardware-specific and
  have not been validated in this repository.

`models/model_card.json` therefore records
`ready_for_esp32_deployment: false`.

## Safety and certification

This project is not:

- a certified gas detector;
- a fire, explosion or toxic-gas protection system;
- a substitute for a commercial alarm;
- a fail-safe device.

The firmware has no buzzer, siren, relay or independent alarm path. It only lights
one LED and prints a serial line. A real installation needs an independent
fail-safe alarm, appropriate sensor certification, enclosure design, power
redundancy and regulatory review.

## Dataset provenance and license

The original dataset's source, generation method and license could not be
verified. The MIT license in this repository covers the source code and
documentation; it does not automatically grant rights to redistribute the
dataset. Confirm permission before publishing or redistributing the CSV.

## Relationship to the legacy project

A separate folder named `Smart-Gas-AI` contains a different 4-class experiment
using a Random Forest model. That project is not the source of this dataset,
model, firmware or results. This repository is self-contained and should not be
merged with the legacy folder.

## What would change the conclusions

Meaningful deployment evidence would require:

1. Labelled real-sensor data from multiple units, sessions and environments.
2. A time-based or session-grouped split that reflects the deployment stream.
3. Calibration and threshold selection on real data.
4. Cross-compilation, on-device timing and memory measurements.
5. Fault injection: sensor disconnect, ADC saturation, supply brownout.
6. An independent fail-safe alarm path and a formal risk assessment.

Until then, treat this repository as an educational and research prototype.
