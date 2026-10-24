/*
 * Smart Gas AI - ESP32 + ML + Status LEDs
 * MQ-5: GPIO34
 * Green: GPIO25 = Normal
 * Yellow: GPIO26 = Warning
 * Red: GPIO27 = Danger
 */

#include "gas_model.h"

constexpr int MQ5_PIN = 34;
constexpr int GREEN_LED = 25;
constexpr int YELLOW_LED = 26;
constexpr int RED_LED = 27;

constexpr unsigned long SERIAL_BAUD = 115200;
constexpr unsigned long SAMPLE_PERIOD_MS = 1000;

void setLEDs(int cls) {
    // Turn all LEDs off first
    digitalWrite(GREEN_LED, LOW);
    digitalWrite(YELLOW_LED, LOW);
    digitalWrite(RED_LED, LOW);

    if (cls == GAS_CLASS_NORMAL) {
        digitalWrite(GREEN_LED, HIGH);
    }
    else if (cls == GAS_CLASS_WARNING) {
        digitalWrite(YELLOW_LED, HIGH);
    }
    else if (cls == GAS_CLASS_DANGER) {
        digitalWrite(RED_LED, HIGH);
    }
}

void setup() {
    Serial.begin(SERIAL_BAUD);

    pinMode(GREEN_LED, OUTPUT);
    pinMode(YELLOW_LED, OUTPUT);
    pinMode(RED_LED, OUTPUT);

    digitalWrite(GREEN_LED, LOW);
    digitalWrite(YELLOW_LED, LOW);
    digitalWrite(RED_LED, LOW);

    analogReadResolution(12);
    analogSetPinAttenuation(MQ5_PIN, ADC_11db);

    delay(200);

    Serial.println("Smart Gas AI - Prototype Classifier");
    Serial.println(
        "raw_adc,class,prob_normal,prob_warning,prob_danger"
    );
}

void loop() {
    int raw = analogRead(MQ5_PIN);
    float adc = static_cast<float>(raw);

    int cls = gas_model_predict(adc);

    float probs[GAS_MODEL_N_CLASSES];
    gas_model_predict_proba(adc, probs);

    // Update LEDs using the model's prediction
    setLEDs(cls);

    // Preserve the original probability output
    if (cls >= 0 && cls < GAS_MODEL_N_CLASSES) {
        Serial.printf(
            "%d,%s,%.4f,%.4f,%.4f\n",
            raw,
            GAS_MODEL_CLASS_NAMES[cls],
            probs[GAS_CLASS_NORMAL],
            probs[GAS_CLASS_WARNING],
            probs[GAS_CLASS_DANGER]
        );
    } else {
        Serial.printf("%d,UNKNOWN,0,0,0\n", raw);
        setLEDs(-1);
    }

    delay(SAMPLE_PERIOD_MS);
}