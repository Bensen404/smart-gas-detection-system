/*
 * Smart Gas AI - frozen classifier export
 * ---------------------------------------
 * Model: gaussian_nb_adc (GaussianNB)
 * Input: MQ-5 ADC, 12-bit (0..4095), feature order: mq5_adc
 * Output classes: 0 = Normal, 1 = Warning, 2 = Danger
 *
 * Auto-generated from models/final_model.joblib. Do not edit by hand.
 * Dataset SHA-256: 82BFC7557E247D06E54C7B81AD5F2A9A294C5A44FD006274D451C6D99F1F01C7
 * Random seed: 42
 *
 * Header-only C99. It performs one standardisation, three class scores and an
 * argmax. Probabilities additionally use expf(). There is no dynamic
 * allocation and no external ML runtime.
 *
 * This is an educational prototype, not a safety-certified gas detector.
 */

#ifndef SMART_GAS_AI_MODEL_H
#define SMART_GAS_AI_MODEL_H

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GAS_MODEL_VERSION "1.0.0"
#define GAS_MODEL_N_FEATURES 1
#define GAS_MODEL_N_CLASSES 3

typedef enum {
    GAS_CLASS_NORMAL = 0,
    GAS_CLASS_WARNING = 1,
    GAS_CLASS_DANGER = 2
} gas_class_t;

static const char *GAS_MODEL_CLASS_NAMES[GAS_MODEL_N_CLASSES] = {
    "Normal", "Warning", "Danger"
};

/* StandardScaler statistics (float32). */
static const float GAS_MODEL_MEAN[GAS_MODEL_N_FEATURES] = {
    1794.64343f
};
static const float GAS_MODEL_SCALE[GAS_MODEL_N_FEATURES] = {
    776.395691f
};

/* Gaussian naive-Bayes class parameters (float32), one value per class. */
static const float GAS_MODEL_THETA[GAS_MODEL_N_CLASSES] = {
    -1.12912595f, -0.0445024557f, 1.17362833f
};
static const float GAS_MODEL_INV_VAR[GAS_MODEL_N_CLASSES] = {
    16.9610138f, 8.88665295f, 5.74041176f
};
static const float GAS_MODEL_LOG_CONST[GAS_MODEL_N_CLASSES] = {
    -0.602092087f, -0.925275505f, -1.14378536f
};
static const float GAS_MODEL_CLASS_PRIOR[GAS_MODEL_N_CLASSES] = {
    0.333333343f, 0.333333343f, 0.333333343f
};

/*
 * Compute three class scores for one ADC sample.
 * adc: raw 12-bit reading (0..4095). Out-of-range values are accepted by this
 * function but are outside the training distribution and should be rejected
 * by the caller.
 */
static inline void gas_model_logits(float adc, float logits[GAS_MODEL_N_CLASSES])
{
    const float z = (adc - GAS_MODEL_MEAN[0]) / GAS_MODEL_SCALE[0];
    for (int c = 0; c < GAS_MODEL_N_CLASSES; ++c) {
        const float d = z - GAS_MODEL_THETA[c];
        logits[c] = GAS_MODEL_LOG_CONST[c] - 0.5f * d * d * GAS_MODEL_INV_VAR[c];
    }
}

/* Return the most likely class index without calling expf(). */
static inline int gas_model_predict(float adc)
{
    float logits[GAS_MODEL_N_CLASSES];
    gas_model_logits(adc, logits);
    int best = 0;
    for (int c = 1; c < GAS_MODEL_N_CLASSES; ++c) {
        if (logits[c] > logits[best]) {
            best = c;
        }
    }
    return best;
}

/* Numerical-stability softmax; writes three probabilities that sum to 1. */
static inline void gas_model_predict_proba(
    float adc,
    float probs[GAS_MODEL_N_CLASSES]
)
{
    float logits[GAS_MODEL_N_CLASSES];
    gas_model_logits(adc, logits);
    float maximum = logits[0];
    for (int c = 1; c < GAS_MODEL_N_CLASSES; ++c) {
        if (logits[c] > maximum) {
            maximum = logits[c];
        }
    }
    float total = 0.0f;
    for (int c = 0; c < GAS_MODEL_N_CLASSES; ++c) {
        probs[c] = expf(logits[c] - maximum);
        total += probs[c];
    }
    for (int c = 0; c < GAS_MODEL_N_CLASSES; ++c) {
        probs[c] /= total;
    }
}

#ifdef __cplusplus
}
#endif

#endif /* SMART_GAS_AI_MODEL_H */
