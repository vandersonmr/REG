#ifndef SEMBLANCE_H__
#define SEMBLANCE_H__

#include <vector.h>
#include <su.h>

typedef struct aperture aperture_t;

struct aperture {
    float ap_m, ap_h, ap_t;
    vector_t(su_trace_t*) traces;
};

float semblance_2d(aperture_t *ap,
        float A, float B, float C, float D, float E,
        float t0, float m0, float h0, float *stack);

#endif /* SEMBLANCE_H__ */
