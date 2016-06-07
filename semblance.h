#ifndef SEMBLANCE_H__
#define SEMBLANCE_H__

#include <vector.h>
#include <su.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif

typedef struct aperture aperture_t;

struct aperture {
    cl_float ap_m, ap_h, ap_t, len;
};

float semblance_2d(aperture_t *ap, su_trace_t *traces_s,
        float A, float B, float C, float D, float E,
        float t0, float m0, float h0, float *stack);

#endif /* SEMBLANCE_H__ */
