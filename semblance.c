#include "semblance.h"

#include <math.h>
#include <string.h>
#include <vector.h>
#include <utils.h>

/* The moveout time function tells the time when a wave, propagating from
 * (m0,h0) at t0 to the tace */
static float time_2d(float A, float B, float C, float D, float E,
        float t0, float m0, float m, float h0, float h)
{
    float dm = m - m0;
    float dh = h - h0;

    float t2 = t0 + (A*dm) + (B*dh);
    t2 = t2*t2 + C*dh*dh + D*dm*dm + E*dh*dm;

    if (t2 < 0)
        return -1;
    else
        return sqrt(t2);
}

float interpol_linear(float x0, float x1, float y0, float y1, float x)
{
    return (y1 - y0) * (x - x0) / (x1 - x0) + y0;
}

/*
 * This method computes how much the given parameters fit a collection of traces
 * from the aperture. The 'stack' is the average of the values from the traces
 * intersected by the fitted curve
 */
float semblance_2d(aperture_t *ap,
        float A, float B, float C, float D, float E,
        float t0, float m0, float h0,
        float *stack)
{
    /* Get the sample rate from the first trace inside the aperture,
       it is the same value for all other traces */
    su_trace_t *tr = vector_get(ap->traces, 0);
    float dt = (float) tr->dt / 1000000;
    float idt = 1 / dt;

    /* Calculate coherence window (number of trace samples in the trace to
       include in the semblance) */
    int tau = MAX((int)(ap->ap_t * idt), 0);
    int w = 2 * tau + 1;

    /* Calculate the semblance  */

    float num[w], den[w];
    memset(&num[0], 0, sizeof(num));
    memset(&den[0], 0, sizeof(den));
    int M = 0, skip = 0;
    float _stack = 0;

    for (int i = 0; i < ap->traces.len; i++) {
        tr = vector_get(ap->traces, i);

        /* Get the trace coordinates in the midpoint and halfoffset spaces */
        float mx, my, hx, hy;
        su_get_midpoint(tr, &mx, &my);
        su_get_halfoffset(tr, &hx, &hy);

        /* Compute the moveout time ignoring mx and hx because the data is 2D */
        float t = time_2d(A, B, C, D, E, t0, m0, my, h0, hy);
        int it = (int)(t * idt);

        /* Check if the time belongs to the range of the trace */
        if (it - tau >= 0 && it + tau < tr->ns) {
            for (int j = 0; j < w; j++) {
                int k = it + j - tau;
                float v = interpol_linear(k, k+1,
                        tr->data[k], tr->data[k+1],
                        t*idt + j - tau);
                num[j] += v;
                den[j] += v*v;
                _stack += v;
            }
            M++;
        } else if (++skip == 2) {
            /* Allow only one trace to be excluded from the semblance
               computation, otherwise the precision of the metric will
               be compromised */
            goto error;
        }
    }

    float sem = 0;
    float aux = 0;
    for (int j = 0; j < w; j++) {
        sem += num[j] * num[j];
        aux += den[j];
    }

    if (stack) {
        _stack /= M*w;
        *stack = _stack;
    }

    if (aux == 0)
        return 0;

    return sem / aux / M;

error:
    return 0;
}
