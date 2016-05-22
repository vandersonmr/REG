#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <utils.h>
#include <vector.h>
#include <semblance.h>
#include <su.h>
#include <errno.h>

/*
 * compute_max finds the best parameters 'Aopt', 'Bopt', 'Copt', 'Dopt' and 'Eopt' 
 * that fit a curve to the data in 'ap' from a reference point (m0, h0, t0). Also 
 * returning its fit (coherence/semblance) through 'sem' and the average of values 
 * along the curve through 'stack'
 *
 * The lower limit for searching each parameter is specified as a element in the 
 * vector 'n0' and the upper limit in vector 'n1', the number of divisions for 
 * the search space is specified through 'np'
 */
void compute_max(aperture_t *ap, float m0, float h0, float t0,
    const float n0[5], const float n1[5], const int np[5], float *Aopt,
    float *Bopt, float *Copt, float *Dopt, float *Eopt, float *sem,
    float *stack)
{
    /* The parallel version of the code will compute the best parameters for 
     * each value of the parameter 'A', so we need to store np[0] different 
     * values of each parameter, stack and semblance */
    float _Aopt[np[0]], _Bopt[np[0]], _Copt[np[0]], 
          _Dopt[np[0]], _Eopt[np[0]];
    float smax[np[0]];
    float _stack[np[0]];

    /* Split the outermost loop between threads. Each thread will
     * compute the best fit for a given parameter 'A' value */
    #pragma omp parallel for schedule(dynamic)
    for (int ia = 0; ia < np[0]; ia++) {
        smax[ia] = -1;
        float a = n0[0] + ((float)ia / (float)np[0])*(n1[0]-n0[0]);
        for (int ib = 0; ib < np[1]; ib++) {
            float b = n0[1] + ((float)ib / (float)np[1])*(n1[1]-n0[1]);
            for (int ic = 0; ic < np[2]; ic++) {
                float c = n0[2] + ((float)ic / (float)np[2])*(n1[2]-n0[2]);
                for (int id = 0; id < np[3]; id++) {
                    float d = n0[3] + ((float)id / (float)np[3])*(n1[3]-n0[3]);
                    for (int ie = 0; ie < np[4]; ie++) {
                        float e = n0[4] + ((float)ie / (float)np[4])*(n1[4]-n0[4]);
                        float st;
                        /* Check the fit of the parameters to the data and update the 
                         * maximum for that point if necessary */
                        float s = semblance_2d(ap, a, b, c, d, e, t0, m0, h0, &st);
                        if (s > smax[ia]) {
                            smax[ia] = s;
                            _stack[ia] = st;
                            _Aopt[ia] = a;
                            _Bopt[ia] = b;
                            _Copt[ia] = c;
                            _Dopt[ia] = d;
                            _Eopt[ia] = e;
                        }
                    }
                }
            }
            /* Uncomment this to roughly check the progress */
            /* fprintf(stderr, "."); */
        }
    }

    /* Now find the best fit between different 'A' values */
    float ssmax = -1.0;
    *stack = 0;
    for (int ia = 0; ia < np[0]; ia++) {
        if (smax[ia] > ssmax) {
            *Aopt = _Aopt[ia];
            *Bopt = _Bopt[ia];
            *Copt = _Copt[ia];
            *Dopt = _Dopt[ia];
            *Eopt = _Eopt[ia];
            *stack = _stack[ia];
            *sem = smax[ia];
            ssmax = smax[ia];
        }
    }
}

int main(int argc, char *argv[])
{
    int i;

    if (argc != 21) {
        fprintf(stderr, "Usage: %s M0 H0 T0 TAU A0 A1 NA B0 B1 NB "
            "C0 C1 NC D0 D1 ND E0 E1 NE INPUT\n", argv[0]);
        exit(1);
    }

    float m0 = atof(argv[1]);
    float h0 = atof(argv[2]);
    float t0 = atof(argv[3]);
    float tau = strtof(argv[4], NULL);

    /* A, B, C, D, E */
    float p0[5], p1[5];
    int np[5];

    /* p0 is where the search starts, p1 is where the search ends and np is the 
     * number of points in between p0 and p1 to do the search */   
    for (i = 0; i < 5; i++) {
        p0[i] = atof(argv[5 + 3*i]);
        p1[i] = atof(argv[5 + 3*i + 1]);
        np[i] = atoi(argv[5 + 3*i + 2]);
    }

    /* Load the traces from the file */

    char *path = argv[20];
    FILE *fp = fopen(path, "r");

    if (!fp) {
        fprintf(stderr, "Failed to open prestack file '%s'!\n", path);
        return 1;
    }

    su_trace_t tr;
    vector_t(su_trace_t) traces;
    vector_init(traces);

    while (su_fgettr(fp, &tr)) {
        vector_push(traces, tr);
    }

    /* Construct the aperture structure from the traces, which is a vector
     * containing pointers to traces */

    aperture_t ap;
    ap.ap_m = 0;
    ap.ap_h = 0;
    ap.ap_t = tau;
    vector_init(ap.traces);
    for (int i = 0; i < traces.len; i++)
        vector_push(ap.traces, &vector_get(traces, i));

    /* Find the best parameter combination */

    float a, b, c, d, e, sem, stack;
    compute_max(&ap, m0, h0, t0, p0, p1, np, &a, &b, &c, &d, &e, &sem, &stack);

    printf("A=%g\n", a);
    printf("B=%g\n", b);
    printf("C=%g\n", c);
    printf("D=%g\n", d);
    printf("E=%g\n", e);
    printf("Stack=%g\n", stack);
    printf("Semblance=%g\n", sem);
    printf("\n");

    return 0;
}
