typedef struct su_trace su_trace_t;

typedef struct aperture aperture_t;

struct aperture {
    float ap_m, ap_h, ap_t, len;
};

#define MAX(a,b) (((a)>(b))?(a):(b))

float semblance_2d(__global aperture_t *ap, __global float *vdm, __global float *vdh,
        __global float *data, float A, float B, float C, float D, float E,
        float t0, float m0, float h0, float idt, float *stack)
{
    int tau = MAX((int)(ap->ap_t * idt), 0);
    int w = 2 * tau + 1;

    float num[5];
    float den[5];
    for (int i = 0; i < w; i++) {
      num[i] = 0;
      den[i] = 0;
    }

    int M = 0, skip = 0;
    float _stack = 0;

    for (int i = 0; i < ap->len; i++) {
        float dm = vdm[i];
        float dh = vdh[i];
        float t;
        float t2 = t0 + (A*dm) + (B*dh);
        t2 = t2*t2 + C*dh*dh + D*dm*dm + E*dh*dm;

        if (t2 < 0) t = -1;
        else t = sqrt(t2);

        int it = (int)(t * idt);

        if (it - tau >= 0 && it + tau < 2502) {
           float v2 = (t*idt - it);
           for (int j = 0; j < 5; j++) {
                int k = it + j - tau;
                float v1 = (data[i*2052 + k+1] - data[i*2052 + k]);
                float v = v1*v2 + data[i*2052 + k];
                num[j] += v;
                den[j] += v*v;
                _stack += v;
            }
            M++;
        } else if (++skip == 2) {
            return 0;
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
}

__kernel void compute_max(__global aperture_t *ap, __global float *dm, __global float *dh,
                          __global float *data, float t0, float m0, float h0, float idt,
                          __global const float ns[2][5],  __global const int np[5], __global float *results)
{
  int ia = get_global_id(0);
  int ib = get_global_id(1);
  int ic = get_global_id(2);
  int base = ia*(np[2]*np[1]*7) + ib*(np[1]*7) + (ic*7);

  results[base] = -1;

  float a = ns[0][0] + ((float)ia / (float)np[0])*(ns[1][0]-ns[0][0]);
  float b = ns[0][1] + ((float)ib / (float)np[1])*(ns[1][1]-ns[0][1]);
  float c = ns[0][2] + ((float)ic / (float)np[2])*(ns[1][2]-ns[0][2]);

  for (int id = 0; id < np[3]; id++) {

      float d = ns[0][3] + ((float)id / (float)np[3])*(ns[1][3]-ns[0][3]);
      for (int ie = 0; ie < np[4]; ie++) {
          float e = ns[0][4] + ((float)ie / (float)np[4])*(ns[1][4]-ns[0][4]);
          float st;

          float s = semblance_2d(ap, dm, dh, data, a, b, c, d, e, t0, m0, h0, idt, &st);
          if (s > results[base]) {
              results[base] = s; //smax
              results[base + 1] = st; //stack
              results[base + 2] = a;
              results[base + 3] = b;
              results[base + 4] = c;
              results[base + 5] = d;
              results[base + 6] = e;
          }

      }
  }
}
