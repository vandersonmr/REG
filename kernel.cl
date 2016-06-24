typedef struct su_trace su_trace_t;

typedef struct aperture aperture_t;

struct aperture {
    float ap_m, ap_h, ap_t, len;
};

#define MAX(a,b) (((a)>(b))?(a):(b))

float semblance_2d( __constant aperture_t * __restrict ap,  __constant float * __restrict vdm,  __constant float * __restrict vdh,
         __global float *__restrict data, float A, float B, float C, float D, float E,
        float t0, float m0, float h0, float idt, float *stack)
{
    int tau = MAX((int)(ap->ap_t * idt), 0);
    int w = 2 * tau + 1;

    __local float num[5*5*5*5];
    int basenum = get_local_id(0)*5 + get_local_id(1)*25 + get_local_id(2)*125;
    float aux = 0;
    for (int i = 0; i < w; i++) {
      num[basenum + i] = 0;
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
	int base = it - tau;
        if (base >= 0 && it + tau < 2502) {
           float v2 = (t*idt - it);
           for (int j = 0; j < w; j++) {
                int k = base + j;
                float v = data[i*2052 + k+1]*v2 + data[i*2052 + k]*(1-v2);
                num[basenum + j] += v;
                aux += v*v;
                _stack += v;
            }
            M++;
        } else if (++skip == 2) {
            return 0;
        }
    }

    float sem = 0;
    for (int j = 0; j < w; j++) {
        sem += num[basenum + j] * num[basenum + j];
    }

    if (stack) {
        _stack /= M*w;
	*stack = _stack;
    }

    if (aux == 0)
        return 0;

    return sem / aux / M;
}

typedef struct {
	float a,b,c;
} index;

__kernel void compute_max(__constant aperture_t * __restrict ap,  __constant float * __restrict dm,  __constant float * __restrict dh,
                          __global float * __restrict data, float t0, float m0, float h0, float idt,
                          __constant float ns[2][5], __constant int np[5], __global float * __restrict results)
{
  short ia = get_global_id(0);
  short ib = get_global_id(1);
  short ic = get_global_id(2);
  int base = ia*(np[2]*np[1]*7) + ib*(np[1]*7) + (ic*7);
  
  results[base] = -1;

  float a = ns[0][0] + ((float)ia / (float)np[0])*(ns[1][0]-ns[0][0]);
  float b = ns[0][1] + ((float)ib / (float)np[1])*(ns[1][1]-ns[0][1]);
  float c = ns[0][2] + ((float)ic / (float)np[2])*(ns[1][2]-ns[0][2]);

  for (short id = 0; id < np[3]; id++) {

      float d = ns[0][3] + ((float)id / (float)np[3])*(ns[1][3]-ns[0][3]);
      for (short ie = 0; ie < np[4]; ie++) {
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
