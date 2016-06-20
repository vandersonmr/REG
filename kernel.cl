#define SU_HEADER_SIZE ((unsigned long)&(((su_trace_t*)0)->data))

typedef struct su_trace su_trace_t;

struct su_trace {
	short scalco; //*
	int sx; //*
	int sy; //*
	int gx; //*
	int gy; //*
	unsigned short ns; //*
	unsigned short dt; //*
};

typedef struct aperture aperture_t;

struct aperture {
    float ap_m, ap_h, ap_t, len;
};

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

#define MAX(a,b) (((a)>(b))?(a):(b))

static float get_scalco(__global su_trace_t *tr)
{
	if (tr->scalco == 0)
		return 1;
	if (tr->scalco > 0)
		return tr->scalco;
	return 1.0f / tr->scalco;
}

void su_get_midpoint(__global su_trace_t *tr, float *mx, float *my)
{
	float s = get_scalco(tr);
	*mx = s * (tr->gx + tr->sx) * 0.5;
	*my = s * (tr->gy + tr->sy) * 0.5;
}

void su_get_halfoffset(__global su_trace_t *tr, float *hx, float *hy)
{
	float s = get_scalco(tr);
	*hx = s * (tr->gx - tr->sx) * 0.5;
	*hy = s * (tr->gy - tr->sy) * 0.5;
}

float semblance_2d(__global aperture_t *ap, __global su_trace_t *traces_s,
        __global float *data, float A, float B, float C, float D, float E,
        float t0, float m0, float h0,
        float *stack)
{
    __global su_trace_t *tr = &traces_s[0];
    float dt = (float) tr->dt / 1000000;
    float idt = 1 / dt;

    int tau = MAX((int)(ap->ap_t * idt), 0);
    int w = 2 * tau + 1;

    float num[10];
    float den[10];
    for (int i = 0; i < w; i++) {
      num[i] = 0;
    }
    for (int i = 0; i < w; i++) {
      den[i] = 0;
    }
    int M = 0, skip = 0;
    float _stack = 0;

    for (int i = 0; i < ap->len; i++) {
        tr = &traces_s[i];

        float mx, my, hx, hy;
        su_get_midpoint(tr, &mx, &my);
        su_get_halfoffset(tr, &hx, &hy);

        float t = time_2d(A, B, C, D, E, t0, m0, my, h0, hy);
        int it = (int)(t * idt);

        if (it - tau >= 0 && it + tau < tr->ns) {
           for (int j = 0; j < w; j++) {
                int k = it + j - tau;
                float v = interpol_linear(k, k+1,
                        data[i*2052 + k], data[i*2052 + k+1],
                        t*idt + j - tau);
                num[j] += v;
                den[j] += v*v;
                _stack += v;
            }
            M++;
        } else if (++skip == 2) {
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


__kernel void compute_max(__global aperture_t *ap, __global su_trace_t *traces_s,
                          __global float *data, float t0, float m0, float h0, 
                          const float ns[2][5], const int np[5], __global float *results)
{
  int ia = get_global_id(0);
  int ib = get_global_id(1);
  int base = ia*(np[1]*7) + (ib*7);

  results[base] = -1;

  float a = ns[0][0] + ((float)ia / (float)np[0])*(ns[1][0]-ns[0][0]);
  float b = ns[0][1] + ((float)ib / (float)np[1])*(ns[1][1]-ns[0][1]);

  for (int ic = 0; ic < np[2]; ic++) {

      float c = ns[0][2] + ((float)ic / (float)np[2])*(ns[1][2]-ns[0][2]);
      for (int id = 0; id < np[3]; id++) {

          float d = ns[0][3] + ((float)id / (float)np[3])*(ns[1][3]-ns[0][3]);
          for (int ie = 0; ie < np[4]; ie++) {

              float e = ns[0][4] + ((float)ie / (float)np[4])*(ns[1][4]-ns[0][4]);
              float st;

              float s = semblance_2d(ap, traces_s, data, a, b, c, d, e, t0, m0, h0, &st);
              
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
}
