#ifndef SU_H__
#define SU_H__

#include <stdio.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif

#define SU_HEADER_SIZE ((unsigned long)&(((su_trace_t*)0)->data))

typedef struct su_trace su_trace_t;

struct su_trace {
	cl_ushort ns;
	cl_ushort dt;
	cl_short mark;
  cl_short shortpad;
	cl_short unass[14];
	cl_short scalel;
	cl_short scalco;
	cl_short counit;
	cl_short wevel;
	cl_short swevel;
	cl_short sut;
	cl_short gut;
	cl_short sstat;
	cl_short gstat;
	cl_short tstat;
	cl_short laga;
	cl_short lagb;
	cl_short delrt;
	cl_short muts;
	cl_short mute;
	cl_short gain;
	cl_short igc;
	cl_short igi;
	cl_short corr;
	cl_short sfs;
	cl_short sfe;
	cl_short slen;
	cl_short styp;
	cl_short stas;
	cl_short stae;
	cl_short tatyp;
	cl_short afilf;
	cl_short afils;
	cl_short nofilf;
	cl_short nofils;
	cl_short lcf;
	cl_short hcf;
	cl_short lcs;
	cl_short hcs;
	cl_short year;
	cl_short day;
	cl_short hour;
	cl_short minute;
	cl_short sec;
	cl_short timbas;
	cl_short trwf;
	cl_short grnors;
	cl_short grnofr;
	cl_short grnlof;
	cl_short gaps;
	cl_short otrav;
	cl_short trid;
	cl_short nvs;
	cl_short nhs;
	cl_short duse;
	cl_int tracl;
	cl_int tracr;
	cl_int fldr;
	cl_int tracf;
	cl_int ep;
	cl_int cdp;
	cl_int cdpt;
	cl_int offset;
	cl_int gelev;
	cl_int selev;
	cl_int sdepth;
	cl_int gdel;
	cl_int sdel;
	cl_int swdep;
	cl_int gwdep;
	cl_int sx;
	cl_int sy;
	cl_int gx;
	cl_int gy;
	cl_int ntr;
  cl_float d1;
	cl_float f1;
	cl_float d2;
	cl_float f2;
	cl_float ungpow;
	cl_float unscale;
	cl_float *data;
};

/*
 * This function allocates `tr->data' array based on `tr.ns', so must be
 * already set!
 */
void su_init(su_trace_t *tr);

/*
 * Frees internal structure.
 */
void su_free(su_trace_t *tr);

/*
 * Reads a trace header and its data from `file'.
 */
int su_fgettr(FILE *file, su_trace_t *tr);

/*
 * Reads a trace header and its data from `stdin'.
 */
int su_gettr(su_trace_t *tr);

/*
 * Writes a trace header and its data to `file'.
 */
int su_fputtr(FILE *file, su_trace_t *tr);

/*
 * Writes a trace header and its data to `stdout'.
 */
int su_puttr(su_trace_t *tr);

/*
 * Returns the CDP key of `tr'.
 */
int su_get_cdp(su_trace_t *tr);

/*
 * Returns the source position of `tr'.
 */
void su_get_source(su_trace_t *tr, float *sx, float *sy);

/*
 * Returns the receiver position of `tr'.
 */
void su_get_receiver(su_trace_t *tr, float *gx, float *gy);

/*
 * Returns the midpoint of `tr'.
 */
void su_get_midpoint(su_trace_t *tr, float *mx, float *my);

/*
 * Returns the half offset of `tr'.
 */
void su_get_halfoffset(su_trace_t *tr, float *hx, float *hy);

#endif
