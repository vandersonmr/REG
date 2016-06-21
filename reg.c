#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <utils.h>
#include <semblance.h>
#include <su.h>
#include <errno.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif

#include "err_code.h"

#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

#define MAX_SOURCE_SIZE (0x100000)

#include <sys/time.h>

double mysecond() {
  struct timeval tp;
  struct timezone tzp;
  gettimeofday(&tp ,&tzp);
  return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
}

int main(int argc, char *argv[])
{
    double start_total = mysecond();
    double start_init = mysecond();
    int i;

    cl_int          err;               // error code returned from OpenCL calls
    size_t global;                  // global domain size

    cl_device_id     device_id;     // compute device id
    cl_context       context;       // compute context
    cl_command_queue commands;      // compute command queue
    cl_program       program;       // compute program
    cl_kernel        kernel;       // compute kernel

    cl_uint numPlatforms;

    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (numPlatforms == 0)
    {
        printf("Found 0 platforms!\n");
        return EXIT_FAILURE;
    }

    cl_platform_id Platform[numPlatforms];
    err = clGetPlatformIDs(numPlatforms, Platform, NULL);

    for (i = 0; i < numPlatforms; i++)
    {
        err = clGetDeviceIDs(Platform[i],  CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
        if (err == CL_SUCCESS)
        {
            break;
        }
    }

    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

    commands = clCreateCommandQueue(context, device_id, 0, &err);

    FILE *kernel_fp;
    const char file_name[] = "./kernel.cl";
    kernel_fp = fopen(file_name, "r");
    size_t source_size;
    char *source_str;

    if (!kernel_fp) 
    {
      printf("Kernel File not found!\n");
      return 0;
    }

    source_str = (char*) malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, kernel_fp);
    fclose(kernel_fp);

    program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &err);

    err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    size_t log_size;
    err = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, 
                                                              NULL, &log_size);
    char* build_log = (char* )malloc((log_size+1));
    err = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 
                                                    log_size, build_log, NULL);
    build_log[log_size] = '\0';
    printf("\n--- Build log ---\n ");
    printf("%s\n\n", build_log);
    free(build_log);

    /* Create OpenCL Kernel */
    kernel = clCreateKernel(program, "compute_max", &err);
    /* -----------------------------------------------------------------------*/

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
    float ps[2][5];
    int np[5];

    /* p0 is where the search starts, p1 is where the search ends and np is the 
     * number of points in between p0 and p1 to do the search */   
    for (i = 0; i < 5; i++) {
        ps[0][i] = atof(argv[5 + 3*i]);
        ps[1][i] = atof(argv[5 + 3*i + 1]);
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

    aperture_t ap;
    ap.ap_m = 0;
    ap.ap_h = 0;
    ap.ap_t = tau;
    ap.len = traces.len;
    su_trace_c_t traces_s[traces.len];
    float *data = (float*) malloc(traces.len*2502*sizeof(float));

    for (int i = 0; i < traces.len; i++) {
        su_trace_t tmp = vector_get(traces, i);
        traces_s[i].scalco = tmp.scalco;
        traces_s[i].sx = tmp.sx;
        traces_s[i].sy = tmp.sy;
        traces_s[i].gx = tmp.gx;
        traces_s[i].gy = tmp.gy;
        traces_s[i].ns = tmp.ns;
        traces_s[i].dt = tmp.dt;
        for (int j = 0; j < 2502; j++) { 
          data[i*2052 + j] = tmp.data[j];
        }
    }

	printf("%d\n", np[0]);
    size_t global_work_size[3] = {np[0], np[1], np[2]};
    size_t local_work_size[3] = {5, 5, 5};

    cl_mem d_ps  = clCreateBuffer(context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  sizeof(float)*5*2, ps, &err);
    checkError(err, "Creating buffer d_ps");
    cl_mem d_np  = clCreateBuffer(context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  sizeof(int)*5, np, &err);
    checkError(err, "Creating buffer d_np");
    cl_mem d_ap  = clCreateBuffer(context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  sizeof(aperture_t), &ap, &err);
    checkError(err, "creating buffer d_ap");
    cl_mem d_traces_s  = clCreateBuffer(context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  sizeof(su_trace_c_t)*ap.len, &traces_s, &err);
    checkError(err, "creating buffer d_traces");
    cl_mem d_data  = clCreateBuffer(context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  sizeof(float)*ap.len*2052, data, &err);
    checkError(err, "creating buffer d_data");


    cl_mem d_results = clCreateBuffer(context, CL_MEM_READ_WRITE, 
				       sizeof(float)*7*np[0]*np[1]*np[2], NULL, &err);
    checkError(err, "Creating buffer d_results");

    err  = clSetKernelArg(kernel, 0, sizeof(d_ap), &d_ap);
    err |= clSetKernelArg(kernel, 1, sizeof(d_traces_s), &d_traces_s);
    err |= clSetKernelArg(kernel, 2, sizeof(d_data), &d_data);
    err |= clSetKernelArg(kernel, 3, sizeof(float), &t0);
    err |= clSetKernelArg(kernel, 4, sizeof(float), &m0);
    err |= clSetKernelArg(kernel, 5, sizeof(float), &h0);
    err |= clSetKernelArg(kernel, 6, sizeof(d_ps), &d_ps);
    err |= clSetKernelArg(kernel, 7, sizeof(d_np), &d_np);
    err |= clSetKernelArg(kernel, 8, sizeof(d_results), &d_results);
    checkError(err, "Setting kernel arguments"); 

    double end_init = mysecond();


    double start_run = mysecond();
    err = clEnqueueNDRangeKernel(commands, kernel, 3, NULL, global_work_size, 
                                                local_work_size, 0, NULL, NULL);
    checkError(err, "Enqueue Range Kernel");

    float results[7*np[0]*np[1]*np[2]];
    err = clEnqueueReadBuffer(commands, d_results, CL_TRUE, 0, 
                              sizeof(float) * 7 * np[0] * np[1]*np[2], results, 0, NULL, NULL );
    checkError(err, "Reading back results");
    double end_run = mysecond();

    float a, b, c, d, e, sem, stack;
    float ssmax = -1.0;
    stack = 0;
    for (int ia = 0; ia < np[0]; ia++) {
      for (int ib = 0; ib < np[1]; ib++) {
      for (int ic = 0; ic < np[2]; ic++) {
        int base = ic*(np[2]*np[1]*7) + ia*(np[1]*7) + (ib*7);
        if (results[base] > ssmax) {
            a = results[base+2];
            b = results[base+3];
            c = results[base+4];
            d = results[base+5];
            e = results[base+6];
            sem = results[base];
            stack = results[base+1];
            ssmax = results[base];
        }
	}
      }
    }

    printf("A=%g\n", a);
    printf("B=%g\n", b);
    printf("C=%g\n", c);
    printf("D=%g\n", d);
    printf("E=%g\n", e);
    printf("Stack=%g\n", stack);
    printf("Semblance=%g\n", sem);
    printf("\n"); 
    
    double end_total = mysecond();

    printf("Time init: %lf Time run: %lf Total time: %lf\n", 
            end_init - start_init, end_run - start_run, end_total - start_total);
    printf("\n");

    return 0;
}
