all:
	gcc-5.3.0 -O3 -msse3 -DERT_INTEL --std=c99 -mavx reg.c semblance.c su.c -lm -I. -lOpenCL -fopenmp -o build/reg
