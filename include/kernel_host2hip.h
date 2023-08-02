#ifndef __IRIS_CPU_DSP_INTERFACE_H__
#define __IRIS_CPU_DSP_INTERFACE_H__

#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

static int iris_kernel_idx = 0;

#define ENABLE_IRIS_HOST2HIP_APIS


#ifdef ENABLE_IRIS_HOST2HIP_APIS
#include "iris/iris_host2hip.h"
#define iris_kernel_lock   iris_host2hip_lock
#define iris_kernel_unlock iris_host2hip_unlock
#define iris_kernel iris_host2hip_kernel
#define iris_setarg iris_host2hip_setarg
#define iris_setmem iris_host2hip_setmem
#define iris_launch iris_host2hip_launch
#define HANDLE
#define HANDLETYPE

#endif //ENABLE_IRIS_HOST2HIP_APISa



typedef struct {
        __global double* Y;
        __global double* X;
        __global double* sym;
        __global int Y___size;
        __global int X___size;
        __global int sym___size;
} iris_spiral_kernel_args;

static iris_spiral_kernel_args _iris_spiral_kernel_args;


//int iris_spiral_kernel(HANDLETYPE IRISBlasType, IRISBlasType, IRISBlasType, int32_t, int32_t, int32_t, double, double*, int, int32_t, double*, int, int32_t, double, double*, int, int32_t, int, int);


static int iris_spiral_kernel_setarg(int idx, size_t size, void* value) {

  switch (idx) {

        default: return IRIS_ERROR;
    }
    return IRIS_SUCCESS;
}

static int iris_spiral_kernel_setmem(int idx, void *mem, int size) {

  switch (idx) {

                        case 0: _iris_spiral_kernel_args.Y = (double*)mem; _iris_spiral_kernel_args.Y___size = size; break;
                        case 1: _iris_spiral_kernel_args.X = (double*)mem; _iris_spiral_kernel_args.X___size = size; break;
                        case 2: _iris_spiral_kernel_args.sym = (double*)mem; _iris_spiral_kernel_args.sym___size = size; break;
        default: return IRIS_ERROR;
    }
    return IRIS_SUCCESS;
}


int iris_setarg(int idx, size_t size, void* value) {
  switch (iris_kernel_idx) {

                         case 1: return iris_spiral_kernel_setarg(idx, size, value);

    }
    return IRIS_ERROR;
}


int iris_setmem(int idx, void *mem, int size) {
  switch (iris_kernel_idx) {

                         case 1: return iris_spiral_kernel_setmem(idx, mem, size);

    }
    return IRIS_ERROR;
}


int iris_kernel(const char* name) {
    iris_kernel_lock();

         if (strcmp(name, "iris_spiral_kernel") == 0) {
                 iris_kernel_idx = 1;
                 return IRIS_SUCCESS;
         }

    return IRIS_ERROR;
}

int iris_spiral_kernel_host2hip(
        double *, double *, double *, size_t, size_t);

int iris_launch(int dim, size_t off, size_t ndr) {
        switch(iris_kernel_idx) {

                        case 1: iris_spiral_kernel_host2hip(
                                _iris_spiral_kernel_args.Y,
                                _iris_spiral_kernel_args.X,
                                _iris_spiral_kernel_args.sym,
                                off,
                                ndr
                                );
                                         break;

        }
        iris_kernel_unlock();
        return IRIS_SUCCESS;
}


#ifdef __cplusplus
} /* end of extern "C" */
#endif
#endif //__IRIS_CPU_DSP_INTERFACE_H__