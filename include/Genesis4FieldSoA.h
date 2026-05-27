#ifndef GENESIS4FIELDSOA_H_
#define GENESIS4FIELDSOA_H_

#ifdef GENESIS_USE_CUDA
#include "Genesis4CudaBuffer.h"

struct Genesis4FieldSoA {
    // Flattened field data: real and imaginary parts separated
    // Total cells = nslice * ngrid * ngrid
    CudaDeviceBuffer<double> field_re;
    CudaDeviceBuffer<double> field_im;

    int nslice = 0;
    int ngrid = 0;
    int total_cells = 0;
    bool initialized = false;
};
#endif // GENESIS_USE_CUDA

#endif // GENESIS4FIELDSOA_H_
