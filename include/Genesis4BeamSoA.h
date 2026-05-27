#ifndef GENESIS4BEAMSOA_H_
#define GENESIS4BEAMSOA_H_

#ifdef GENESIS_USE_CUDA
#include "Genesis4CudaBuffer.h"

struct Genesis4BeamSoA{
    CudaDeviceBuffer<double> x;
    CudaDeviceBuffer<double> y;
    CudaDeviceBuffer<double> px;
    CudaDeviceBuffer<double> py;
    CudaDeviceBuffer<double> gamma;
    CudaDeviceBuffer<double> theta;

    CudaDeviceBuffer<int> slice_id;
    CudaDeviceBuffer<int> slice_offsets;

    int nslice = 0;
    int total_particles = 0;
    bool initialized = false;
};
#endif // GENESIS_USE_CUDA

#endif // GENESIS4BEAMSOA_H_
