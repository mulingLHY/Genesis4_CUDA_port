#include "ControlCUDA.h"

#ifdef GENESIS_USE_CUDA
#include "Genesis4CudaLaunch.h"
#include "Genesis4CudaRuntime.h"
#include "Genesis4CudaBuffer.h"

#include <climits>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

#include <mpi.h>

#include "Field.h"
#include "Genesis4FieldSoA.h"
#include "Beam.h"
#include "Undulator.h"
#include "Control.h"


using namespace std;

namespace {

void pack_field_slice_to_interleaved_gpu(const double* field_re,
                                         const double* field_im,
                                         double* buffer,
                                         int slice,
                                         int ncells)
{
  const int base = slice * ncells;
  g4_parallel_for(ncells,
    [=] GENESIS4_CUDA_DEVICE (int i) noexcept {
      buffer[2 * i    ] = field_re[base + i];
      buffer[2 * i + 1] = field_im[base + i];
    });
}

void unpack_interleaved_to_field_slice_gpu(double* field_re,
                                           double* field_im,
                                           const double* buffer,
                                           int slice,
                                           int ncells)
{
  const int base = slice * ncells;
  g4_parallel_for(ncells,
    [=] GENESIS4_CUDA_DEVICE (int i) noexcept {
      field_re[base + i] = buffer[2 * i    ];
      field_im[base + i] = buffer[2 * i + 1];
    });
}

void zero_field_slice_gpu(double* field_re,
                          double* field_im,
                          int slice,
                          int ncells)
{
  const int base = slice * ncells;
  g4_parallel_for(ncells,
    [=] GENESIS4_CUDA_DEVICE (int i) noexcept {
      field_re[base + i] = 0.0;
      field_im[base + i] = 0.0;
    });
}

} // namespace

ControlCUDA::ControlCUDA()
  : timerun_(false),
    periodic_(false),
    rank_(0),
    mpiSize_(1),
    sample_(1.0),
    deviceWorkCapacity_(0),
    deviceWork_(nullptr),
    pinnedWorkCapacity_(0),
    pinnedWork_(nullptr)
{}

ControlCUDA::~ControlCUDA()
{
  releaseWorkBuffers();
}

void ControlCUDA::releaseWorkBuffers()
{
  if (deviceWork_ != nullptr) {
    cudaFree(deviceWork_);
    deviceWork_ = nullptr;
    deviceWorkCapacity_ = 0;
  }
  if (pinnedWork_ != nullptr) {
    cudaFreeHost(pinnedWork_);
    pinnedWork_ = nullptr;
    pinnedWorkCapacity_ = 0;
  }
}

void ControlCUDA::ensureWorkBuffers(long long needed)
{
  if (deviceWorkCapacity_ < needed) {
    if (deviceWork_ != nullptr) {
      cudaFree(deviceWork_);
      deviceWork_ = nullptr;
    }
    g4_cuda_check(cudaMalloc(reinterpret_cast<void**>(&deviceWork_), sizeof(double) * static_cast<std::size_t>(needed)), "ControlCUDA device buffer allocation");
    deviceWorkCapacity_ = needed;
  }

  if (pinnedWorkCapacity_ < needed) {
    if (pinnedWork_ != nullptr) {
      cudaFreeHost(pinnedWork_);
      pinnedWork_ = nullptr;
    }
    g4_cuda_check(cudaMallocHost(reinterpret_cast<void**>(&pinnedWork_), sizeof(double) * static_cast<std::size_t>(needed)), "ControlCUDA pinned buffer allocation");
    pinnedWorkCapacity_ = needed;
  }
}

void ControlCUDA::init(int inrank, int insize, bool inTime, bool inPeriodic, double inSample)
{
  rank_ = inrank;
  mpiSize_ = insize;
  timerun_ = inTime;
  periodic_ = inPeriodic;
  sample_ = inSample;
}

bool ControlCUDA::applyMarker(Control *control, Beam *beam, vector<Field*>*field, Undulator *und, bool& error_IO)
{
  int marker=und->getMarker();

  if ((marker & 1) != 0){
    beam->unpack_soa_to_beam();
  }
  
  if ((marker & 2) != 0){
    for (int i = 0; i < field->size(); ++i) {
        field->at(i)->unpack_soa_to_field();
    }
  }

  return control->applyMarker(beam, field, und, error_IO);
}

bool ControlCUDA::applySlippage(double slippage, Field *field)
{
  if (timerun_ == false) { return true; }
  if (field == nullptr || field->fieldSoA == nullptr || !field->fieldSoA->initialized) {
    return false;
  }

  // Update scalar slippage on host.  No field data is touched until the same
  // threshold condition used by the original CPU routine is crossed.
  field->accuslip += slippage;

  Genesis4FieldSoA* soa = field->fieldSoA;
  const int nslices = soa->nslice;
  const int ngrid = soa->ngrid;
  if (nslices <= 0 || ngrid <= 0) {
    return true;
  }

  const long long ncells_ll = static_cast<long long>(ngrid) * static_cast<long long>(ngrid);
  if (2 * ncells_ll > INT_MAX) {
    if (rank_ == 0) {
      cout << "Large field mesh size results in request for MPI transfer size exceeding INT_MAX, exiting." << endl;
    }
    MPI_Abort(MPI_COMM_WORLD,1);
  }
  const int ncells = static_cast<int>(ncells_ll);

  // Two interleaved device buffers and two pinned host buffers are enough to preserve
  // the original odd/even MPI exchange semantics:
  //   sendbuf = local outgoing plane, recvbuf = incoming plane from neighbor.
  // MPI is intentionally issued on pinned host buffers instead of managed/device memory.
  const long long needed = 4LL * static_cast<long long>(ncells);
  ensureWorkBuffers(needed);

  double* d_sendbuf = deviceWork_;
  double* d_recvbuf = deviceWork_ + 2LL * static_cast<long long>(ncells);
  double* h_sendbuf = pinnedWork_;
  double* h_recvbuf = pinnedWork_ + 2LL * static_cast<long long>(ncells);
  double* field_re = soa->field_re.data();
  double* field_im = soa->field_im.data();

  int direction = 1;
  while (std::abs(field->accuslip) > (sample_ * 0.8)) {
    if (field->accuslip < 0) { direction = -1; }
    else { direction = 1; }
    field->accuslip -= sample_ * direction;

    int rank_next = rank_ + 1;
    int rank_prev = rank_ - 1;
    if (rank_next >= mpiSize_) { rank_next = 0; }
    if (rank_prev < 0) { rank_prev = mpiSize_ - 1; }

    if (direction < 0) {
      int tmp = rank_next;
      rank_next = rank_prev;
      rank_prev = tmp;
    }

    const int tag = 1;

    int last = (field->first + nslices - 1) % nslices;
    if (direction < 0) {
      last = (last + 1) % nslices;
    }

    MPI_Status status;
    if (mpiSize_ > 1) {
      // Pack one plane only.  This is intentionally much cheaper than unpacking/packing
      // the whole Field::field array in Gencore every step.
      pack_field_slice_to_interleaved_gpu(field_re, field_im, d_sendbuf, last, ncells);
      genesis4_cuda::copy_device_to_host(d_sendbuf, d_sendbuf + 2 * ncells, h_sendbuf);
      g4_cuda_synchronize();

      if ((rank_ % 2) == 0) {
        MPI_Send(h_sendbuf, 2 * ncells, MPI_DOUBLE, rank_next, tag, MPI_COMM_WORLD);
        MPI_Recv(h_recvbuf, 2 * ncells, MPI_DOUBLE, rank_prev, tag, MPI_COMM_WORLD, &status);
        genesis4_cuda::copy_host_to_device(h_recvbuf, h_recvbuf + 2 * ncells, d_recvbuf);
        g4_cuda_synchronize();
        unpack_interleaved_to_field_slice_gpu(field_re, field_im, d_recvbuf, last, ncells);
      } else {
        MPI_Recv(h_recvbuf, 2 * ncells, MPI_DOUBLE, rank_prev, tag, MPI_COMM_WORLD, &status);
        genesis4_cuda::copy_host_to_device(h_recvbuf, h_recvbuf + 2 * ncells, d_recvbuf);
        g4_cuda_synchronize();
        unpack_interleaved_to_field_slice_gpu(field_re, field_im, d_recvbuf, last, ncells);
        g4_cuda_synchronize();
        MPI_Send(h_sendbuf, 2 * ncells, MPI_DOUBLE, rank_next, tag, MPI_COMM_WORLD);
      }
    }

    if (!periodic_) {
      if ((rank_ == 0) && (direction > 0)) {
        zero_field_slice_gpu(field_re, field_im, last, ncells);
      }
      if ((rank_ == (mpiSize_ - 1)) && (direction < 0)) {
        zero_field_slice_gpu(field_re, field_im, last, ncells);
      }
    }

    field->first = last;
    if (direction < 0) {
      field->first = (last + 1) % nslices;
    }
  }
  return true;
}

#endif
