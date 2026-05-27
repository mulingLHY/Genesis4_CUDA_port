#ifndef __GENESIS_FIELDSOLVERADICUDA__
#define __GENESIS_FIELDSOLVERADICUDA__

#include <complex>

#ifdef GENESIS_USE_CUDA
#include "Genesis4CudaBuffer.h"
#endif

class Field;
class Beam;

#include "Undulator.h"
#include "FieldSolver.h"

using namespace std;

class FieldSolverADICUDA : public FieldSolver{
 public:
  ~FieldSolverADICUDA();

  void init(double,double,double,unsigned int) override;
  void advance(double, Field *, Beam *, Undulator *) override;
  void initSourceFilter(double,double,double,bool) override;

 private:
  unsigned int ngrid {0};
  double delz_save {0};
  complex<double> cstep;


  // GPU coefficient/work arrays.  Complex numbers are stored as split real/imag
  // arrays so device kernels do not depend on std::complex device support.
  CudaDeviceBuffer<double> d_c_re, d_c_im;
  CudaDeviceBuffer<double> d_cbet_re, d_cbet_im;
  CudaDeviceBuffer<double> d_cwet_re, d_cwet_im;

  // Precomputed PCR elimination factors for the constant ADI tridiagonal
  // matrix.  Size = d_pcr_num_stages * ngrid for alpha/beta and ngrid for
  // final diagonal.  These keep each field line from recomputing identical
  // complex divisions in every GPU block.
  CudaDeviceBuffer<double> d_pcr_alpha_re, d_pcr_alpha_im;
  CudaDeviceBuffer<double> d_pcr_beta_re, d_pcr_beta_im;
  CudaDeviceBuffer<double> d_pcr_b_re, d_pcr_b_im;

  // Per-slice ADI RHS arrays.  Size = nslice * ngrid * ngrid.
  CudaDeviceBuffer<double> d_r_re, d_r_im;

  // Source grid arrays.  Size = nslice * ngrid * ngrid.  Particle source is
  // deposited once per advance and reused by both ADI half-steps.
  CudaDeviceBuffer<double> d_source_re, d_source_im;

  // Source current and scale per beam slice.
  CudaDeviceBuffer<double> d_current;
  CudaDeviceBuffer<double> d_slice_scl;

  int d_alloc_nslice {0};
  int d_alloc_cells_per_slice {0};
  int d_pcr_factor_ngrid {0};
  int d_pcr_num_stages {0};
};

inline void FieldSolverADICUDA::initSourceFilter(double, double, double, bool) {}

#endif
