#ifndef __GENESIS_FIELDSOLVERFFTCUDA__
#define __GENESIS_FIELDSOLVERFFTCUDA__


#ifdef GENESIS_USE_CUDA
#include "Genesis4CudaBuffer.h"
#include <cufft.h>
#endif

class Field;
class Beam;

#include "Undulator.h"
#include "FieldSolver.h"

using namespace std;

class FieldSolverFFTCUDA : public FieldSolver{
 public:
  ~FieldSolverFFTCUDA();

  void init(double,double,double,unsigned int) override;
  void advance(double, Field *, Beam *, Undulator *) override;
  void initSourceFilter(double,double,double,bool) override;

 private:
  unsigned int ngrid {0};
  double delz_save {0};
  double ks {1};
  double dk {1};
  double xc {1};
  double yc {1};
  double sig {1};
  bool doFilter_ {false};

#ifdef GENESIS_USE_CUDA
  void destroy_gpu_plans();
  void ensure_gpu_plan(int);

  CudaDeviceBuffer<cufftDoubleComplex> d_field_fft;
  CudaDeviceBuffer<cufftDoubleComplex> d_source_fft;
  CudaDeviceBuffer<double> d_k2_im;
  CudaDeviceBuffer<double> d_sigmoid;
  CudaDeviceBuffer<double> d_slice_scl;

  cufftHandle gpu_forward_plan {0};
  cufftHandle gpu_inverse_plan {0};
  int gpu_plan_nslice {0};
  int gpu_plan_ngrid {0};
  int gpu_alloc_total_cells {0};
  bool gpu_has_plan {false};
  bool gpu_coefficients_initialized {false};
#endif
};

#endif
