#include "FieldSolverFFTCUDA.h"
#include "Field.h"
#include "Beam.h"

#include <cmath>
#include <string>
#include <vector>

#ifdef GENESIS_USE_CUDA
#include "Genesis4CudaLaunch.h"
#include "Genesis4CudaRuntime.h"
#include "Genesis4BeamSoA.h"
#include "Genesis4FieldSoA.h"
#include <cuda_runtime.h>
#endif

#ifdef GENESIS_USE_CUDA
namespace {

void genesis4_fieldsolverfftcuda_abort_if_cufft_failed(cufftResult result,
                                                    const char* where)
{
  if (result != CUFFT_SUCCESS) {
    g4_cuda_abort(std::string("FieldSolverFFTCUDA cuFFT failure in ") + where +
                 ": error code " + std::to_string(static_cast<int>(result)));
  }
}

void genesis4_fieldsolverfftcuda_abort_if_cuda_failed(cudaError_t result,
                                                   const char* where)
{
  if (result != cudaSuccess) {
    g4_cuda_abort(std::string("FieldSolverFFTCUDA CUDA failure in ") + where +
                 ": " + cudaGetErrorString(result));
  }
}

GENESIS4_CUDA_HOST_DEVICE GENESIS4_CUDA_FORCE_INLINE
cufftDoubleComplex genesis4_fft_make_complex(double re, double im) noexcept
{
  cufftDoubleComplex v;
  v.x = re;
  v.y = im;
  return v;
}

GENESIS4_CUDA_HOST_DEVICE GENESIS4_CUDA_FORCE_INLINE
cufftDoubleComplex genesis4_fft_mul(cufftDoubleComplex a,
                                     cufftDoubleComplex b) noexcept
{
  return genesis4_fft_make_complex(a.x * b.x - a.y * b.y,
                                   a.x * b.y + a.y * b.x);
}

void genesis4_fieldsolverfftcuda_pack_field_gpu(int total_cells,
                                             const double* field_re,
                                             const double* field_im,
                                             cufftDoubleComplex* field_fft)
{
  g4_parallel_for(total_cells,
    [=] GENESIS4_CUDA_DEVICE (int i) noexcept {
      field_fft[i].x = field_re[i];
      field_fft[i].y = field_im[i];
    });
}

void genesis4_fieldsolverfftcuda_zero_source_gpu(int total_cells,
                                              cufftDoubleComplex* source_fft)
{
  g4_parallel_for(total_cells,
    [=] GENESIS4_CUDA_DEVICE (int i) noexcept {
      source_fft[i].x = 0.0;
      source_fft[i].y = 0.0;
    });
}

void genesis4_fieldsolverfftcuda_build_source_gpu(int total_particles,
                                               int nslice,
                                               int ngrid,
                                               int first,
                                               int harm,
                                               double gridmax,
                                               double dgrid,
                                               double ax,
                                               double ay,
                                               double kx,
                                               double ky,
                                               double gradx,
                                               double grady,
                                               const double* slice_scl,
                                               const double* x,
                                               const double* y,
                                               const double* gamma,
                                               const double* theta,
                                               const int* slice_id,
                                               cufftDoubleComplex* source_fft)
{
  const int plane = ngrid * ngrid;

  g4_parallel_for(total_particles,
    [=] GENESIS4_CUDA_DEVICE (int ip) noexcept {
      const int beam_slice = slice_id[ip];
      if (beam_slice < 0 || beam_slice >= nslice) { return; }

      const double scl = slice_scl[beam_slice];
      if (scl == 0.0) { return; }

      const double xp = x[ip];
      const double yp = y[ip];
      if (!(xp > -gridmax && xp < gridmax && yp > -gridmax && yp < gridmax)) {
        return;
      }

      const double wx_raw = (xp + gridmax) / dgrid;
      const double wy_raw = (yp + gridmax) / dgrid;
      const int ix = static_cast<int>(floor(wx_raw));
      const int iy = static_cast<int>(floor(wy_raw));

      if (ix < 0 || ix >= ngrid - 1 || iy < 0 || iy >= ngrid - 1) {
        return;
      }

      const double wx = 1.0 + static_cast<double>(ix) - wx_raw;
      const double wy = 1.0 + static_cast<double>(iy) - wy_raw;

      const double dx = xp - ax;
      const double dy = yp - ay;
      const double faw2 = 1.0 + kx * dx * dx + ky * dy * dy
                        + 2.0 * (gradx * dx + grady * dy);
      const double part = sqrt(faw2) * scl / gamma[ip];
      const double phase = static_cast<double>(harm) * theta[ip];
      const double cpart_re = sin(phase) * part;
      const double cpart_im = cos(phase) * part;

      const int target_slice = (beam_slice + first) % nslice;
      const int base = target_slice * plane + ix + iy * ngrid;

      const double w00 = wx * wy;
      const double w10 = (1.0 - wx) * wy;
      const double w01 = wx * (1.0 - wy);
      const double w11 = (1.0 - wx) * (1.0 - wy);

      atomicAdd(&source_fft[base].x, cpart_re * w00);
      atomicAdd(&source_fft[base].y, cpart_im * w00);

      atomicAdd(&source_fft[base + 1].x, cpart_re * w10);
      atomicAdd(&source_fft[base + 1].y, cpart_im * w10);

      atomicAdd(&source_fft[base + ngrid].x, cpart_re * w01);
      atomicAdd(&source_fft[base + ngrid].y, cpart_im * w01);

      atomicAdd(&source_fft[base + ngrid + 1].x, cpart_re * w11);
      atomicAdd(&source_fft[base + ngrid + 1].y, cpart_im * w11);
    });
}

void genesis4_fieldsolverfftcuda_apply_propagator_gpu(int total_cells,
                                                   int ngrid,
                                                   double delz,
                                                   bool do_filter,
                                                   const double* k2_im,
                                                   const double* sigmoid,
                                                   cufftDoubleComplex* field_fft,
                                                   const cufftDoubleComplex* source_fft)
{
  const int plane = ngrid * ngrid;

  g4_parallel_for(total_cells,
    [=] GENESIS4_CUDA_DEVICE (int g) noexcept {
      const int local = g % plane;
      const double phase = k2_im[local] * delz;
      const cufftDoubleComplex prop = genesis4_fft_make_complex(cos(phase), sin(phase));
      const cufftDoubleComplex uf = field_fft[g];
      cufftDoubleComplex out = genesis4_fft_mul(uf, prop);

      double sf_re = source_fft[g].x;
      double sf_im = source_fft[g].y;
      if (do_filter) {
        const double filt = sigmoid[local];
        sf_re *= filt;
        sf_im *= filt;
      }

      out.x += 2.0 * sf_re;
      out.y += 2.0 * sf_im;
      field_fft[g] = out;
    });
}

void genesis4_fieldsolverfftcuda_unpack_field_gpu(int total_cells,
                                               double norm,
                                               const cufftDoubleComplex* field_fft,
                                               double* field_re,
                                               double* field_im)
{
  g4_parallel_for(total_cells,
    [=] GENESIS4_CUDA_DEVICE (int i) noexcept {
      field_re[i] = field_fft[i].x * norm;
      field_im[i] = field_fft[i].y * norm;
    });
}

} // namespace
#endif

FieldSolverFFTCUDA::~FieldSolverFFTCUDA()
{
#ifdef GENESIS_USE_CUDA
  destroy_gpu_plans();
#endif
}

void FieldSolverFFTCUDA::initSourceFilter(double xc_in, double yc_in, double sig_in, bool do_filter) {
  xc = xc_in;
  yc = yc_in;
  sig = sig_in;
  doFilter_ = do_filter;
  if (sig <= 0 || xc <= 0 || yc <= 0) {
    doFilter_ = false;
  }
}

#ifdef GENESIS_USE_CUDA
void FieldSolverFFTCUDA::destroy_gpu_plans()
{
  if (gpu_has_plan) {
    cufftDestroy(gpu_forward_plan);
    cufftDestroy(gpu_inverse_plan);
    gpu_forward_plan = 0;
    gpu_inverse_plan = 0;
    gpu_plan_nslice = 0;
    gpu_plan_ngrid = 0;
    gpu_has_plan = false;
  }
}

void FieldSolverFFTCUDA::ensure_gpu_plan(int nslice)
{
  const int ngrid_i = static_cast<int>(ngrid);
  if (gpu_has_plan && gpu_plan_nslice == nslice && gpu_plan_ngrid == ngrid_i) {
    return;
  }

  destroy_gpu_plans();

  int dims[2] = {ngrid_i, ngrid_i};
  const int plane = ngrid_i * ngrid_i;
  genesis4_fieldsolverfftcuda_abort_if_cufft_failed(
      cufftPlanMany(&gpu_forward_plan, 2, dims,
                    nullptr, 1, plane,
                    nullptr, 1, plane,
                    CUFFT_Z2Z, nslice),
      "cufftPlanMany forward");
  genesis4_fieldsolverfftcuda_abort_if_cufft_failed(
      cufftPlanMany(&gpu_inverse_plan, 2, dims,
                    nullptr, 1, plane,
                    nullptr, 1, plane,
                    CUFFT_Z2Z, nslice),
      "cufftPlanMany inverse");

  genesis4_fieldsolverfftcuda_abort_if_cufft_failed(
      cufftSetStream(gpu_forward_plan, g4_cuda_stream()),
      "cufftSetStream forward");
  genesis4_fieldsolverfftcuda_abort_if_cufft_failed(
      cufftSetStream(gpu_inverse_plan, g4_cuda_stream()),
      "cufftSetStream inverse");

  gpu_plan_nslice = nslice;
  gpu_plan_ngrid = ngrid_i;
  gpu_has_plan = true;
}

void FieldSolverFFTCUDA::init(double delz,double dgrid, double xks, unsigned int ngrid_in)
{
  delz_save = delz;

  if (gpu_coefficients_initialized && ngrid == ngrid_in) {
    return;
  }

  destroy_gpu_plans();

  ks = xks;
  ngrid = ngrid_in;
  dk = 4.*asin(1.)/(static_cast<double>(ngrid)*dgrid);

  const int ngrid_i = static_cast<int>(ngrid);
  const int plane = ngrid_i * ngrid_i;
  std::vector<double> h_k2_im(plane);
  std::vector<double> h_sigmoid(plane);

  const bool filter_shape_valid = (xc > 0.0 && yc > 0.0 && sig > 0.0);
  if (!filter_shape_valid) {
    doFilter_ = false;
  }

  double shift=-0.5*static_cast<double> (ngrid-1);

  for (int iy=0;iy<ngrid_i;iy++) {
    double dy=static_cast<double>(iy)+shift;
    double y = filter_shape_valid ? dy / static_cast<double>(ngrid) /yc : 0.0;
    for (int ix=0;ix<ngrid_i;ix++) {
      double dx=static_cast<double>(ix)+shift;
      double x = filter_shape_valid ? dx / static_cast<double>(ngrid) /xc : 0.0;
      int iiy=(iy+(ngrid_i+1)/2) % ngrid_i;
      int iix=(ix+(ngrid_i+1)/2) % ngrid_i;
      int ii=iiy*ngrid_i+iix;
      h_k2_im[ii] = -(dx*dx+dy*dy)*dk*dk/2./xks;
      if (filter_shape_valid) {
        double r = (sqrt(x * x + y * y) - 1) / sig;
        h_sigmoid[ii] = 1. / (1 + exp(r));
      } else {
        h_sigmoid[ii] = 1.0;
      }
    }
  }

  d_k2_im.copy_from_host(h_k2_im.data(), h_k2_im.size());
  d_sigmoid.copy_from_host(h_sigmoid.data(), h_sigmoid.size());
  g4_cuda_synchronize();

  gpu_coefficients_initialized = true;
}

void FieldSolverFFTCUDA::advance(double delz, Field *field, Beam *beam, Undulator *und)
{
  if (field == nullptr || beam == nullptr || und == nullptr) {
    g4_cuda_abort("FieldSolverFFTCUDA::advance_gpu received a null field/beam/undulator pointer");
  }

  Genesis4BeamSoA* bsoa = beam->beamSoA;
  Genesis4FieldSoA* fsoa = field->fieldSoA;

  if (bsoa == nullptr || !bsoa->initialized ||
      fsoa == nullptr || !fsoa->initialized ||
      ngrid == 0 || field->ngrid != static_cast<int>(ngrid)) {
    g4_cuda_abort("FieldSolverFFTCUDA::advance_gpu requires initialized BeamSoA/FieldSoA and matching ngrid");
  }

  const int nslice = fsoa->nslice;
  const int ngrid_i = static_cast<int>(ngrid);
  const int plane = ngrid_i * ngrid_i;
  const int total_cells = nslice * plane;
  const int total_particles = bsoa->total_particles;

  if (nslice <= 0 || plane <= 0 || total_cells <= 0 ||
      bsoa->nslice != nslice ||
      fsoa->ngrid != ngrid_i ||
      static_cast<int>(fsoa->field_re.size()) < total_cells ||
      static_cast<int>(fsoa->field_im.size()) < total_cells ||
      static_cast<int>(bsoa->x.size()) < total_particles ||
      static_cast<int>(bsoa->y.size()) < total_particles ||
      static_cast<int>(bsoa->gamma.size()) < total_particles ||
      static_cast<int>(bsoa->theta.size()) < total_particles ||
      static_cast<int>(bsoa->slice_id.size()) < total_particles ||
      static_cast<int>(bsoa->slice_offsets.size()) < nslice + 1 ||
      static_cast<int>(beam->current.size()) < nslice ||
      static_cast<int>(d_k2_im.size()) < plane ||
      static_cast<int>(d_sigmoid.size()) < plane) {
    g4_cuda_abort("FieldSolverFFTCUDA::advance_gpu found inconsistent SoA dimensions");
  }

  if (gpu_alloc_total_cells < total_cells) {
    d_field_fft.resize(total_cells);
    d_source_fft.resize(total_cells);
    gpu_alloc_total_cells = total_cells;
  }
  if (static_cast<int>(d_slice_scl.size()) < nslice) {
    d_slice_scl.resize(nslice);
  }

  ensure_gpu_plan(nslice);

  std::vector<int> h_slice_offsets(nslice + 1);
  genesis4_cuda::copy_device_to_host(bsoa->slice_offsets.begin(),
                                     bsoa->slice_offsets.begin() + nslice + 1,
                                     h_slice_offsets.begin());
  g4_cuda_synchronize();

  const int harm = field->getHarm();
  const bool do_source = und->inUndulator() && field->isEnabled() && ((harm % 2) == 1);

  std::vector<double> h_slice_scl(nslice);
  for (int is = 0; is < nslice; ++is) {
    double scl = 0.0;
    if (do_source) {
      const int npar = h_slice_offsets[is + 1] - h_slice_offsets[is];
      if (npar > 0) {
        scl = und->fc(harm) * vacimp * beam->current[is] * field->xks * delz;
        scl /= 4.0 * eev * static_cast<double>(npar) * field->dgrid * field->dgrid;
      }
    }
    h_slice_scl[is] = scl;
  }
  d_slice_scl.copy_from_host(h_slice_scl.data(), h_slice_scl.size());

  double ax = 0.0;
  double ay = 0.0;
  double kx = 0.0;
  double ky = 0.0;
  double gradx = 0.0;
  double grady = 0.0;
  const int istep = und->getStep();
  if (do_source && istep >= 0 && istep < static_cast<int>(und->ax.size())) {
    ax = und->ax[istep];
    ay = und->ay[istep];
    kx = und->kx[istep];
    ky = und->ky[istep];
    gradx = und->gradx[istep];
    grady = und->grady[istep];
  }

  genesis4_fieldsolverfftcuda_pack_field_gpu(total_cells,
                                          fsoa->field_re.data(),
                                          fsoa->field_im.data(),
                                          d_field_fft.data());
  genesis4_fieldsolverfftcuda_zero_source_gpu(total_cells, d_source_fft.data());

  if (do_source && total_particles > 0) {
    genesis4_fieldsolverfftcuda_build_source_gpu(total_particles,
                                             nslice,
                                             ngrid_i,
                                             field->first,
                                             harm,
                                             field->gridmax,
                                             field->dgrid,
                                             ax,
                                             ay,
                                             kx,
                                             ky,
                                             gradx,
                                             grady,
                                             d_slice_scl.data(),
                                             bsoa->x.data(),
                                             bsoa->y.data(),
                                             bsoa->gamma.data(),
                                             bsoa->theta.data(),
                                             bsoa->slice_id.data(),
                                             d_source_fft.data());
  }

  genesis4_fieldsolverfftcuda_abort_if_cuda_failed(cudaGetLastError(), "source construction");

  genesis4_fieldsolverfftcuda_abort_if_cufft_failed(
      cufftExecZ2Z(gpu_forward_plan,
                   d_field_fft.data(),
                   d_field_fft.data(),
                   CUFFT_FORWARD),
      "cufftExecZ2Z field forward");

  if (do_source && total_particles > 0) {
    genesis4_fieldsolverfftcuda_abort_if_cufft_failed(
        cufftExecZ2Z(gpu_forward_plan,
                     d_source_fft.data(),
                     d_source_fft.data(),
                     CUFFT_FORWARD),
        "cufftExecZ2Z source forward");
  }

  genesis4_fieldsolverfftcuda_apply_propagator_gpu(total_cells,
                                                ngrid_i,
                                                delz_save,
                                                doFilter_,
                                                d_k2_im.data(),
                                                d_sigmoid.data(),
                                                d_field_fft.data(),
                                                d_source_fft.data());

  genesis4_fieldsolverfftcuda_abort_if_cufft_failed(
      cufftExecZ2Z(gpu_inverse_plan,
                   d_field_fft.data(),
                   d_field_fft.data(),
                   CUFFT_INVERSE),
      "cufftExecZ2Z inverse");

  const double norm = 1./static_cast<double>(plane);
  genesis4_fieldsolverfftcuda_unpack_field_gpu(total_cells,
                                            norm,
                                            d_field_fft.data(),
                                            fsoa->field_re.data(),
                                            fsoa->field_im.data());
  genesis4_fieldsolverfftcuda_abort_if_cuda_failed(cudaGetLastError(), "field unpack");
}
#endif
