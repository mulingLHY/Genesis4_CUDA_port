#include "FieldSolverADICUDA.h"
#include "Genesis4CudaLaunch.h"
#include "Genesis4CudaRuntime.h"
#include "Field.h"
#include "Beam.h"

#include <cmath>
#include <cstddef>
#include <vector>

#include "Genesis4BeamSoA.h"
#include "Genesis4FieldSoA.h"
#include <cuda_runtime.h>
#include <thrust/complex.h>


#ifndef GENESIS_FIELD_ADI_PCR_MAX_NGRID
#define GENESIS_FIELD_ADI_PCR_MAX_NGRID 512
#endif


namespace {

using CudaComplex = thrust::complex<double>;

struct AdiCoefficientsView {
  const double* c_re;
  const double* c_im;
  const double* cbet_re;
  const double* cbet_im;
  const double* cwet_re;
  const double* cwet_im;
};

struct AdiFieldView {
  double* r_re;
  double* r_im;
  double* field_re;
  double* field_im;
};

struct PcrFactorView {
  int num_stages;
  const double* alpha_re;
  const double* alpha_im;
  const double* beta_re;
  const double* beta_im;
  const double* b_re;
  const double* b_im;
};

struct SourceParams {
  int nslice;
  int ngrid;
  int first;
  int harm;
  double gridmax;
  double dgrid;
  double ax;
  double ay;
  double kx;
  double ky;
  double gradx;
  double grady;
};

struct SourceParticleView {
  const double* x;
  const double* y;
  const double* gamma;
  const double* theta;
  const int* slice_id;
};

void adi_compute_slice_scl(int nslice,
                           const int* slice_offsets,
                           const double* current,
                           double fc_harm,
                           double xks,
                           double dgrid,
                           double delz,
                           double vacimp_value,
                           double eev_value,
                           double* slice_scl)
{
  g4_parallel_for(nslice,
    [=] GENESIS4_CUDA_DEVICE (int is) noexcept {
      double scl = 0.0;
      const int npar = slice_offsets[is + 1] - slice_offsets[is];
      if (npar > 0) {
        scl = fc_harm * vacimp_value * current[is] * xks * delz;
        scl /= 4.0 * eev_value * static_cast<double>(npar) * dgrid * dgrid;
      }
      slice_scl[is] = scl;
    });
}

void adi_zero_source_grid(int total_cells,
                          double* source_re,
                          double* source_im)
{
  g4_parallel_for(total_cells,
    [=] GENESIS4_CUDA_DEVICE (int g) noexcept {
      source_re[g] = 0.0;
      source_im[g] = 0.0;
    });
}

void adi_deposit_source_atomic(int total_particles,
                               SourceParams params,
                               SourceParticleView particles,
                               const double* slice_scl,
                               double* target_re,
                               double* target_im)
{
  const int plane = params.ngrid * params.ngrid;

  g4_parallel_for(total_particles,
    [=] GENESIS4_CUDA_DEVICE (int ip) noexcept {
      const int beam_slice = particles.slice_id[ip];
      if (beam_slice < 0 || beam_slice >= params.nslice) { return; }

      const double scl = slice_scl[beam_slice];
      if (scl == 0.0) { return; }

      const double xp = particles.x[ip];
      const double yp = particles.y[ip];
      if (!(xp > -params.gridmax && xp < params.gridmax &&
            yp > -params.gridmax && yp < params.gridmax)) {
        return;
      }

      const double wx_raw = (xp + params.gridmax) / params.dgrid;
      const double wy_raw = (yp + params.gridmax) / params.dgrid;
      const int ix = static_cast<int>(floor(wx_raw));
      const int iy = static_cast<int>(floor(wy_raw));

      if (ix < 0 || ix >= params.ngrid - 1 || iy < 0 || iy >= params.ngrid - 1) {
        return;
      }

      const double wx = 1.0 + static_cast<double>(ix) - wx_raw;
      const double wy = 1.0 + static_cast<double>(iy) - wy_raw;

      const double dx = xp - params.ax;
      const double dy = yp - params.ay;
      const double faw2 = 1.0 + params.kx * dx * dx + params.ky * dy * dy
                        + 2.0 * (params.gradx * dx + params.grady * dy);
      const double part = sqrt(faw2) * scl / particles.gamma[ip];
      const double phase = static_cast<double>(params.harm) * particles.theta[ip];
      const double cpart_re = sin(phase) * part;
      const double cpart_im = cos(phase) * part;

      const int target_slice = (beam_slice + params.first) % params.nslice;
      const int base = target_slice * plane + ix + iy * params.ngrid;

      const double w00 = wx * wy;
      const double w10 = (1.0 - wx) * wy;
      const double w01 = wx * (1.0 - wy);
      const double w11 = (1.0 - wx) * (1.0 - wy);

      g4_cuda_atomic_add_double(&target_re[base], cpart_re * w00);
      g4_cuda_atomic_add_double(&target_im[base], cpart_im * w00);

      g4_cuda_atomic_add_double(&target_re[base + 1], cpart_re * w10);
      g4_cuda_atomic_add_double(&target_im[base + 1], cpart_im * w10);

      g4_cuda_atomic_add_double(&target_re[base + params.ngrid], cpart_re * w01);
      g4_cuda_atomic_add_double(&target_im[base + params.ngrid], cpart_im * w01);

      g4_cuda_atomic_add_double(&target_re[base + params.ngrid + 1], cpart_re * w11);
      g4_cuda_atomic_add_double(&target_im[base + params.ngrid + 1], cpart_im * w11);
    });
}

void adi_build_rhs_y_laplacian(int total_cells,
                               int ngrid,
                               double cstep_im,
                               const double* source_re,
                               const double* source_im,
                               AdiFieldView field)
{
  const int plane = ngrid * ngrid;
  const double* field_re = field.field_re;
  const double* field_im = field.field_im;
  double* r_re = field.r_re;
  double* r_im = field.r_im;

  g4_parallel_for(total_cells,
    [=] GENESIS4_CUDA_DEVICE (int g) noexcept {
      const int local = g % plane;
      const int iy = local / ngrid;

      const CudaComplex center(field_re[g], field_im[g]);
      CudaComplex lap(0.0, 0.0);

      if (iy == 0) {
        lap = CudaComplex(field_re[g + ngrid], field_im[g + ngrid])
            - center * 2.0;
      } else if (iy == ngrid - 1) {
        lap = CudaComplex(field_re[g - ngrid], field_im[g - ngrid])
            - center * 2.0;
      } else {
        lap = CudaComplex(field_re[g + ngrid], field_im[g + ngrid])
            + CudaComplex(field_re[g - ngrid], field_im[g - ngrid])
            - center * 2.0;
      }

      // rhs = center + (i * cstep_im) * lap
      const CudaComplex i_cstep_lap(-cstep_im * lap.imag(),
                                   cstep_im * lap.real());

      CudaComplex rhs = center + i_cstep_lap;
      if (source_re != nullptr && source_im != nullptr) {
        rhs += CudaComplex(source_re[g], source_im[g]);
      }
      r_re[g] = rhs.real();
      r_im[g] = rhs.imag();
    });
}

void adi_solve_x_thomas(int nslice,
                        int ngrid,
                        AdiFieldView field,
                        AdiCoefficientsView coeff)
{
  const int nlines = nslice * ngrid;
  const int plane = ngrid * ngrid;
  const double* r_re = field.r_re;
  const double* r_im = field.r_im;
  const double* c_re = coeff.c_re;
  const double* c_im = coeff.c_im;
  const double* cbet_re = coeff.cbet_re;
  const double* cbet_im = coeff.cbet_im;
  const double* cwet_re = coeff.cwet_re;
  const double* cwet_im = coeff.cwet_im;
  double* field_re = field.field_re;
  double* field_im = field.field_im;

  g4_parallel_for(nlines,
    [=] GENESIS4_CUDA_DEVICE (int line) noexcept {
      const int s = line / ngrid;
      const int row = line - s * ngrid;
      const int base = s * plane + row * ngrid;

      CudaComplex u = CudaComplex(r_re[base], r_im[base])
                  * CudaComplex(cbet_re[0], cbet_im[0]);
      field_re[base] = u.real();
      field_im[base] = u.imag();

      for (int k = 1; k < ngrid; ++k) {
        const int idx = base + k;

        const CudaComplex ck(c_re[k], c_im[k]);
        const CudaComplex prev(field_re[idx - 1], field_im[idx - 1]);
        const CudaComplex rhs(r_re[idx], r_im[idx]);

        u = (rhs - ck * prev) * CudaComplex(cbet_re[k], cbet_im[k]);

        field_re[idx] = u.real();
        field_im[idx] = u.imag();
      }

      for (int k = ngrid - 2; k >= 0; --k) {
        const int idx = base + k;

        const CudaComplex corr =
            CudaComplex(cwet_re[k + 1], cwet_im[k + 1])
          * CudaComplex(field_re[idx + 1], field_im[idx + 1]);

        u = CudaComplex(field_re[idx], field_im[idx]) - corr;

        field_re[idx] = u.real();
        field_im[idx] = u.imag();
      }
    });
}

void adi_build_rhs_x_laplacian(int total_cells,
                               int ngrid,
                               double cstep_im,
                               const double* source_re,
                               const double* source_im,
                               AdiFieldView field)
{
  const int plane = ngrid * ngrid;
  const double* field_re = field.field_re;
  const double* field_im = field.field_im;
  double* r_re = field.r_re;
  double* r_im = field.r_im;

  g4_parallel_for(total_cells,
    [=] GENESIS4_CUDA_DEVICE (int g) noexcept {
      const int local = g % plane;
      const int ix = local % ngrid;

      const CudaComplex center(field_re[g], field_im[g]);
      CudaComplex lap(0.0, 0.0);

      if (ix == 0) {
        lap = CudaComplex(field_re[g + 1], field_im[g + 1])
            - center * 2.0;
      } else if (ix == ngrid - 1) {
        lap = CudaComplex(field_re[g - 1], field_im[g - 1])
            - center * 2.0;
      } else {
        lap = CudaComplex(field_re[g + 1], field_im[g + 1])
            + CudaComplex(field_re[g - 1], field_im[g - 1])
            - center * 2.0;
      }

      // rhs = center + (i * cstep_im) * lap
      const CudaComplex i_cstep_lap(-cstep_im * lap.imag(),
                                   cstep_im * lap.real());

      CudaComplex rhs = center + i_cstep_lap;
      if (source_re != nullptr && source_im != nullptr) {
        rhs += CudaComplex(source_re[g], source_im[g]);
      }
      r_re[g] = rhs.real();
      r_im[g] = rhs.imag();
    });
}

void adi_solve_y_thomas(int nslice,
                        int ngrid,
                        AdiFieldView field,
                        AdiCoefficientsView coeff)
{
  const int nlines = nslice * ngrid;
  const int plane = ngrid * ngrid;
  const double* r_re = field.r_re;
  const double* r_im = field.r_im;
  const double* c_re = coeff.c_re;
  const double* c_im = coeff.c_im;
  const double* cbet_re = coeff.cbet_re;
  const double* cbet_im = coeff.cbet_im;
  const double* cwet_re = coeff.cwet_re;
  const double* cwet_im = coeff.cwet_im;
  double* field_re = field.field_re;
  double* field_im = field.field_im;

  g4_parallel_for(nlines,
    [=] GENESIS4_CUDA_DEVICE (int line) noexcept {
      const int s = line / ngrid;
      const int col = line - s * ngrid;
      const int base = s * plane + col;

      CudaComplex u = CudaComplex(r_re[base], r_im[base])
                  * CudaComplex(cbet_re[0], cbet_im[0]);
      field_re[base] = u.real();
      field_im[base] = u.imag();

      for (int k = 1; k < ngrid; ++k) {
        const int idx = base + k * ngrid;

        const CudaComplex ck(c_re[k], c_im[k]);
        const CudaComplex prev(field_re[idx - ngrid], field_im[idx - ngrid]);
        const CudaComplex rhs(r_re[idx], r_im[idx]);

        u = (rhs - ck * prev) * CudaComplex(cbet_re[k], cbet_im[k]);

        field_re[idx] = u.real();
        field_im[idx] = u.imag();
      }

      for (int k = ngrid - 2; k >= 0; --k) {
        const int idx = base + k * ngrid;

        const CudaComplex corr =
            CudaComplex(cwet_re[k + 1], cwet_im[k + 1])
          * CudaComplex(field_re[idx + ngrid], field_im[idx + ngrid]);

        u = CudaComplex(field_re[idx], field_im[idx]) - corr;

        field_re[idx] = u.real();
        field_im[idx] = u.imag();
      }
    });
}

void adi_prepare_pcr_factors(int ngrid,
                             double rtmp,
                             CudaDeviceBuffer<double>& alpha_re,
                             CudaDeviceBuffer<double>& alpha_im,
                             CudaDeviceBuffer<double>& beta_re,
                             CudaDeviceBuffer<double>& beta_im,
                             CudaDeviceBuffer<double>& b_re,
                             CudaDeviceBuffer<double>& b_im,
                             int& factor_ngrid,
                             int& num_stages)
{
  num_stages = 0;
  for (int stride = 1; stride < ngrid; stride <<= 1) {
    ++num_stages;
  }

  std::vector<double> h_alpha_re(num_stages * ngrid);
  std::vector<double> h_alpha_im(num_stages * ngrid);
  std::vector<double> h_beta_re(num_stages * ngrid);
  std::vector<double> h_beta_im(num_stages * ngrid);
  std::vector<double> h_b_re(ngrid);
  std::vector<double> h_b_im(ngrid);

  std::vector<CudaComplex> a(ngrid);
  std::vector<CudaComplex> b(ngrid);
  std::vector<CudaComplex> c(ngrid);
  std::vector<CudaComplex> na(ngrid);
  std::vector<CudaComplex> nb(ngrid);
  std::vector<CudaComplex> nc(ngrid);

  for (int i = 0; i < ngrid; ++i) {
    a[i] = (i == 0) ? CudaComplex(0.0, 0.0) : CudaComplex(0.0, -rtmp);
    b[i] = CudaComplex(1.0, 2.0 * rtmp);
    c[i] = (i == ngrid - 1) ? CudaComplex(0.0, 0.0) : CudaComplex(0.0, -rtmp);
  }

  int stage = 0;
  for (int stride = 1; stride < ngrid; stride <<= 1, ++stage) {
    for (int tid = 0; tid < ngrid; ++tid) {
      CudaComplex alpha(0.0, 0.0);
      CudaComplex beta(0.0, 0.0);
      CudaComplex a_new(0.0, 0.0);
      CudaComplex b_new = b[tid];
      CudaComplex c_new(0.0, 0.0);

      if (tid >= stride) {
        alpha = -a[tid] / b[tid - stride];
        a_new = alpha * a[tid - stride];
        b_new += alpha * c[tid - stride];
      }
      if (tid + stride < ngrid) {
        beta = -c[tid] / b[tid + stride];
        c_new = beta * c[tid + stride];
        b_new += beta * a[tid + stride];
      }

      const int off = stage * ngrid + tid;
      h_alpha_re[off] = alpha.real();
      h_alpha_im[off] = alpha.imag();
      h_beta_re[off] = beta.real();
      h_beta_im[off] = beta.imag();
      na[tid] = a_new;
      nb[tid] = b_new;
      nc[tid] = c_new;
    }

    a.swap(na);
    b.swap(nb);
    c.swap(nc);
  }

  for (int i = 0; i < ngrid; ++i) {
    h_b_re[i] = b[i].real();
    h_b_im[i] = b[i].imag();
  }

  alpha_re.copy_from_host(h_alpha_re.data(), h_alpha_re.size());
  alpha_im.copy_from_host(h_alpha_im.data(), h_alpha_im.size());
  beta_re.copy_from_host(h_beta_re.data(), h_beta_re.size());
  beta_im.copy_from_host(h_beta_im.data(), h_beta_im.size());
  b_re.copy_from_host(h_b_re.data(), h_b_re.size());
  b_im.copy_from_host(h_b_im.data(), h_b_im.size());
  g4_cuda_synchronize();

  factor_ngrid = ngrid;
}

__global__ void adi_solve_x_pcr_kernel(int ngrid,
                                       PcrFactorView pcr,
                                       AdiFieldView field)
{
  extern __shared__ double shared[];

  double* cur_re = shared;
  double* cur_im = cur_re + blockDim.x;
  double* nxt_re = cur_im + blockDim.x;
  double* nxt_im = nxt_re + blockDim.x;

  const int tid = threadIdx.x;
  const int line = blockIdx.x;
  const int plane = ngrid * ngrid;
  const int s = line / ngrid;
  const int row = line - s * ngrid;
  const int base = s * plane + row * ngrid;

  if (tid < ngrid) {
    const int idx = base + tid;
    cur_re[tid] = field.r_re[idx];
    cur_im[tid] = field.r_im[idx];
  }
  __syncthreads();

  int stride = 1;
  for (int stage = 0; stage < pcr.num_stages; ++stage, stride <<= 1) {
    if (tid < ngrid) {
      const int off = stage * ngrid + tid;
      CudaComplex d(cur_re[tid], cur_im[tid]);
      if (tid >= stride) {
        d += CudaComplex(pcr.alpha_re[off], pcr.alpha_im[off])
           * CudaComplex(cur_re[tid - stride], cur_im[tid - stride]);
      }
      if (tid + stride < ngrid) {
        d += CudaComplex(pcr.beta_re[off], pcr.beta_im[off])
           * CudaComplex(cur_re[tid + stride], cur_im[tid + stride]);
      }
      nxt_re[tid] = d.real();
      nxt_im[tid] = d.imag();
    }
    __syncthreads();
    double* tmp_re = cur_re;
    double* tmp_im = cur_im;
    cur_re = nxt_re;
    cur_im = nxt_im;
    nxt_re = tmp_re;
    nxt_im = tmp_im;
  }

  if (tid < ngrid) {
    const int idx = base + tid;
    const CudaComplex u =
        CudaComplex(cur_re[tid], cur_im[tid])
      / CudaComplex(pcr.b_re[tid], pcr.b_im[tid]);
    field.field_re[idx] = u.real();
    field.field_im[idx] = u.imag();
  }
}

__global__ void adi_solve_y_pcr_kernel(int ngrid,
                                       PcrFactorView pcr,
                                       AdiFieldView field)
{
  extern __shared__ double shared[];

  double* cur_re = shared;
  double* cur_im = cur_re + blockDim.x;
  double* nxt_re = cur_im + blockDim.x;
  double* nxt_im = nxt_re + blockDim.x;

  const int tid = threadIdx.x;
  const int line = blockIdx.x;
  const int plane = ngrid * ngrid;
  const int s = line / ngrid;
  const int col = line - s * ngrid;
  const int base = s * plane + col;

  if (tid < ngrid) {
    const int idx = base + tid * ngrid;
    cur_re[tid] = field.r_re[idx];
    cur_im[tid] = field.r_im[idx];
  }
  __syncthreads();

  int stride = 1;
  for (int stage = 0; stage < pcr.num_stages; ++stage, stride <<= 1) {
    if (tid < ngrid) {
      const int off = stage * ngrid + tid;
      CudaComplex d(cur_re[tid], cur_im[tid]);
      if (tid >= stride) {
        d += CudaComplex(pcr.alpha_re[off], pcr.alpha_im[off])
           * CudaComplex(cur_re[tid - stride], cur_im[tid - stride]);
      }
      if (tid + stride < ngrid) {
        d += CudaComplex(pcr.beta_re[off], pcr.beta_im[off])
           * CudaComplex(cur_re[tid + stride], cur_im[tid + stride]);
      }
      nxt_re[tid] = d.real();
      nxt_im[tid] = d.imag();
    }
    __syncthreads();
    double* tmp_re = cur_re;
    double* tmp_im = cur_im;
    cur_re = nxt_re;
    cur_im = nxt_im;
    nxt_re = tmp_re;
    nxt_im = tmp_im;
  }

  if (tid < ngrid) {
    const int idx = base + tid * ngrid;
    const CudaComplex u =
        CudaComplex(cur_re[tid], cur_im[tid])
      / CudaComplex(pcr.b_re[tid], pcr.b_im[tid]);
    field.field_re[idx] = u.real();
    field.field_im[idx] = u.imag();
  }
}

int g4_next_power_of_two(int n) noexcept
{
  int p = 1;
  while (p < n) { p <<= 1; }
  return p;
}

bool adi_try_solve_x_pcr(int nlines,
                         int ngrid,
                         PcrFactorView pcr,
                         AdiFieldView field)
{
  if (ngrid <= 1 || ngrid > GENESIS_FIELD_ADI_PCR_MAX_NGRID || pcr.num_stages <= 0) {
    return false;
  }
  const int block = g4_next_power_of_two(ngrid);
  const std::size_t shmem = 4 * static_cast<std::size_t>(block) * sizeof(double);
  adi_solve_x_pcr_kernel<<<nlines, block, shmem, g4_cuda_stream()>>>(ngrid,
                                                                            pcr,
                                                                            field);
  return cudaGetLastError() == cudaSuccess;
}

bool adi_try_solve_y_pcr(int nlines,
                         int ngrid,
                         PcrFactorView pcr,
                         AdiFieldView field)
{
  if (ngrid <= 1 || ngrid > GENESIS_FIELD_ADI_PCR_MAX_NGRID || pcr.num_stages <= 0) {
    return false;
  }
  const int block = g4_next_power_of_two(ngrid);
  const std::size_t shmem = 4 * static_cast<std::size_t>(block) * sizeof(double);
  adi_solve_y_pcr_kernel<<<nlines, block, shmem, g4_cuda_stream()>>>(ngrid,
                                                                            pcr,
                                                                            field);
  return cudaGetLastError() == cudaSuccess;
}

} // namespace


FieldSolverADICUDA::FieldSolverADICUDA(bool atomic_source)
    : atomic_source_(atomic_source)
{}

FieldSolverADICUDA::~FieldSolverADICUDA() = default;


void FieldSolverADICUDA::advance(double delz, Field *field, Beam *beam, Undulator *und) {

  if (field == nullptr || beam == nullptr || und == nullptr) {
    g4_cuda_abort("FieldSolverADICUDA::advance received a null field/beam/undulator pointer");
  }

  Genesis4BeamSoA* bsoa = beam->beamSoA;
  Genesis4FieldSoA* fsoa = field->fieldSoA;

  if (bsoa == nullptr || !bsoa->initialized ||
      fsoa == nullptr || !fsoa->initialized ||
      ngrid == 0 || field->ngrid != static_cast<int>(ngrid)) {
    g4_cuda_abort("FieldSolverADICUDA::advance requires initialized BeamSoA/FieldSoA and matching ngrid");
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
      static_cast<int>(beam->current.size()) < nslice) {
    g4_cuda_abort("FieldSolverADICUDA::advance found inconsistent SoA dimensions");
  }

  if (d_alloc_nslice != nslice || d_alloc_cells_per_slice != plane) {
    d_r_re.resize_discard(total_cells);
    d_r_im.resize_discard(total_cells);
    d_current.resize_discard(nslice);
    d_slice_scl.resize_discard(nslice);
    d_alloc_nslice = nslice;
    d_alloc_cells_per_slice = plane;
  }

  const bool pcr_factor_required = (ngrid_i > 1 && ngrid_i <= GENESIS_FIELD_ADI_PCR_MAX_NGRID);
  if (static_cast<int>(d_c_re.size()) < ngrid_i ||
      static_cast<int>(d_cbet_re.size()) < ngrid_i ||
      static_cast<int>(d_cwet_re.size()) < ngrid_i ||
      d_pcr_factor_ngrid != ngrid_i ||
      static_cast<int>(d_pcr_b_re.size()) < ngrid_i ||
      (pcr_factor_required &&
       (d_pcr_num_stages <= 0 ||
        static_cast<int>(d_pcr_alpha_re.size()) < d_pcr_num_stages * ngrid_i ||
        static_cast<int>(d_pcr_beta_re.size()) < d_pcr_num_stages * ngrid_i))) {
    g4_cuda_abort("FieldSolverADICUDA::advance was called before GPU ADI coefficients/PCR factors were initialized");
  }

  const int harm = field->getHarm();
  const bool do_source = und->inUndulator() && field->isEnabled() && ((harm % 2) == 1);

  if (do_source) {
    genesis4_cuda::copy_host_to_device(beam->current.begin(),
                                       beam->current.begin() + nslice,
                                       d_current.begin());

    adi_compute_slice_scl(nslice,
                          bsoa->slice_offsets.data(),
                          d_current.data(),
                          und->fc(harm),
                          field->xks,
                          field->dgrid,
                          delz,
                          vacimp,
                          eev,
                          d_slice_scl.data());
  }

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

  double* r_re = d_r_re.data();
  double* r_im = d_r_im.data();
  double* fld_re = fsoa->field_re.data();
  double* fld_im = fsoa->field_im.data();

  AdiFieldView adi_field {r_re,
                          r_im,
                          fld_re,
                          fld_im};
  AdiCoefficientsView adi_coeff {d_c_re.data(),
                                  d_c_im.data(),
                                  d_cbet_re.data(),
                                  d_cbet_im.data(),
                                  d_cwet_re.data(),
                                  d_cwet_im.data()};
  PcrFactorView pcr_factors {d_pcr_num_stages,
                             d_pcr_alpha_re.data(),
                             d_pcr_alpha_im.data(),
                             d_pcr_beta_re.data(),
                             d_pcr_beta_im.data(),
                             d_pcr_b_re.data(),
                             d_pcr_b_im.data()};
  SourceParams source_params {nslice,
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
                              grady};
  SourceParticleView source_particles {bsoa->x.data(),
                                       bsoa->y.data(),
                                       bsoa->gamma.data(),
                                       bsoa->theta.data(),
                                       bsoa->slice_id.data()};
  const bool use_source = do_source && total_particles > 0;
  const double* source_re = nullptr;
  const double* source_im = nullptr;

  if (use_source && !atomic_source_) {
    d_source_re.resize_discard(total_cells);
    d_source_im.resize_discard(total_cells);
    adi_zero_source_grid(total_cells, d_source_re.data(), d_source_im.data());
    adi_deposit_source_atomic(total_particles,
                              source_params,
                              source_particles,
                              d_slice_scl.data(),
                              d_source_re.data(),
                              d_source_im.data());
    source_re = d_source_re.data();
    source_im = d_source_im.data();
  }

  const int nlines = nslice * ngrid_i;
  const double cstep_im = cstep.imag();

  adi_build_rhs_y_laplacian(total_cells, ngrid_i, cstep_im, source_re, source_im, adi_field);
  if (use_source && atomic_source_) {
    adi_deposit_source_atomic(total_particles,
                              source_params,
                              source_particles,
                              d_slice_scl.data(),
                              r_re,
                              r_im);
  }
  if (!adi_try_solve_x_pcr(nlines, ngrid_i, pcr_factors, adi_field)) {
    adi_solve_x_thomas(nslice, ngrid_i, adi_field, adi_coeff);
  }

  adi_build_rhs_x_laplacian(total_cells, ngrid_i, cstep_im, source_re, source_im, adi_field);
  if (use_source && atomic_source_) {
    adi_deposit_source_atomic(total_particles,
                              source_params,
                              source_particles,
                              d_slice_scl.data(),
                              r_re,
                              r_im);
  }
  if (!adi_try_solve_y_pcr(nlines, ngrid_i, pcr_factors, adi_field)) {
    adi_solve_y_thomas(nslice, ngrid_i, adi_field, adi_coeff);
  }

}


void FieldSolverADICUDA::init(double delz, double dgrid, double xks, unsigned int ngrid_in) {

  if (delz == delz_save && ngrid == ngrid_in) {
    return;
  }

  delz_save = delz;
  ngrid = ngrid_in;

  double rtmp = 0.25 * delz / (xks * dgrid * dgrid); //factor dz/(4 ks dx^2)
  cstep = complex<double>(0, rtmp);

  vector<double> mupp(ngrid);
  vector<double> mmid(ngrid);
  vector<double> mlow(ngrid);
  vector<CudaComplex> c_host(ngrid);
  vector<CudaComplex> cbet_host(ngrid);
  vector<CudaComplex> cwet_host(ngrid);
  vector<CudaComplex> cwrk1(ngrid);
  vector<CudaComplex> cwrk2(ngrid);

  mupp[0] = rtmp;
  mmid[0] = -2 * rtmp;
  mlow[0] = 0;
  for (unsigned int i = 1; i < ngrid - 1; i++) {
    mupp[i] = rtmp;
    mmid[i] = -2 * rtmp;
    mlow[i] = rtmp;
  }
  mupp[ngrid - 1] = 0;
  mmid[ngrid - 1] = -2 * rtmp;
  mlow[ngrid - 1] = rtmp;

  for (unsigned int i = 0; i < ngrid; i++) {
    cwrk1[i] = CudaComplex(0.0, -mupp[i]);
    cwrk2[i] = CudaComplex(1.0, -mmid[i]);
    c_host[i] = CudaComplex(0.0, -mlow[i]);
  }

  cbet_host[0] = CudaComplex(1.0, 0.0) / cwrk2[0];
  cwet_host[0] = CudaComplex(0.0, 0.0);
  for (unsigned int i = 1; i < ngrid; i++) {
    cwet_host[i] = cwrk1[i - 1] * cbet_host[i - 1];
    cbet_host[i] = CudaComplex(1.0, 0.0) /
                   (cwrk2[i] - c_host[i] * cwet_host[i]);
  }

  std::vector<double> h_c_re(ngrid);
  std::vector<double> h_c_im(ngrid);
  std::vector<double> h_cbet_re(ngrid);
  std::vector<double> h_cbet_im(ngrid);
  std::vector<double> h_cwet_re(ngrid);
  std::vector<double> h_cwet_im(ngrid);

  for (int i = 0; i < static_cast<int>(ngrid); ++i) {
    h_c_re[i] = c_host[i].real();
    h_c_im[i] = c_host[i].imag();
    h_cbet_re[i] = cbet_host[i].real();
    h_cbet_im[i] = cbet_host[i].imag();
    h_cwet_re[i] = cwet_host[i].real();
    h_cwet_im[i] = cwet_host[i].imag();
  }

  d_c_re.copy_from_host(h_c_re.data(), h_c_re.size());
  d_c_im.copy_from_host(h_c_im.data(), h_c_im.size());
  d_cbet_re.copy_from_host(h_cbet_re.data(), h_cbet_re.size());
  d_cbet_im.copy_from_host(h_cbet_im.data(), h_cbet_im.size());
  d_cwet_re.copy_from_host(h_cwet_re.data(), h_cwet_re.size());
  d_cwet_im.copy_from_host(h_cwet_im.data(), h_cwet_im.size());
  g4_cuda_synchronize();

  adi_prepare_pcr_factors(static_cast<int>(ngrid),
                          rtmp,
                          d_pcr_alpha_re,
                          d_pcr_alpha_im,
                          d_pcr_beta_re,
                          d_pcr_beta_im,
                          d_pcr_b_re,
                          d_pcr_b_im,
                          d_pcr_factor_ngrid,
                          d_pcr_num_stages);
}
