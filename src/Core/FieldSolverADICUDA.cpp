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


#ifndef GENESIS_FIELD_ADI_PCR_MAX_NGRID
#define GENESIS_FIELD_ADI_PCR_MAX_NGRID 512
#endif


namespace {

struct G4ComplexPair {
  double re;
  double im;
};

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

GENESIS4_CUDA_HOST_DEVICE GENESIS4_CUDA_FORCE_INLINE
G4ComplexPair g4_make_complex(double re, double im) noexcept
{
  return {re, im};
}

GENESIS4_CUDA_HOST_DEVICE GENESIS4_CUDA_FORCE_INLINE
G4ComplexPair g4_add(G4ComplexPair a, G4ComplexPair b) noexcept
{
  return {a.re + b.re, a.im + b.im};
}

GENESIS4_CUDA_HOST_DEVICE GENESIS4_CUDA_FORCE_INLINE
G4ComplexPair g4_sub(G4ComplexPair a, G4ComplexPair b) noexcept
{
  return {a.re - b.re, a.im - b.im};
}

GENESIS4_CUDA_HOST_DEVICE GENESIS4_CUDA_FORCE_INLINE
G4ComplexPair g4_mul(G4ComplexPair a, G4ComplexPair b) noexcept
{
  return {a.re * b.re - a.im * b.im,
          a.re * b.im + a.im * b.re};
}

GENESIS4_CUDA_HOST_DEVICE GENESIS4_CUDA_FORCE_INLINE
G4ComplexPair g4_div(G4ComplexPair a, G4ComplexPair b) noexcept
{
  const double den = b.re * b.re + b.im * b.im;
  return {(a.re * b.re + a.im * b.im) / den,
          (a.im * b.re - a.re * b.im) / den};
}

GENESIS4_CUDA_HOST_DEVICE GENESIS4_CUDA_FORCE_INLINE
G4ComplexPair g4_neg(G4ComplexPair a) noexcept
{
  return {-a.re, -a.im};
}

GENESIS4_CUDA_HOST_DEVICE GENESIS4_CUDA_FORCE_INLINE
G4ComplexPair g4_scale(G4ComplexPair a, double s) noexcept
{
  return {a.re * s, a.im * s};
}

GENESIS4_CUDA_HOST_DEVICE GENESIS4_CUDA_FORCE_INLINE
G4ComplexPair g4_mul_i_scale(G4ComplexPair a, double s) noexcept
{
  // (i*s) * (a.re + i*a.im) = -s*a.im + i*s*a.re
  return {-s * a.im, s * a.re};
}

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

void adi_deposit_source_grid(int total_particles,
                             SourceParams params,
                             SourceParticleView particles,
                             const double* slice_scl,
                             double* source_re,
                             double* source_im)
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

      g4_cuda_atomic_add_double(&source_re[base], cpart_re * w00);
      g4_cuda_atomic_add_double(&source_im[base], cpart_im * w00);

      g4_cuda_atomic_add_double(&source_re[base + 1], cpart_re * w10);
      g4_cuda_atomic_add_double(&source_im[base + 1], cpart_im * w10);

      g4_cuda_atomic_add_double(&source_re[base + params.ngrid], cpart_re * w01);
      g4_cuda_atomic_add_double(&source_im[base + params.ngrid], cpart_im * w01);

      g4_cuda_atomic_add_double(&source_re[base + params.ngrid + 1], cpart_re * w11);
      g4_cuda_atomic_add_double(&source_im[base + params.ngrid + 1], cpart_im * w11);
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

      G4ComplexPair lap = g4_make_complex(0.0, 0.0);
      const G4ComplexPair center = g4_make_complex(field_re[g], field_im[g]);

      if (iy == 0) {
        lap = g4_sub(g4_make_complex(field_re[g + ngrid], field_im[g + ngrid]),
                     g4_scale(center, 2.0));
      } else if (iy == ngrid - 1) {
        lap = g4_sub(g4_make_complex(field_re[g - ngrid], field_im[g - ngrid]),
                     g4_scale(center, 2.0));
      } else {
        lap = g4_add(g4_make_complex(field_re[g + ngrid], field_im[g + ngrid]),
                     g4_make_complex(field_re[g - ngrid], field_im[g - ngrid]));
        lap = g4_sub(lap, g4_scale(center, 2.0));
      }

      G4ComplexPair rhs = g4_add(center, g4_mul_i_scale(lap, cstep_im));
      if (source_re != nullptr && source_im != nullptr) {
        rhs.re += source_re[g];
        rhs.im += source_im[g];
      }
      r_re[g] = rhs.re;
      r_im[g] = rhs.im;
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

      G4ComplexPair u = g4_mul(g4_make_complex(r_re[base], r_im[base]),
                               g4_make_complex(cbet_re[0], cbet_im[0]));
      field_re[base] = u.re;
      field_im[base] = u.im;

      for (int k = 1; k < ngrid; ++k) {
        const int idx = base + k;
        const G4ComplexPair ck = g4_make_complex(c_re[k], c_im[k]);
        const G4ComplexPair prev = g4_make_complex(field_re[idx - 1], field_im[idx - 1]);
        const G4ComplexPair tmp = g4_sub(g4_make_complex(r_re[idx], r_im[idx]),
                                         g4_mul(ck, prev));
        u = g4_mul(tmp, g4_make_complex(cbet_re[k], cbet_im[k]));
        field_re[idx] = u.re;
        field_im[idx] = u.im;
      }

      for (int k = ngrid - 2; k >= 0; --k) {
        const int idx = base + k;
        const G4ComplexPair corr = g4_mul(g4_make_complex(cwet_re[k + 1], cwet_im[k + 1]),
                                          g4_make_complex(field_re[idx + 1], field_im[idx + 1]));
        u = g4_sub(g4_make_complex(field_re[idx], field_im[idx]), corr);
        field_re[idx] = u.re;
        field_im[idx] = u.im;
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

      G4ComplexPair lap = g4_make_complex(0.0, 0.0);
      const G4ComplexPair center = g4_make_complex(field_re[g], field_im[g]);

      if (ix == 0) {
        lap = g4_sub(g4_make_complex(field_re[g + 1], field_im[g + 1]),
                     g4_scale(center, 2.0));
      } else if (ix == ngrid - 1) {
        lap = g4_sub(g4_make_complex(field_re[g - 1], field_im[g - 1]),
                     g4_scale(center, 2.0));
      } else {
        lap = g4_add(g4_make_complex(field_re[g + 1], field_im[g + 1]),
                     g4_make_complex(field_re[g - 1], field_im[g - 1]));
        lap = g4_sub(lap, g4_scale(center, 2.0));
      }

      G4ComplexPair rhs = g4_add(center, g4_mul_i_scale(lap, cstep_im));
      if (source_re != nullptr && source_im != nullptr) {
        rhs.re += source_re[g];
        rhs.im += source_im[g];
      }
      r_re[g] = rhs.re;
      r_im[g] = rhs.im;
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

      G4ComplexPair u = g4_mul(g4_make_complex(r_re[base], r_im[base]),
                               g4_make_complex(cbet_re[0], cbet_im[0]));
      field_re[base] = u.re;
      field_im[base] = u.im;

      for (int k = 1; k < ngrid; ++k) {
        const int idx = base + k * ngrid;
        const G4ComplexPair ck = g4_make_complex(c_re[k], c_im[k]);
        const G4ComplexPair prev = g4_make_complex(field_re[idx - ngrid], field_im[idx - ngrid]);
        const G4ComplexPair tmp = g4_sub(g4_make_complex(r_re[idx], r_im[idx]),
                                         g4_mul(ck, prev));
        u = g4_mul(tmp, g4_make_complex(cbet_re[k], cbet_im[k]));
        field_re[idx] = u.re;
        field_im[idx] = u.im;
      }

      for (int k = ngrid - 2; k >= 0; --k) {
        const int idx = base + k * ngrid;
        const G4ComplexPair corr = g4_mul(g4_make_complex(cwet_re[k + 1], cwet_im[k + 1]),
                                          g4_make_complex(field_re[idx + ngrid], field_im[idx + ngrid]));
        u = g4_sub(g4_make_complex(field_re[idx], field_im[idx]), corr);
        field_re[idx] = u.re;
        field_im[idx] = u.im;
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

  std::vector<G4ComplexPair> a(ngrid);
  std::vector<G4ComplexPair> b(ngrid);
  std::vector<G4ComplexPair> c(ngrid);
  std::vector<G4ComplexPair> na(ngrid);
  std::vector<G4ComplexPair> nb(ngrid);
  std::vector<G4ComplexPair> nc(ngrid);

  for (int i = 0; i < ngrid; ++i) {
    a[i] = (i == 0) ? g4_make_complex(0.0, 0.0) : g4_make_complex(0.0, -rtmp);
    b[i] = g4_make_complex(1.0, 2.0 * rtmp);
    c[i] = (i == ngrid - 1) ? g4_make_complex(0.0, 0.0) : g4_make_complex(0.0, -rtmp);
  }

  int stage = 0;
  for (int stride = 1; stride < ngrid; stride <<= 1, ++stage) {
    for (int tid = 0; tid < ngrid; ++tid) {
      G4ComplexPair alpha = g4_make_complex(0.0, 0.0);
      G4ComplexPair beta = g4_make_complex(0.0, 0.0);
      G4ComplexPair a_new = g4_make_complex(0.0, 0.0);
      G4ComplexPair b_new = b[tid];
      G4ComplexPair c_new = g4_make_complex(0.0, 0.0);

      if (tid >= stride) {
        alpha = g4_neg(g4_div(a[tid], b[tid - stride]));
        a_new = g4_mul(alpha, a[tid - stride]);
        b_new = g4_add(b_new, g4_mul(alpha, c[tid - stride]));
      }
      if (tid + stride < ngrid) {
        beta = g4_neg(g4_div(c[tid], b[tid + stride]));
        c_new = g4_mul(beta, c[tid + stride]);
        b_new = g4_add(b_new, g4_mul(beta, a[tid + stride]));
      }

      const int off = stage * ngrid + tid;
      h_alpha_re[off] = alpha.re;
      h_alpha_im[off] = alpha.im;
      h_beta_re[off] = beta.re;
      h_beta_im[off] = beta.im;
      na[tid] = a_new;
      nb[tid] = b_new;
      nc[tid] = c_new;
    }

    a.swap(na);
    b.swap(nb);
    c.swap(nc);
  }

  for (int i = 0; i < ngrid; ++i) {
    h_b_re[i] = b[i].re;
    h_b_im[i] = b[i].im;
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
  extern __shared__ G4ComplexPair shared[];
  G4ComplexPair* cur = shared;
  G4ComplexPair* nxt = cur + blockDim.x;

  const int tid = threadIdx.x;
  const int line = blockIdx.x;
  const int plane = ngrid * ngrid;
  const int s = line / ngrid;
  const int row = line - s * ngrid;
  const int base = s * plane + row * ngrid;

  if (tid < ngrid) {
    const int idx = base + tid;
    cur[tid] = g4_make_complex(field.r_re[idx], field.r_im[idx]);
  }
  __syncthreads();

  int stride = 1;
  for (int stage = 0; stage < pcr.num_stages; ++stage, stride <<= 1) {
    if (tid < ngrid) {
      const int off = stage * ngrid + tid;
      G4ComplexPair d = cur[tid];
      if (tid >= stride) {
        d = g4_add(d, g4_mul(g4_make_complex(pcr.alpha_re[off], pcr.alpha_im[off]),
                             cur[tid - stride]));
      }
      if (tid + stride < ngrid) {
        d = g4_add(d, g4_mul(g4_make_complex(pcr.beta_re[off], pcr.beta_im[off]),
                             cur[tid + stride]));
      }
      nxt[tid] = d;
    }
    __syncthreads();
    G4ComplexPair* tmp = cur;
    cur = nxt;
    nxt = tmp;
  }

  if (tid < ngrid) {
    const int idx = base + tid;
    const G4ComplexPair u = g4_div(cur[tid], g4_make_complex(pcr.b_re[tid], pcr.b_im[tid]));
    field.field_re[idx] = u.re;
    field.field_im[idx] = u.im;
  }
}

__global__ void adi_solve_y_pcr_kernel(int ngrid,
                                       PcrFactorView pcr,
                                       AdiFieldView field)
{
  extern __shared__ G4ComplexPair shared[];
  G4ComplexPair* cur = shared;
  G4ComplexPair* nxt = cur + blockDim.x;

  const int tid = threadIdx.x;
  const int line = blockIdx.x;
  const int plane = ngrid * ngrid;
  const int s = line / ngrid;
  const int col = line - s * ngrid;
  const int base = s * plane + col;

  if (tid < ngrid) {
    const int idx = base + tid * ngrid;
    cur[tid] = g4_make_complex(field.r_re[idx], field.r_im[idx]);
  }
  __syncthreads();

  int stride = 1;
  for (int stage = 0; stage < pcr.num_stages; ++stage, stride <<= 1) {
    if (tid < ngrid) {
      const int off = stage * ngrid + tid;
      G4ComplexPair d = cur[tid];
      if (tid >= stride) {
        d = g4_add(d, g4_mul(g4_make_complex(pcr.alpha_re[off], pcr.alpha_im[off]),
                             cur[tid - stride]));
      }
      if (tid + stride < ngrid) {
        d = g4_add(d, g4_mul(g4_make_complex(pcr.beta_re[off], pcr.beta_im[off]),
                             cur[tid + stride]));
      }
      nxt[tid] = d;
    }
    __syncthreads();
    G4ComplexPair* tmp = cur;
    cur = nxt;
    nxt = tmp;
  }

  if (tid < ngrid) {
    const int idx = base + tid * ngrid;
    const G4ComplexPair u = g4_div(cur[tid], g4_make_complex(pcr.b_re[tid], pcr.b_im[tid]));
    field.field_re[idx] = u.re;
    field.field_im[idx] = u.im;
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
  const std::size_t shmem = 2 * static_cast<std::size_t>(block) * sizeof(G4ComplexPair);
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
  const std::size_t shmem = 2 * static_cast<std::size_t>(block) * sizeof(G4ComplexPair);
  adi_solve_y_pcr_kernel<<<nlines, block, shmem, g4_cuda_stream()>>>(ngrid,
                                                                            pcr,
                                                                            field);
  return cudaGetLastError() == cudaSuccess;
}

} // namespace


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
  const double* source_re = nullptr;
  const double* source_im = nullptr;
  if (do_source && total_particles > 0) {
    d_source_re.resize_discard(total_cells);
    d_source_im.resize_discard(total_cells);
    adi_zero_source_grid(total_cells, d_source_re.data(), d_source_im.data());
    adi_deposit_source_grid(total_particles,
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
  if (!adi_try_solve_x_pcr(nlines, ngrid_i, pcr_factors, adi_field)) {
    adi_solve_x_thomas(nslice, ngrid_i, adi_field, adi_coeff);
  }

  adi_build_rhs_x_laplacian(total_cells, ngrid_i, cstep_im, source_re, source_im, adi_field);
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
  vector<complex<double> > c_host(ngrid);
  vector<complex<double> > cbet_host(ngrid);
  vector<complex<double> > cwet_host(ngrid);
  vector<complex<double> > cwrk1(ngrid);
  vector<complex<double> > cwrk2(ngrid);

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
    cwrk1[i] = complex<double>(0, -mupp[i]);
    cwrk2[i] = complex<double>(1, -mmid[i]);
    c_host[i] = complex<double>(0, -mlow[i]);
  }

  cbet_host[0] = 1. / cwrk2[0];
  cwet_host[0] = 0.;
  for (unsigned int i = 1; i < ngrid; i++) {
    cwet_host[i] = cwrk1[i - 1] * cbet_host[i - 1];
    cbet_host[i] = 1. / (cwrk2[i] - c_host[i] * cwet_host[i]);
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
