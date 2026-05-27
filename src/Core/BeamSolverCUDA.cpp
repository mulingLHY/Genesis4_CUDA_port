#include "BeamSolverCUDA.h"
#include "Genesis4CudaLaunch.h"
#include "Genesis4CudaRuntime.h"

#include "Field.h"
#include "Beam.h"
#include "Particle.h"

#include "Genesis4BeamSoA.h"
#include "Genesis4FieldSoA.h"
#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

namespace {
constexpr int kMaxBeamSolverFields = 8;

struct BeamParticleView {
  const int* slice_id;
  double* x;
  double* y;
  const double* px;
  const double* py;
  double* gamma;
  double* theta;
};

struct BeamFieldView {
  const int* first;
  const int* ngrid;
  const int* nslice;
  const double* gridmax;
  const double* dgrid;
  const double* rtmp;
  const double* rharm;
  const double* const* field_re;
  const double* const* field_im;
  int count;
};

struct BeamSingleFieldView {
  int first;
  int ngrid;
  int nslice;
  double gridmax;
  double dgrid;
  double rtmp;
  double rharm;
  const double* field_re;
  const double* field_im;
};

struct BeamAdvanceParams {
  double xks;
  double xku;
  double aw;
  double autophase;
  double delz;
  double ax;
  double ay;
  double kx;
  double ky;
  double gradx;
  double grady;
};

template <typename Vector>
void resize_if_needed(Vector& vec, std::size_t size)
{
  if (vec.size() < size) {
    vec.resize(size);
  }
}

GENESIS4_CUDA_HOST_DEVICE GENESIS4_CUDA_FORCE_INLINE
double local_undulator_factor(double x,
                              double y,
                              BeamAdvanceParams params) noexcept
{
  const double dx = x - params.ax;
  const double dy = y - params.ay;
  return 1.0 + 0.5 * (params.kx * dx * dx + params.ky * dy * dy)
       + params.gradx * dx + params.grady * dy;
}

GENESIS4_CUDA_HOST_DEVICE GENESIS4_CUDA_FORCE_INLINE
void sample_single_field(BeamSingleFieldView field,
                         int slice,
                         double x,
                         double y,
                         double awloc,
                         double& rpart_re,
                         double& rpart_im) noexcept
{
  rpart_re = 0.0;
  rpart_im = 0.0;

  const int ng = field.ngrid;
  const double gm = field.gridmax;
  if ((x <= -gm) || (x >= gm) || (y <= -gm) || (y >= gm)) {
    return;
  }

  double wx = (x + gm) / field.dgrid;
  double wy = (y + gm) / field.dgrid;
  const int ix_raw = static_cast<int>(floor(wx));
  const int iy_raw = static_cast<int>(floor(wy));
  if (ix_raw < 0 || ix_raw >= ng - 1 || iy_raw < 0 || iy_raw >= ng - 1) {
    return;
  }

  const int ix = ix_raw;
  const int iy = iy_raw;
  wx = 1.0 + static_cast<double>(ix) - wx;
  wy = 1.0 + static_cast<double>(iy) - wy;

  const int islice = (slice + field.first) % field.nslice;
  const int plane = ng * ng;
  int idx = ix + iy * ng;
  const int base = islice * plane;

  const double w00 = wx * wy;
  const double w10 = (1.0 - wx) * wy;
  const double w01 = wx * (1.0 - wy);
  const double w11 = (1.0 - wx) * (1.0 - wy);

  double cre = field.field_re[base + idx] * w00;
  double cim = field.field_im[base + idx] * w00;
  ++idx;
  cre += field.field_re[base + idx] * w10;
  cim += field.field_im[base + idx] * w10;
  idx += ng - 1;
  cre += field.field_re[base + idx] * w01;
  cim += field.field_im[base + idx] * w01;
  ++idx;
  cre += field.field_re[base + idx] * w11;
  cim += field.field_im[base + idx] * w11;

  const double scale = field.rtmp * awloc;
  rpart_re = scale * cre;
  rpart_im = -scale * cim;
}

GENESIS4_CUDA_HOST_DEVICE GENESIS4_CUDA_FORCE_INLINE
void evaluate_longitudinal_ode_multi_field(double tgam,
                                           double tthet,
                                           double btpar,
                                           double ez,
                                           double xks,
                                           double xku,
                                           int nfield,
                                           const double* rharm,
                                           const double* rpart_re,
                                           const double* rpart_im,
                                           double& k2gg,
                                           double& k2pp) noexcept
{
  const double ztemp1 = -2.0 / xks;
  double ctmp_re = 0.0;
  double ctmp_im = 0.0;

  for (int i = 0; i < nfield; ++i) {
    const double arg = rharm[i] * tthet;
    const double c = cos(arg);
    const double s = sin(arg);
    const double rr = rpart_re[i];
    const double ri = rpart_im[i];

    // (rr + i*ri) * (cos(arg) - i*sin(arg))
    ctmp_re += rr * c + ri * s;
    ctmp_im += ri * c - rr * s;
  }

  const double btper0 = btpar + ztemp1 * ctmp_re;
  const double btpar0 = sqrt(1.0 - btper0 / (tgam * tgam));

  k2pp += xks * (1.0 - 1.0 / btpar0) + xku;
  k2gg += ctmp_im / btpar0 / tgam - ez;
}


template <bool UseParticleEz>
GENESIS4_CUDA_HOST_DEVICE GENESIS4_CUDA_FORCE_INLINE
double load_longitudinal_ez(int ip_global,
                            int slice,
                            const double* particle_ez,
                            const double* slice_eloss) noexcept
{
  if constexpr (UseParticleEz) {
    return particle_ez[ip_global];
  } else {
    return slice_eloss[slice];
  }
}

GENESIS4_CUDA_HOST_DEVICE GENESIS4_CUDA_FORCE_INLINE
void evaluate_longitudinal_ode_single_field(double tgam,
                                            double tthet,
                                            double btpar,
                                            double ez,
                                            double xks,
                                            double xku,
                                            double rharm,
                                            double rpart_re,
                                            double rpart_im,
                                            double& k2gg,
                                            double& k2pp) noexcept
{
  const double arg = rharm * tthet;
  const double c = cos(arg);
  const double s = sin(arg);

  const double ctmp_re = rpart_re * c + rpart_im * s;
  const double ctmp_im = rpart_im * c - rpart_re * s;

  const double ztemp1 = -2.0 / xks;
  const double btper0 = btpar + ztemp1 * ctmp_re;
  const double btpar0 = sqrt(1.0 - btper0 / (tgam * tgam));

  k2pp += xks * (1.0 - 1.0 / btpar0) + xku;
  k2gg += ctmp_im / btpar0 / tgam - ez;
}

template <bool UseParticleEz>
void push_longitudinal_particles_single_field(int total_particles,
                                              BeamParticleView particles,
                                              const double* particle_ez,
                                              const double* slice_eloss,
                                              BeamSingleFieldView field,
                                              BeamAdvanceParams params)
{
  // Keep the extended __device__ lambda in a namespace-scope helper.  NVCC
  // rejects extended device lambdas whose enclosing function is a private or
  // protected class member.
  g4_parallel_for(total_particles,
  [=] GENESIS4_CUDA_DEVICE (int ip_global) noexcept {
    const int is = particles.slice_id[ip_global];

    double gamma = particles.gamma[ip_global];
    double theta = particles.theta[ip_global] + params.autophase;

    const double x_particle = particles.x[ip_global];
    const double y_particle = particles.y[ip_global];
    const double px_particle = particles.px[ip_global];
    const double py_particle = particles.py[ip_global];

    const double awloc = local_undulator_factor(x_particle, y_particle, params);
    const double btpar = 1.0 + px_particle * px_particle + py_particle * py_particle
                       + params.aw * params.aw * awloc * awloc;

    const double ez = load_longitudinal_ez<UseParticleEz>(ip_global, is, particle_ez, slice_eloss);

    double rpart_re = 0.0;
    double rpart_im = 0.0;
    sample_single_field(field, is, x_particle, y_particle, awloc, rpart_re, rpart_im);

    double k2gg = 0.0;
    double k2pp = 0.0;
    double k3gg = 0.0;
    double k3pp = 0.0;

    evaluate_longitudinal_ode_single_field(gamma, theta, btpar, ez, params.xks, params.xku,
                                           field.rharm, rpart_re, rpart_im, k2gg, k2pp);

    double stpz = 0.5 * params.delz;
    gamma += stpz * k2gg;
    theta += stpz * k2pp;
    k3gg = k2gg;
    k3pp = k2pp;
    k2gg = 0.0;
    k2pp = 0.0;

    evaluate_longitudinal_ode_single_field(gamma, theta, btpar, ez, params.xks, params.xku,
                                           field.rharm, rpart_re, rpart_im, k2gg, k2pp);

    gamma += stpz * (k2gg - k3gg);
    theta += stpz * (k2pp - k3pp);
    k3gg /= 6.0;
    k3pp /= 6.0;
    k2gg *= -0.5;
    k2pp *= -0.5;

    evaluate_longitudinal_ode_single_field(gamma, theta, btpar, ez, params.xks, params.xku,
                                           field.rharm, rpart_re, rpart_im, k2gg, k2pp);

    stpz = params.delz;
    gamma += stpz * k2gg;
    theta += stpz * k2pp;
    k3gg -= k2gg;
    k3pp -= k2pp;
    k2gg *= 2.0;
    k2pp *= 2.0;

    evaluate_longitudinal_ode_single_field(gamma, theta, btpar, ez, params.xks, params.xku,
                                           field.rharm, rpart_re, rpart_im, k2gg, k2pp);

    gamma += stpz * (k3gg + k2gg / 6.0);
    theta += stpz * (k3pp + k2pp / 6.0);

    particles.gamma[ip_global] = gamma;
    particles.theta[ip_global] = theta;
  });
}

template <bool UseParticleEz>
void push_longitudinal_particles_multi_field(int total_particles,
                                             BeamParticleView particles,
                                             const double* particle_ez,
                                             const double* slice_eloss,
                                             BeamFieldView fields,
                                             BeamAdvanceParams params)
{
  // Keep the extended __device__ lambda in a namespace-scope helper.  NVCC
  // rejects extended device lambdas whose enclosing function is a private or
  // protected class member.
  g4_parallel_for(total_particles,
  [=] GENESIS4_CUDA_DEVICE (int ip_global) noexcept {
    const int is = particles.slice_id[ip_global];

    double gamma = particles.gamma[ip_global];
    double theta = particles.theta[ip_global] + params.autophase;

    const double x_particle = particles.x[ip_global];
    const double y_particle = particles.y[ip_global];
    const double px_particle = particles.px[ip_global];
    const double py_particle = particles.py[ip_global];

    const double awloc = local_undulator_factor(x_particle, y_particle, params);
    const double btpar = 1.0 + px_particle * px_particle + py_particle * py_particle
                       + params.aw * params.aw * awloc * awloc;

    const double ez = load_longitudinal_ez<UseParticleEz>(ip_global, is, particle_ez, slice_eloss);

    double rpart_re_stack[kMaxBeamSolverFields];
    double rpart_im_stack[kMaxBeamSolverFields];
    for (int ifld = 0; ifld < fields.count; ++ifld) {
      rpart_re_stack[ifld] = 0.0;
      rpart_im_stack[ifld] = 0.0;
    }

    for (int ifld = 0; ifld < fields.count; ++ifld) {
      const BeamSingleFieldView field {
        fields.first[ifld],
        fields.ngrid[ifld],
        fields.nslice[ifld],
        fields.gridmax[ifld],
        fields.dgrid[ifld],
        fields.rtmp[ifld],
        fields.rharm[ifld],
        fields.field_re[ifld],
        fields.field_im[ifld]
      };

      sample_single_field(field, is, x_particle, y_particle, awloc,
                          rpart_re_stack[ifld], rpart_im_stack[ifld]);
    }

    double k2gg = 0.0;
    double k2pp = 0.0;
    double k3gg = 0.0;
    double k3pp = 0.0;

    evaluate_longitudinal_ode_multi_field(gamma, theta, btpar, ez, params.xks, params.xku,
                                          fields.count, fields.rharm,
                                          rpart_re_stack, rpart_im_stack, k2gg, k2pp);

    double stpz = 0.5 * params.delz;
    gamma += stpz * k2gg;
    theta += stpz * k2pp;
    k3gg = k2gg;
    k3pp = k2pp;
    k2gg = 0.0;
    k2pp = 0.0;

    evaluate_longitudinal_ode_multi_field(gamma, theta, btpar, ez, params.xks, params.xku,
                                          fields.count, fields.rharm,
                                          rpart_re_stack, rpart_im_stack, k2gg, k2pp);

    gamma += stpz * (k2gg - k3gg);
    theta += stpz * (k2pp - k3pp);
    k3gg /= 6.0;
    k3pp /= 6.0;
    k2gg *= -0.5;
    k2pp *= -0.5;

    evaluate_longitudinal_ode_multi_field(gamma, theta, btpar, ez, params.xks, params.xku,
                                          fields.count, fields.rharm,
                                          rpart_re_stack, rpart_im_stack, k2gg, k2pp);

    stpz = params.delz;
    gamma += stpz * k2gg;
    theta += stpz * k2pp;
    k3gg -= k2gg;
    k3pp -= k2pp;
    k2gg *= 2.0;
    k2pp *= 2.0;

    evaluate_longitudinal_ode_multi_field(gamma, theta, btpar, ez, params.xks, params.xku,
                                          fields.count, fields.rharm,
                                          rpart_re_stack, rpart_im_stack, k2gg, k2pp);

    gamma += stpz * (k3gg + k2gg / 6.0);
    theta += stpz * (k3pp + k2pp / 6.0);

    particles.gamma[ip_global] = gamma;
    particles.theta[ip_global] = theta;
  });
}

} // namespace


struct BeamSolverCUDA::Buffers {
  std::vector<double> h_particle_ez;
  CudaDeviceBuffer<double> d_particle_ez;
  std::vector<double> h_slice_eloss;
  CudaDeviceBuffer<double> d_slice_eloss;

  std::vector<int> h_first;
  std::vector<int> h_ngrid;
  std::vector<int> h_nslice;
  std::vector<double> h_gridmax;
  std::vector<double> h_dgrid;
  std::vector<double> h_rtmp;
  std::vector<double> h_rharm;
  std::vector<const double*> h_field_re;
  std::vector<const double*> h_field_im;

  CudaDeviceBuffer<int> d_first;
  CudaDeviceBuffer<int> d_ngrid;
  CudaDeviceBuffer<int> d_nslice;
  CudaDeviceBuffer<double> d_gridmax;
  CudaDeviceBuffer<double> d_dgrid;
  CudaDeviceBuffer<double> d_rtmp;
  CudaDeviceBuffer<double> d_rharm;
  CudaDeviceBuffer<const double*> d_field_re;
  CudaDeviceBuffer<const double*> d_field_im;
};

BeamSolverCUDA::BeamSolverCUDA()
  : buffers(std::make_unique<Buffers>())
{
  onlyFundamental=false;
}

BeamSolverCUDA::~BeamSolverCUDA()= default;

void BeamSolverCUDA::advance(double delz, Beam *beam, vector<Field *> *field, Undulator *und) {
  if (beam == nullptr || field == nullptr || und == nullptr) {
    g4_cuda_abort("BeamSolverCUDA::advance received a null beam/field/undulator pointer");
  }

  Genesis4BeamSoA* bsoa = beam->beamSoA;
  if (bsoa == nullptr || !bsoa->initialized || bsoa->total_particles <= 0 || bsoa->nslice <= 0) {
    return;
  }

  const int nslice = bsoa->nslice;
  const int total_particles = bsoa->total_particles;
  Buffers& work = *buffers;

  // Build the active harmonic/field list on host.  Only POD metadata and raw SoA
  // pointers are copied into GPU-visible vectors; no std::vector or class methods
  // are called inside device code.
  std::vector<int> active_first;
  std::vector<int> active_ngrid;
  std::vector<int> active_nslice;
  std::vector<double> active_gridmax;
  std::vector<double> active_dgrid;
  std::vector<double> active_rtmp;
  std::vector<double> active_rharm;
  std::vector<const double*> active_field_re;
  std::vector<const double*> active_field_im;

  double xks_local = 1.0;
  for (int i = 0; i < static_cast<int>(field->size()); ++i) {
    Field* fld = field->at(i);
    if (fld == nullptr) { continue; }

    const int harm = fld->getHarm();
    if ((harm != 1) && onlyFundamental) { continue; }

    if (fld->fieldSoA == nullptr || !fld->fieldSoA->initialized) {
      fld->pack_field_to_soa();
    }

    Genesis4FieldSoA* fsoa = fld->fieldSoA;
    if (fsoa == nullptr || !fsoa->initialized || fld->ngrid <= 1) {
      g4_cuda_abort("BeamSolverCUDA::advance requires initialized FieldSoA");
    }

    const int field_nslice = fsoa->nslice > 0
                               ? fsoa->nslice
                               : static_cast<int>(fld->field.size());

    xks_local = fld->xks / static_cast<double>(harm);
    active_first.push_back(fld->first);
    active_ngrid.push_back(fld->ngrid);
    active_nslice.push_back(field_nslice);
    active_gridmax.push_back(fld->gridmax);
    active_dgrid.push_back(fld->dgrid);
    active_rtmp.push_back(und->fc(harm) / fld->xks);
    active_rharm.push_back(static_cast<double>(harm));
    active_field_re.push_back(fsoa->field_re.data());
    active_field_im.push_back(fsoa->field_im.data());
  }

  const int nfield = static_cast<int>(active_rharm.size());
  if (nfield > kMaxBeamSolverFields) {
    g4_cuda_abort("BeamSolverCUDA::advance exceeds kMaxBeamSolverFields");
  }

  double xku_local = und->getku();
  if (xku_local == 0.0) {
    const double gamma_ref = und->getGammaRef();
    xku_local = xks_local * 0.5 / gamma_ref / gamma_ref;
  }

  const double aw = und->getaw();
  const double autophase = und->autophase();
  const double gammaz2 = und->getGammaRef() * und->getGammaRef() / (1.0 + aw * aw);
  const bool use_short_range = efield.hasShortRange();

  if (use_short_range) {
    resize_if_needed(work.h_particle_ez, static_cast<std::size_t>(total_particles));
    resize_if_needed(work.d_particle_ez, static_cast<std::size_t>(total_particles));

    int particle_offset = 0;
    for (int is = 0; is < nslice; ++is) {
      const int np = static_cast<int>(beam->beam.at(is).size());
      const double eloss = -beam->longESC[is] / 511000.0;

      efield.shortRange(&beam->beam.at(is), beam->current.at(is), gammaz2, is);
      for (int ip = 0; ip < np; ++ip) {
        work.h_particle_ez[particle_offset + ip] = efield.getEField(ip) + eloss;
      }
      particle_offset += np;
    }

    if (particle_offset != total_particles) {
      g4_cuda_abort("BeamSolverCUDA::advance requires CPU beam slice sizes to match BeamSoA layout");
    }

    genesis4_cuda::copy_host_to_device(work.h_particle_ez.begin(),
                                       work.h_particle_ez.begin() + total_particles,
                                       work.d_particle_ez.begin());
  } else {
    resize_if_needed(work.h_slice_eloss, static_cast<std::size_t>(nslice));
    resize_if_needed(work.d_slice_eloss, static_cast<std::size_t>(nslice));

    std::vector<Particle> empty_slice;
    for (int is = 0; is < nslice; ++is) {
      efield.shortRange(&empty_slice, 0.0, gammaz2, is);
      work.h_slice_eloss[is] = -beam->longESC[is] / 511000.0;
    }

    genesis4_cuda::copy_host_to_device(work.h_slice_eloss.begin(),
                                       work.h_slice_eloss.begin() + nslice,
                                       work.d_slice_eloss.begin());
  }

  if (nfield > 1) {
    resize_if_needed(work.h_first, static_cast<std::size_t>(nfield));
    resize_if_needed(work.h_ngrid, static_cast<std::size_t>(nfield));
    resize_if_needed(work.h_nslice, static_cast<std::size_t>(nfield));
    resize_if_needed(work.h_gridmax, static_cast<std::size_t>(nfield));
    resize_if_needed(work.h_dgrid, static_cast<std::size_t>(nfield));
    resize_if_needed(work.h_rtmp, static_cast<std::size_t>(nfield));
    resize_if_needed(work.h_rharm, static_cast<std::size_t>(nfield));
    resize_if_needed(work.h_field_re, static_cast<std::size_t>(nfield));
    resize_if_needed(work.h_field_im, static_cast<std::size_t>(nfield));

    resize_if_needed(work.d_first, static_cast<std::size_t>(nfield));
    resize_if_needed(work.d_ngrid, static_cast<std::size_t>(nfield));
    resize_if_needed(work.d_nslice, static_cast<std::size_t>(nfield));
    resize_if_needed(work.d_gridmax, static_cast<std::size_t>(nfield));
    resize_if_needed(work.d_dgrid, static_cast<std::size_t>(nfield));
    resize_if_needed(work.d_rtmp, static_cast<std::size_t>(nfield));
    resize_if_needed(work.d_rharm, static_cast<std::size_t>(nfield));
    resize_if_needed(work.d_field_re, static_cast<std::size_t>(nfield));
    resize_if_needed(work.d_field_im, static_cast<std::size_t>(nfield));

    for (int i = 0; i < nfield; ++i) {
      work.h_first[i] = active_first[i];
      work.h_ngrid[i] = active_ngrid[i];
      work.h_nslice[i] = active_nslice[i];
      work.h_gridmax[i] = active_gridmax[i];
      work.h_dgrid[i] = active_dgrid[i];
      work.h_rtmp[i] = active_rtmp[i];
      work.h_rharm[i] = active_rharm[i];
      work.h_field_re[i] = active_field_re[i];
      work.h_field_im[i] = active_field_im[i];
    }

    genesis4_cuda::copy_host_to_device(work.h_first.begin(), work.h_first.begin() + nfield, work.d_first.begin());
    genesis4_cuda::copy_host_to_device(work.h_ngrid.begin(), work.h_ngrid.begin() + nfield, work.d_ngrid.begin());
    genesis4_cuda::copy_host_to_device(work.h_nslice.begin(), work.h_nslice.begin() + nfield, work.d_nslice.begin());
    genesis4_cuda::copy_host_to_device(work.h_gridmax.begin(), work.h_gridmax.begin() + nfield, work.d_gridmax.begin());
    genesis4_cuda::copy_host_to_device(work.h_dgrid.begin(), work.h_dgrid.begin() + nfield, work.d_dgrid.begin());
    genesis4_cuda::copy_host_to_device(work.h_rtmp.begin(), work.h_rtmp.begin() + nfield, work.d_rtmp.begin());
    genesis4_cuda::copy_host_to_device(work.h_rharm.begin(), work.h_rharm.begin() + nfield, work.d_rharm.begin());
    genesis4_cuda::copy_host_to_device(work.h_field_re.begin(), work.h_field_re.begin() + nfield, work.d_field_re.begin());
    genesis4_cuda::copy_host_to_device(work.h_field_im.begin(), work.h_field_im.begin() + nfield, work.d_field_im.begin());
  }

  const int istep = und->getStep();

  BeamParticleView particles {
    bsoa->slice_id.data(),
    bsoa->x.data(),
    bsoa->y.data(),
    bsoa->px.data(),
    bsoa->py.data(),
    bsoa->gamma.data(),
    bsoa->theta.data()
  };

  BeamAdvanceParams params {
    xks_local,
    xku_local,
    aw,
    autophase,
    delz,
    und->ax[istep],
    und->ay[istep],
    und->kx[istep],
    und->ky[istep],
    und->gradx[istep],
    und->grady[istep]
  };

  const double* particle_ez = use_short_range ? work.d_particle_ez.data() : nullptr;
  const double* slice_eloss = use_short_range ? nullptr : work.d_slice_eloss.data();

  if (nfield == 1) {
    BeamSingleFieldView single_field {
      active_first[0],
      active_ngrid[0],
      active_nslice[0],
      active_gridmax[0],
      active_dgrid[0],
      active_rtmp[0],
      active_rharm[0],
      active_field_re[0],
      active_field_im[0]
    };

    if (use_short_range) {
      push_longitudinal_particles_single_field<true>(total_particles, particles,
                                                     particle_ez, slice_eloss,
                                                     single_field, params);
    } else {
      push_longitudinal_particles_single_field<false>(total_particles, particles,
                                                      particle_ez, slice_eloss,
                                                      single_field, params);
    }
    return;
  }

  BeamFieldView fields {
    nfield > 1 ? work.d_first.data() : nullptr,
    nfield > 1 ? work.d_ngrid.data() : nullptr,
    nfield > 1 ? work.d_nslice.data() : nullptr,
    nfield > 1 ? work.d_gridmax.data() : nullptr,
    nfield > 1 ? work.d_dgrid.data() : nullptr,
    nfield > 1 ? work.d_rtmp.data() : nullptr,
    nfield > 1 ? work.d_rharm.data() : nullptr,
    nfield > 1 ? work.d_field_re.data() : nullptr,
    nfield > 1 ? work.d_field_im.data() : nullptr,
    nfield
  };

  if (use_short_range) {
    push_longitudinal_particles_multi_field<true>(total_particles, particles,
                                                  particle_ez, slice_eloss,
                                                  fields, params);
  } else {
    push_longitudinal_particles_multi_field<false>(total_particles, particles,
                                                   particle_ez, slice_eloss,
                                                   fields, params);
  }
}

void BeamSolverCUDA::checkAllocation(unsigned long nslice) {
  efield.allocateForOutput(nslice);
}
