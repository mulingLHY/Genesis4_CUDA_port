#include "TrackBeamCUDA.h"
#include "Genesis4CudaLaunch.h"
#include "Beam.h"
#include "Undulator.h"

#include "Genesis4BeamSoA.h"
#include <cmath>

namespace {

enum class FocusingMode : int {
  Drift = 0,
  Focusing = 1,
  Defocusing = 2
};

struct PlaneMap {
  FocusingMode mode;
  double strength;
  double offset;
};

struct Matrix4 {
  double v[4][4];
};

GENESIS4_CUDA_HOST_DEVICE GENESIS4_CUDA_FORCE_INLINE
void advance_focusing_plane(const PlaneMap map,
                            double delz,
                            double& coord,
                            double& momentum,
                            double gammaz) noexcept
{
  if (map.mode == FocusingMode::Drift) {
    coord += momentum * delz / gammaz;
    return;
  }

  if (map.mode == FocusingMode::Focusing) {
    const double foc = std::sqrt(map.strength / gammaz);
    const double omg = foc * delz;
    const double a1 = std::cos(omg);
    const double a2 = std::sin(omg) / foc;
    const double a3 = -a2 * foc * foc;
    const double xtmp = coord - map.offset;
    coord = a1 * xtmp + a2 * momentum / gammaz + map.offset;
    momentum = a3 * xtmp * gammaz + a1 * momentum;
    return;
  }

  const double foc = std::sqrt(-map.strength / gammaz);
  const double omg = foc * delz;
  const double a1 = std::cosh(omg);
  const double a2 = std::sinh(omg) / foc;
  const double a3 = a2 * foc * foc;
  const double xtmp = coord - map.offset;
  coord = a1 * xtmp + a2 * momentum / gammaz + map.offset;
  momentum = a3 * xtmp * gammaz + a1 * momentum;
}

PlaneMap make_plane_map(double strength, double offset)
{
  PlaneMap map {FocusingMode::Drift, strength, 0.0};
  if (strength == 0.0) { return map; }

  map.offset = offset / strength;
  map.mode = (strength > 0.0) ? FocusingMode::Focusing : FocusingMode::Defocusing;
  return map;
}

Matrix4 identity_matrix()
{
  Matrix4 m{};
  for (int i = 0; i < 4; ++i) {
    m.v[i][i] = 1.0;
  }
  return m;
}

void left_multiply(Matrix4& m, const Matrix4& e)
{
  Matrix4 t{};
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 4; ++k) {
        t.v[i][j] += e.v[i][k] * m.v[k][j];
      }
    }
  }
  m = t;
}

Matrix4 build_chicane_matrix(double angle, double lb, double ld, double lt)
{
  Matrix4 m = identity_matrix();
  Matrix4 d1 = identity_matrix();
  Matrix4 d2 = identity_matrix();
  Matrix4 d3 = identity_matrix();
  Matrix4 bp = identity_matrix();
  Matrix4 bn = identity_matrix();
  Matrix4 ep = identity_matrix();
  Matrix4 en = identity_matrix();

  d1.v[0][1] = ld / std::cos(angle);  // drift between dipoles
  d1.v[2][3] = ld / std::cos(angle);
  d2.v[0][1] = lt - 4 * lb - 2 * ld;  // drift in the middle
  d2.v[2][3] = lt - 4 * lb - 2 * ld;
  d3.v[0][1] = -lt;                   // negative drift over total chicane to get a zero element
  d3.v[2][3] = -lt;

  const double radius = lb / std::sin(angle);
  const double path_length = radius * angle;

  bp.v[2][3] = path_length;  // positive deflection angle
  bp.v[0][0] = std::cos(angle);
  bp.v[0][1] = radius * std::sin(angle);
  bp.v[1][0] = -std::sin(angle) / radius;
  bp.v[1][1] = std::cos(angle);

  bn.v[2][3] = path_length;  // negative deflection angle
  bn.v[0][0] = std::cos(-angle);
  bn.v[0][1] = radius * std::sin(-angle) * -1;
  bn.v[1][0] = -std::sin(-angle) / radius * -1;
  bn.v[1][1] = std::cos(-angle);

  const double edge_focusing = std::tan(angle) / radius;
  ep.v[1][0] = edge_focusing;
  ep.v[3][2] = -edge_focusing;
  en.v[1][0] = -edge_focusing * -1;
  en.v[3][2] = edge_focusing * -1;

  // the transfer matrix order is
  //  m -> bp -> ep -> d1 -> en -> bn -> d2 -> bn -> en-> d1 -> ep-> bp ->d3
  left_multiply(m, bp);
  left_multiply(m, ep);
  left_multiply(m, d1);
  left_multiply(m, en);
  left_multiply(m, bn);
  left_multiply(m, d2);
  left_multiply(m, bn);
  left_multiply(m, en);
  left_multiply(m, d1);
  left_multiply(m, ep);
  left_multiply(m, bp);
  left_multiply(m, d3);  // transport backwards because the main tracking still has to do the drift

  return m;
}

} // namespace


TrackBeamCUDA::TrackBeamCUDA(){}
TrackBeamCUDA::~TrackBeamCUDA(){}

void TrackBeamCUDA::track(double delz, Beam *beam, Undulator *und, bool lastStep)
{
  double aw, dax, day, ku, kx, ky;
  double qf, dqx, dqy;
  double cx, cy;
  double angle, lb, ld, lt;

  const double gamma0 = und->getGammaRef();
  und->getUndulatorParameters(&aw, &dax, &day, &ku, &kx, &ky);
  und->getQuadrupoleParameters(&qf, &dqx, &dqy);
  und->getCorrectorParameters(&cx, &cy);
  und->getChicaneParameters(&angle, &lb, &ld, &lt);

  const double betpar0 = std::sqrt(1 - (1 + aw * aw) / gamma0 / gamma0);

  // effective focusing in x and y with the correct energy dependence
  const double quad_strength = qf * gamma0;
  const double qnatx = kx * aw * aw / gamma0 / betpar0;  // kx has already the scaling with ku^2
  const double qnaty = ky * aw * aw / gamma0 / betpar0;  // same with ky

  const double qx = quad_strength + qnatx;
  const double qy = -quad_strength + qnaty;
  const double xoff = quad_strength * dqx + qnatx * dax;
  const double yoff = -quad_strength * dqy + qnaty * day;

  if (lastStep) {
    if ((cx != 0) || (cy != 0)) { this->applyCorrector(beam, cx * gamma0, cy * gamma0); }
  } else {
    if (angle != 0) { this->applyChicane(beam, angle, lb, ld, lt, gamma0); }
  }

  const PlaneMap xmap = make_plane_map(qx, xoff);
  const PlaneMap ymap = make_plane_map(qy, yoff);

  Genesis4BeamSoA* soa = beam->beamSoA;
  const long n = static_cast<long>(soa->total_particles);

  double* gamma = soa->gamma.data();
  double* x = soa->x.data();
  double* y = soa->y.data();
  double* px = soa->px.data();
  double* py = soa->py.data();

  g4_parallel_for(n, [=] GENESIS4_CUDA_DEVICE (long i) noexcept {
    const double gammaz = std::sqrt(gamma[i] * gamma[i] - 1.0 - aw * aw - px[i] * px[i] - py[i] * py[i]);
    advance_focusing_plane(xmap, delz, x[i], px[i], gammaz);
    advance_focusing_plane(ymap, delz, y[i], py[i], gammaz);
  });
}

void TrackBeamCUDA::applyCorrector(Beam *beam, double cx, double cy)
{
  Genesis4BeamSoA* soa = beam->beamSoA;
  const long n = static_cast<long>(soa->total_particles);

  double* px = soa->px.data();
  double* py = soa->py.data();

  g4_parallel_for(n, [=] GENESIS4_CUDA_DEVICE (long i) noexcept {
    px[i] += cx;
    py[i] += cy;
  });
}

void TrackBeamCUDA::applyChicane(Beam *beam, double angle, double lb, double ld, double lt, double gamma0)
{
  // the tracking is done my applying the transfer matrix for the chicane and  backtracking for a drift over the length of the chicane
  // the effect of the R56 is applied here to the particle phase.  Then the normal tracking should do the drift
  const Matrix4 m = build_chicane_matrix(angle, lb, ld, lt);

  Genesis4BeamSoA* soa = beam->beamSoA;
  const long n = static_cast<long>(soa->total_particles);

  double* gamma = soa->gamma.data();
  double* x = soa->x.data();
  double* y = soa->y.data();
  double* px = soa->px.data();
  double* py = soa->py.data();

  const double m00 = m.v[0][0];
  const double m01 = m.v[0][1];
  const double m10 = m.v[1][0];
  const double m11 = m.v[1][1];
  const double m22 = m.v[2][2];
  const double m23 = m.v[2][3];
  const double m32 = m.v[3][2];
  const double m33 = m.v[3][3];

  g4_parallel_for(n, [=] GENESIS4_CUDA_DEVICE (long i) noexcept {
    const double gammaz = std::sqrt(gamma[i] * gamma[i] - 1.0 - px[i] * px[i] - py[i] * py[i]);

    double tmp = x[i];
    x[i] = m00 * tmp + m01 * px[i] / gammaz;
    px[i] = m10 * tmp * gammaz + m11 * px[i];

    tmp = y[i];
    y[i] = m22 * tmp + m23 * py[i] / gammaz;
    py[i] = m32 * tmp * gammaz + m33 * py[i];
  });
}

void TrackBeamCUDA::applyR56(Beam *beam, Undulator *und, double lambda0)
{
  double angle, lb, ld, lt;

  const double gamma0 = und->getGammaRef();
  und->getChicaneParameters(&angle, &lb, &ld, &lt);
  if (angle == 0) { return; }

  double R56 = (4 * lb / std::sin(angle) * (1 - angle / std::tan(angle))
             + 2 * ld * std::tan(angle) / std::cos(angle)) * angle;
  R56 = R56 * 4 * std::asin(1) / lambda0 / gamma0;

  Genesis4BeamSoA* soa = beam->beamSoA;
  const long n = static_cast<long>(soa->total_particles);

  double* gamma = soa->gamma.data();
  double* theta = soa->theta.data();

  g4_parallel_for(n, [=] GENESIS4_CUDA_DEVICE (long i) noexcept {
    theta[i] += R56 * (gamma[i] - gamma0);
  });
}
