#ifndef __GENESIS_BEAMSOLVERCUDA__
#define __GENESIS_BEAMSOLVERCUDA__

#include <memory>
#include <vector>


class Field;
class Beam;

#include "Undulator.h"
#include "EFieldSolver.h"
#include "TrackBeamCUDA.h"

using namespace std;

class BeamSolverCUDA{
 public:
  BeamSolverCUDA();
  virtual ~BeamSolverCUDA();

  void initEField(double rmax, int ngrid, int nz, int nphi, double lambda, bool longr);
  void advance(double, Beam *, vector<Field *> *, Undulator *);
  void track(double, Beam *, Undulator *, bool);
  void applyR56(Beam *, Undulator *, double);
  double getSCField(int);
  void checkAllocation(unsigned long i);

 private:

  bool onlyFundamental;
  EFieldSolver efield;
  TrackBeamCUDA tracker;

  struct Buffers;
  std::unique_ptr<Buffers> buffers;
};

inline double BeamSolverCUDA::getSCField(int islice){ return efield.getSCField(islice); }

inline void BeamSolverCUDA::initEField(double rmax, int ngrid, int nz, int nphi, double lambda, bool longr){
  efield.init(rmax,ngrid,nz,nphi,lambda,longr);
}

inline void BeamSolverCUDA::track(double dz, Beam *beam, Undulator *und, bool last) {
  tracker.track(dz,beam,und,last);
}

inline void BeamSolverCUDA::applyR56(Beam *beam, Undulator *und, double reflen){
  tracker.applyR56(beam,und,reflen);
}

#endif
