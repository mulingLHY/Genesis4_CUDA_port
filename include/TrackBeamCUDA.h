#ifndef __GENESIS_TRACKBEAMCUDA__
#define __GENESIS_TRACKBEAMCUDA__

class Beam;
class Undulator;

class TrackBeamCUDA{
 public:
  TrackBeamCUDA();
  virtual ~TrackBeamCUDA();
  void track(double, Beam *, Undulator *, bool);
  void applyR56(Beam *, Undulator *, double);
  void applyCorrector(Beam *, double, double);
  void applyChicane(Beam *, double, double, double, double, double);
};

#endif
