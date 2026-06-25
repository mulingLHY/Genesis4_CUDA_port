#ifndef __GENESIS_CONTROLCUDA__
#define __GENESIS_CONTROLCUDA__

#include <vector>

#ifdef GENESIS_USE_CUDA

class Field;
class Control;
class Beam;
class Undulator;

using namespace std;

class ControlCUDA {
 public:
   ControlCUDA();
   ~ControlCUDA();

   void init(int, int, bool, bool, double);
   bool applySlippage(double, Field *);
   bool applyMarker(Control *,Beam *, vector<Field *> *, Undulator *, bool&);

 private:
   void ensureWorkBuffers(long long);
   void releaseWorkBuffers();

   bool timerun_;
   bool periodic_;
   int rank_;
   int mpiSize_;
   double sample_;

   long long deviceWorkCapacity_;
   double *deviceWork_;
   long long pinnedWorkCapacity_;
   double *pinnedWork_;
};

#endif

#endif
