#ifndef __GENESIS_GENCORE__
#define __GENESIS_GENCORE__

#include <iostream>
#include <string>
#include <vector>
#include <complex>

#include "Beam.h"
#include "Field.h"
#include "Undulator.h"
#include "Control.h"
#include "Diagnostic.h"
#include "Setup.h"

using namespace std;

class Gencore{
 public:
   Gencore(){};
   virtual ~Gencore(){};
   bool run(Beam *, vector<Field*> *, Setup *, Undulator *, bool, bool, bool, bool, FilterDiagnostics &filter);
};

#endif
