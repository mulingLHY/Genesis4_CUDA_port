#include "Gencore.h"

#include <memory>

#ifdef USE_DPI
  #include "DiagnosticHookS.h"
#endif
#ifdef GENESIS_USE_CUDA
#include "Genesis4CudaRuntime.h"
#include <mpi.h>
#include <cuda_runtime.h>
#include "ControlCUDA.h"
#include "DiagnosticCUDA.h"
static bool cuda_initialized = false;
#endif
extern bool MPISingle;


bool Gencore::run(Beam *beam, vector<Field*> *field, Setup *setup, Undulator *und,bool isTime, bool isScan, bool periodic, bool use_cuda, FilterDiagnostics &filter)
{
    // function returns 'true' if everything is ok
#ifdef GENESIS_USE_CUDA
    if (use_cuda && !cuda_initialized) {
        genesis4_cuda::initialize();
        cuda_initialized = true;
    }

    if (use_cuda) {
        beam->pack_beam_to_soa();
        for (int i = 0; i < field->size(); ++i) {
            field->at(i)->pack_field_to_soa();
        }
    }
#endif
{// To ensure local variables are deleted before CUDA finalization is called
    //-------------------------------------------------------
    // init MPI and get size etc.
    //
    int size=1;
    int rank=0;
    if (!MPISingle){
        MPI_Comm_rank(MPI_COMM_WORLD, &rank); // assign rank to node
        MPI_Comm_size(MPI_COMM_WORLD, &size); // assign rank to node
    }
#ifdef GENESIS_USE_CUDA
    if (use_cuda) {
        int dev = genesis4_cuda::device_id();
        cudaDeviceProp prop;
        g4_cuda_check(cudaGetDeviceProperties(&prop, dev), "cudaGetDeviceProperties");
        std::cout << "MPI rank " << rank
                  << " uses CUDA device " << dev
                  << " name=" << prop.name
                  << " pciBusID=" << prop.pciBusID
                  << std::endl;
    }
#endif
    if (rank==0) {
        cout << endl << "Running Core Simulation..." << endl;
    }

    //-----------------------------------------
    // init beam, field and undulator class

    string rn, fnbase;
    setup->getRootName(&rn);
    setup->RootName_to_FileName(&fnbase, &rn); // includes .RunX. if not the first &track command
    Control   *control=new Control;
    control->init(rank,size,fnbase,beam,field,und,isTime,isScan, periodic);
#ifdef GENESIS_USE_CUDA
    ControlCUDA control_cuda;
    if (use_cuda) {
        control_cuda.init(rank, size, isTime, periodic, beam->slicelength / beam->reflength);
    }
#endif
    auto apply_slippage = [&](double shift, Field *fld) {
#ifdef GENESIS_USE_CUDA
        if (use_cuda && control_cuda.applySlippage(shift, fld)) { return; }
#endif
        control->applySlippage(shift, fld);
    };

    Diagnostic diag;
#ifdef GENESIS_USE_CUDA
    std::unique_ptr<DiagnosticCUDA> diag_cuda;
    if (use_cuda) {
        diag_cuda.reset(new DiagnosticCUDA());
        diag_cuda->init(filter);
    }
#endif
    auto calc_diagnostic = [&](double z) {
#ifdef GENESIS_USE_CUDA
        if (use_cuda && diag_cuda && diag_cuda->calc(beam, field, z, diag)) { return; }
#endif
        diag.calc(beam, field, z);
    };
#ifdef USE_DPI
    und->plugin_info_txt.clear();

    for(int kk=0; kk<setup->diagpluginfield_.size(); kk++) {
        if(rank==0) {
            cout << "Setting up DiagFieldHook for libfile=\"" << setup->diagpluginfield_.at(kk).libfile << "\", obj_prefix=\"" << setup->diagpluginfield_.at(kk).obj_prefix << "\"" << endl;
        }
        DiagFieldHook *pdfh = new DiagFieldHook(); /* !do not delete this instance, it will be destroyed when DiagFieldHook instance is deleted! */
        bool diaghook_ok = pdfh->init(&setup->diagpluginfield_.at(kk));
        if(diaghook_ok) {
            pdfh->set_runid(setup->getCount()); // propagate run id so that it can be used in the plugins, for instance for filename generation

            string tmp_infotxt = pdfh->get_info_txt();
            und->plugin_info_txt.push_back(tmp_infotxt);
            stringstream tmp_prefix;
            tmp_prefix << "/Field/" << setup->diagpluginfield_.at(kk).obj_prefix;
            und->plugin_hdf5_prefix.push_back(tmp_prefix.str());

            diag.add_field_diag(pdfh);
            if(rank==0) {
                cout << "DONE: Registered DiagFieldHook" << endl;
            }
        } else {
            delete pdfh;
            if(rank==0) {
                cout << "failed to set up DiagFieldHook, not registering" << endl;
            }
        }
    }

    for(int kk=0; kk<setup->diagpluginbeam_.size(); kk++) {
        if(rank==0) {
            cout << "Setting up DiagBeamHook for libfile=\"" << setup->diagpluginbeam_.at(kk).libfile << "\", obj_prefix=\"" << setup->diagpluginbeam_.at(kk).obj_prefix << "\"" << endl;
        }
        DiagBeamHook *pdbh = new DiagBeamHook(); /* !do not delete this instance, it will be destroyed when DiagFieldHook instance is deleted! */
        bool diaghook_ok = pdbh->init(&setup->diagpluginbeam_.at(kk));
        if(diaghook_ok) {
            pdbh->set_runid(setup->getCount()); // propagate run id so that it can be used in the plugins, for instance for filename generation

            string tmp_infotxt = pdbh->get_info_txt();
            und->plugin_info_txt.push_back(tmp_infotxt);
            stringstream tmp_prefix;
            tmp_prefix << "/Beam/" << setup->diagpluginbeam_.at(kk).obj_prefix;
            und->plugin_hdf5_prefix.push_back(tmp_prefix.str());

            diag.add_beam_diag(pdbh);
            if(rank==0) {
                cout << "DONE: Registered DiagBeamHook" << endl;
            }
        } else {
            delete pdbh;
            if(rank==0) {
                cout << "failed to set up DiagBeamHook, not registering" << endl;
            }
        }
    }
#endif
    if (rank==0) { cout << "Initial analysis of electron beam and radiation field..."  << endl; }
    diag.init(rank, size, und->outlength(), beam->beam.size(),field->size(),isTime,isScan,filter);
    calc_diagnostic(und->getz());  // initial calculation

double t0 = 0.0;
double t_beam_track = 0.0;
double t_field_track = 0.0;
double t_apply_slip = 0.0;
double t_unpack = 0.0;
double t_diag_calc = 0.0;
    /*************/
    /* MAIN LOOP */
    /*************/
    while(und->advance(rank))
    {
      double delz=und->steplength();

      // ----------------------------------------
      // step 1 - apply most marker action  (always at beginning of a step)
      bool error_IO=false;
      bool sort=control->applyMarker(beam, field, und, error_IO);
      if(error_IO) {
        return(false);
      }

      // ---------------------------------------
      // step 2 - Advance electron beam
#ifdef GENESIS_USE_CUDA
if (use_cuda) { g4_cuda_synchronize(); }
#endif
t0 = MPI_Wtime();
      beam->track(delz,field,und);
#ifdef GENESIS_USE_CUDA
if (use_cuda) { g4_cuda_synchronize(); }
#endif
t_beam_track += MPI_Wtime() - t0;
      // -----------------------------------------
      // step 3 - Beam post processing, e.g. sorting


      if (sort){
        int shift=beam->sort();

        if (shift!=0){
          for (int i=0;i<field->size();i++){
              apply_slippage(shift, field->at(i));
          }
        }
      }

      // ---------------------------------------
      // step 4 - Advance radiation field
#ifdef GENESIS_USE_CUDA
if (use_cuda) { g4_cuda_synchronize(); }
#endif
t0 = MPI_Wtime();
      for (int i=0; i<field->size();i++){
        field->at(i)->track(delz,beam,und);
      }
#ifdef GENESIS_USE_CUDA
if (use_cuda) { g4_cuda_synchronize(); }
#endif
t_field_track += MPI_Wtime() - t0;
      //-----------------------------------------
      // step 5 - Apply slippage
#ifdef GENESIS_USE_CUDA
if (use_cuda) { g4_cuda_synchronize(); }
#endif
t0 = MPI_Wtime();
      for (int i=0;i<field->size();i++){
        apply_slippage(und->slippage(), field->at(i));  
      }
#ifdef GENESIS_USE_CUDA
if (use_cuda) { g4_cuda_synchronize(); }
#endif
t_apply_slip += MPI_Wtime() - t0;
      //-------------------------------
      // step 6 - Calculate beam parameter stored into a buffer for output

      //beam->diagnostics(und->outstep(),und->getz());
      //for (int i=0;i<field->size();i++){
      //  field->at(i)->diagnostics(und->outstep());
      //}

#ifdef GENESIS_USE_CUDA
if (use_cuda) { g4_cuda_synchronize(); }
#endif
t0 = MPI_Wtime();
      if (und->outstep()) {
        calc_diagnostic(und->getz());
      }
#ifdef GENESIS_USE_CUDA
if (use_cuda) { g4_cuda_synchronize(); }
#endif
t_diag_calc += MPI_Wtime() - t0;
    }
t0 = MPI_Wtime();
#ifdef GENESIS_USE_CUDA
    if (use_cuda) {
        beam->unpack_soa_to_beam();
        for (int i = 0; i < field->size(); ++i) {
            field->at(i)->unpack_soa_to_field();
        }
    }
#endif 
t_unpack += MPI_Wtime() - t0;
if (rank==0){ 
cout << "Time spent in beam tracking: " << t_beam_track << endl;
cout << "Time spent in field tracking: " << t_field_track << endl;
cout << "Time spent in slippage: " << t_apply_slip << endl;
cout << "Time spent in diagnostic calculation: " << t_diag_calc << endl;
cout << "Time spent in unpacking: " << t_unpack << endl;}
        //---------------------------
        // end and clean-up 

    // perform last marker action
        bool error_IO=false;
    bool sort=control->applyMarker(beam, field, und, error_IO);
    if(error_IO) {
      return(false);
    }
    if (sort){
        int shift=beam->sort();

        if (shift!=0){
          for (int i=0;i<field->size();i++){
            apply_slippage(shift, field->at(i));
          }
        }
    }


    /* write out diagnostic arrays */
    if (rank==0){
      cout << "Writing output file..." << endl;
    }

    // control->output(beam,field,und,diag);
    if(!diag.writeToOutputFile(beam, field, setup, und)) {
      delete control;
      return(false);
    }

    delete control;
      
    if (rank==0){
      cout << endl << "Core Simulation done." << endl;
    }
}
#ifdef GENESIS_USE_CUDA
    if (use_cuda && cuda_initialized) {
        genesis4_cuda::finalize();
        cuda_initialized = false;
    }
#endif  
    return(true);
}
