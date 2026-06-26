## Genesis4 CUDA port

CUDA acceleration port for [GENESIS 1.3 Version 4](https://github.com/svenreiche/Genesis-1.3-Version4).

The CUDA backend uses lightweight helper headers under `include/`:

- `Genesis4CudaBuffer.h`: RAII device/pinned buffers.
- `Genesis4CudaLaunch.h`: `g4_parallel_for(...)` wrapper for CUDA extended device lambdas.
- `Genesis4CudaRuntime.h`: CUDA runtime checks, default stream access, synchronization helpers.

### Build CUDA version

```bash
cmake -S . -B build \
  -DGENESIS_USE_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
```

### Build pure CPU upstream fallback

```bash
cmake -S . -B build_cpu \
  -DGENESIS_USE_CUDA=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build_cpu -j 4
```


### `&track/use_cuda`
`use_cuda` controls the CUDA solver in this port, default is true:
- `use_cuda = false`：FieldSolver, BeamSolver, TrackBeam, Control::applySlippage, Diagnostic::calc use Genesis4 upstream CPU implementation.
- `use_cuda = true`：FieldSolver uses `FieldSolverADICUDA`/`FieldSolverFFTCUDA`, BeamSolver uses `BeamSolverCUDA`, beam transverse/R56 path uses `TrackBeamCUDA`, slippage uses `ControlCUDA`, built-in diagnostic calculations use `DiagnosticCUDA`.


### `&track/low_memory_adisolver`

`low_memory_adisolver` controls the source-deposition path used by the CUDA ADI field solver, default is `false`:

- `low_memory_adisolver = false`: `FieldSolverADICUDA` uses the source-grid path. Particle source terms are deposited once into an auxiliary source grid `d_source_re`/`d_source_im`, then reused by the two ADI half-steps.

- `low_memory_adisolver = true`: `FieldSolverADICUDA` uses the direct atomic source path. No auxiliary source grid is allocated. Each ADI half-step scans particles and deposits the source contribution. This reduces GPU memory usage, but may be a little bit slower because particles are scanned and deposition twice.

This option only affects the CUDA ADI field solver when `use_cuda = true` and `fft_fieldsolver = false`. 

### Known issues
1. one4one `sort` for `Marker` is not implemented.
