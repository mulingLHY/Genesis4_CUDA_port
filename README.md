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


### Known issues
1. one4one `sort` for `Marker` is not implemented.
