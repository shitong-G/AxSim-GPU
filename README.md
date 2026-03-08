# AxSim-GPU

**GPU-accelerated Monte Carlo simulation and error metrics for approximate logic synthesis.**

AxSim-GPU is an open-source library for fast evaluation of approximate circuits. It runs massive Monte Carlo simulation on CUDA and computes standard error metrics—Error Rate (ER), Mean Relative Error Distance (MRED), and Mean Squared Error (MSE)—using bit-parallel simulation and parallel reduction. The design is compatible with AIG (And-Inverter Graph) representations and can be wired to tools such as [ABC](https://people.eecs.berkeley.edu/~alanmi/abc/).

---

## Features

- **Structure-of-Arrays (SoA) circuit representation**  
  Flattens an AIG into GPU-friendly arrays (`fanin0_indices`, `fanin1_indices`, `is_compl0`, `is_compl1`) on the CPU; a single `cudaMemcpy` sends the circuit to the device.

- **Bit-parallel Monte Carlo simulation**  
  Each CUDA thread simulates 64 random patterns in one `uint64_t`. Random inputs are generated on-device (LCG). Optional use of shared memory for primary inputs to reduce global memory traffic.

- **Efficient metric computation**  
  - **Error Rate**: XOR of approximate vs reference outputs, `__popcll()`, then block reduction and host sum; result normalized by total bits.
  - **MRED**: Per-pattern relative error with a small δ to avoid division by zero; parallel sum and mean on the host.
  - **MSE**: For 0/1 outputs, equivalent to error count; extensible to integer outputs.

- **Python reference implementation**  
  `src/metrics.py` provides CPU, truth-table–based implementations of ER, MRED, and MSE (as in the ACT-style formulation) for validation and small benchmarks.

---

## Implemented Metrics

| Metric | Formula | Notes |
|--------|--------|--------|
| **Error Rate (ER)** | \( \frac{1}{|\mathcal{X}|} \sum_{x} \mathbb{I}[g(x) \neq f(x)] \) | Fraction of differing output bits |
| **MRED** | \( \frac{1}{|\mathcal{X}|} \sum_{x} \frac{|g(x)-f(x)|}{\max(|f(x)|,\delta)} \) | Mean relative error; δ default 1e-6 |
| **MSE** | \( \frac{1}{|\mathcal{X}|} \sum_{x} (g(x)-f(x))^2 \) | For Boolean outputs, matches ER |

*Maximum Error Magnitude (MEM)* is planned for a future release.

---

## Requirements

- **Build**: C++17, CMake ≥ 3.18, CUDA Toolkit (with `nvcc`).
- **Runtime**: NVIDIA GPU with compute capability ≥ 7.0 (e.g. Volta, Turing, Ampere).
- **Python** (optional): Python 3 with `numpy` and `bitarray` for `src/metrics.py`.

---

## Build and run

**Using Make (recommended):**

```bash
git clone https://github.com/YOUR_USERNAME/AxSim-GPU.git
cd AxSim-GPU
make
./build/axsim_main
```

Optional: debug build `make DEBUG=1`; target a different GPU `make CUDA_ARCH=sm_80`; run directly `make run`.

**Using CMake:**

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
./axsim_main
```

To target specific GPU architectures with CMake: `-DCMAKE_CUDA_ARCHITECTURES="70;80"`. With Make: `make CUDA_ARCH=sm_80`.

---

## Project layout

```
AxSim-GPU/
├── include/axsim/
│   ├── circuit_soa.hpp    # SoA struct and flatten_from_aig()
│   ├── gpu_metrics.hpp    # run_gpu_metrics(), run_gpu_simulation_only(), config/result types
│   └── abc_interface.hpp  # soa_from_abc_file() — read AIG/BLIF/Verilog via ABC
├── src/
│   ├── circuit_soa.cpp    # CPU SoA construction from AIG list
│   ├── interface.cpp      # ABC → SoA (optional; build with AXSIM_USE_ABC and link libabc)
│   ├── main_axsim.cpp     # Example: small AIG → SoA → GPU metrics
│   └── metrics.py         # Python reference: ER, MRED, MSE (truth-table)
├── cuda/
│   └── sim_kernels.cu     # Monte Carlo kernels, error/MRED/MSE reductions, host API impl
├── docs/
│   └── GPU_SKELETON.md    # Design notes and extension points (e.g. ABC, CUB, shared memory)
├── Makefile               # Build with make (default)
├── CMakeLists.txt         # Alternative: build with CMake
├── LICENSE
└── README.md
```

---

## API overview

### C++ (GPU path)

1. **Build a circuit in SoA form** — from an ABC netlist file (AIG/BLIF/Verilog, etc.):

   ```cpp
   #include "axsim/abc_interface.hpp"
   using namespace axsim;

   CircuitSoA soa = soa_from_abc_file("design.aig");  // or .blif, .v, ...
   if (!soa.valid()) { /* handle read/strash failure */ }
   ```

   Or build SoA manually (e.g. from your own AIG front-end):

   ```cpp
   #include "axsim/circuit_soa.hpp"
   int num_pis = 2;
   std::vector<std::tuple<int,int,bool,bool>> and_nodes = { {0, 1, false, false} };
   std::vector<int> out_ids = {2};
   CircuitSoA soa = flatten_from_aig(num_pis, and_nodes, out_ids);
   ```

2. **Run Monte Carlo and get ER, MRED, MSE**:

   ```cpp
   #include "axsim/gpu_metrics.hpp"

   GpuMetricsConfig config;
   config.num_patterns = 65536;
   config.seed = 12345;

   std::vector<uint64_t> ref_outputs = ...;  // reference bits: num_outputs * ceil(num_patterns/64)
   GpuMetricsResult res = run_gpu_metrics(soa, ref_outputs, config);
   // res.error_rate, res.mred, res.mse
   ```

3. **Simulation only** (e.g. to generate approximate outputs):

   ```cpp
   std::vector<uint64_t> approx = run_gpu_simulation_only(soa, config);
   ```

### Python (reference metrics)

```python
from src.metrics import ErrorMetrics, compute_error_rate, compute_mred, compute_mse

m = ErrorMetrics(num_inputs=4, delta=1e-6)
er = m.compute_error_rate(g_outputs, f_outputs)
mred = m.compute_mred(g_values, f_values)
mse = m.compute_mse(g_values, f_values)
```

---

## Data layout (GPU)

- **Circuit SoA (device)**  
  `fanin0`, `fanin1`, `is_compl0`, `is_compl1`: one entry per AND node; node IDs 0…`num_pis-1` are primary inputs, then ANDs in topological order.

- **Simulation buffer**  
  `node_values`: shape `[num_blocks_64][num_nodes]` with `num_blocks_64 = num_patterns/64`. Each row is one 64-pattern block.

- **Outputs**  
  `approx` / `reference`: `[num_outputs][num_blocks_64]` uint64_t, packed in the same order as the Python/truth-table bit layout.

---

## Extending the library

- **ABC integration**: In `circuit_soa.cpp` (or a new module), traverse the AIG from your ABC handle, collect `(fanin0, fanin1, compl0, compl1)` in topological order, and call `flatten_from_aig()` or fill `CircuitSoA` directly.
- **Reference outputs**: Obtain `ref_outputs` from a golden netlist (e.g. CPU simulation or another GPU run) in the packed format above.
- **Performance**: See `docs/GPU_SKELETON.md` for shared-memory caching of PIs, CUB-based reduction, and a single output-gather kernel.

**Building with ABC (optional):** To enable `soa_from_abc_file()` for AIG/BLIF/Verilog, build with ABC: define `AXSIM_USE_ABC`, add ABC include path (e.g. `-I path/to/abc/src`), and link libabc. Example with Make: `make CXXFLAGS="-DAXSIM_USE_ABC -I/path/to/abc/src" LDFLAGS="-L/path/to/abc -labc"`. Without ABC, the interface compiles as a stub that returns an invalid SoA.

A **checklist of remaining work** (ref output pipeline, large-pattern grid limit, ABC/AIG wiring, optional optimizations, MEM, tests) is in [docs/TODO.md](docs/TODO.md).

---

## Citation

If you use AxSim-GPU in academic work, please cite:

```bibtex
@software{axsim_gpu,
  author = {Guo, Shitong},
  title  = {AxSim-GPU: GPU-Accelerated Monte Carlo Simulation and Error Metrics for Approximate Logic},
  year   = {2026},
  url    = {https://github.com/YOUR_USERNAME/AxSim-GPU}
}
```

---

## License

This project is licensed under the MIT License—see [LICENSE](LICENSE).

---

## References

- Error metrics (ER, MRED, MSE) follow the formulation used in approximate circuit synthesis and validation (e.g. ACT-style metrics).
- AIG format is compatible with [ABC](https://people.eecs.berkeley.edu/~alanmi/abc/).
