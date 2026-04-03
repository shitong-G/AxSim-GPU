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

## Current status

- **Safe pair-evaluation API**: `run_gpu_metrics_pair(approx_soa, golden_soa, config)` validates PI/PO compatibility, runs golden simulation internally, and evaluates the approximate circuit under the same random vectors.
- **Large-pattern correctness fix**: Monte Carlo launch now uses tiled launches, so pattern coverage is complete beyond the 65535-grid limit.
- **Robust runtime checks**: CUDA API calls and kernel launches are checked; `GpuMetricsResult.ok` reports whether computation succeeded.
- **Metric semantics clarified**: `mae_norm` is the normalized MAE (and `mred` is kept as backward-compatible alias). Output bit order is configurable via `outputs_msb_first`.
- **Manual verification tests**: `src/manual_metrics_test.cpp` includes hand-computable cases (PASS/FAIL printout) to validate ER/MAE/MSE and bit-order behavior.
- **Build (Make)**: `CUDA_HOME` defaults to `/usr/local/cuda` (override if `cuda_runtime.h` is not found). Device code uses **`-std=c++14`** for older `nvcc`; host code remains **C++17**. Link/search paths: `-I$(CUDA_HOME)/include`, `-L$(CUDA_HOME)/lib64`.
- **Still open**: output gather kernel, CUB reduction, optional RNG/curand path, and broader ABC parser compatibility across all third-party Verilog styles—see [docs/TODO.md](docs/TODO.md).
- **Benchmark harness**: Shared **AXPI010** (`.axpi`) primary-input files align random vectors across **CPU iverilog** (`verilog_eval_runtime_native.py`), **BLASYS** (`verilog_eval_runtime.py`), and **`axsim_main --patterns-file`**. The native Verilog evaluator streams a hex stimulus file and uses `$readmemh` in the testbench so million-vector runs do not generate multi-gigabyte generated Verilog.

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
- **Runtime benchmarks** (optional): Yosys + Icarus Verilog (`iverilog` / `vvp`) for `verilog_eval_runtime_native.py`; a local BLASYS tree for `verilog_eval_runtime.py` (point `blasys_root` in the JSON at your checkout; use the same Python environment as BLASYS, e.g. conda with `regex`).

---

## Build and run

**Using Make (recommended):**

```bash
git clone https://github.com/YOUR_USERNAME/AxSim-GPU.git
cd AxSim-GPU
make
./build/axsim_main BACS/abs_diff/abs_diff.aig BACS/abs_diff/abs_diff_approx.aig
```

Optional: debug build `make DEBUG=1`; target a different GPU `make CUDA_ARCH=sm_80`; CUDA toolkit path `make CUDA_HOME=/path/to/cuda`; older CUDA 11+ nvcc may use `make NVCC_STD=c++17`; run directly `make run`.

`axsim_main` command-line arguments:

```bash
./build/axsim_main <golden_netlist> <approx_netlist> [num_patterns] [seed] \
  [--patterns N] [--seed S] [--mae-normalizer X] [--lsb-first] [--patterns-file <file.axpi>]
```

- Input format is auto-detected by extension through ABC (e.g. `.aig`, `.v`, `.blif`).
- For unsigned arithmetic where `O[0]` is LSB (e.g. many EvoApprox circuits), use `--lsb-first`.
- `MAE% = 100 * mae_norm`; pick `--mae-normalizer` per benchmark convention (e.g. 255 for 8-bit `abs_diff`, 511 for 9-bit adders).
- Optional **`--patterns-file`**: load shared PI planes from an AXPI010 binary (same RNG layout as `benchmarks/tools/pattern_io.py`; use with the same `num_patterns` as stored in the file).

### Runtime benchmarking (same-task comparison)

A runtime harness is provided at `benchmarks/runtime_benchmark.py` to compare **error-evaluation runtime for the same task**:

- baseline: CPU Verilog simulation + metric computation (`benchmarks/tools/verilog_eval_runtime_native.py`)
- SOTA: BLASYS-compatible evaluation flow (`benchmarks/tools/verilog_eval_runtime.py`, optional)
- SOTA: Mockturtle Monte Carlo evaluator (AIG input)
- this project: GPU Monte Carlo (`axsim_main`)

The harness is command-template based, so each method can call your local toolchain directly.

#### Shared primary-input patterns (AXPI010)

To compare **the same** random input vectors on CPU Verilog, BLASYS, and GPU, generate a binary **AXPI010** file (`.axpi`) and pass it via `--patterns-file` in each tool. The format and RNG packing match `benchmarks/tools/pattern_io.py` (LSB-indexed PI bits per pattern). **The PI bit width must match** the flattened design’s total primary-input width (`--bits` in the generator = number of PIs after ABC flattening).

```bash
python benchmarks/tools/gen_pi_patterns.py benchmarks/patterns/pi_n16_p4M_s42.axpi \
  --bits 16 --patterns 4000000 --seed 42
```

Large `.axpi` files are listed in `.gitignore` under `benchmarks/patterns/`. The native Verilog path writes a compact `stim.hex` and uses `$readmemh` in the testbench instead of emitting millions of `#1 pi=...` lines (which previously exhausted RAM and disk).

#### Fair preset (`runtime_config.fair.json`)

`benchmarks/runtime_config.fair.json` is an example **fair** preset: CPU + BLASYS + GPU with a shared `pi_patterns_file`, optional `cpu_affinity` (Linux `taskset`), and Mockturtle disabled by default. Edit **`workspace`**, **`axsim_root`**, **`blasys_root`**, **`pi_patterns_file`**, and benchmark paths to match your machine.

```bash
python benchmarks/runtime_benchmark.py --config benchmarks/runtime_config.fair.json
```

Use **`globals.vectors`** / **`patterns`** consistent with the `.axpi` header. If BLASYS needs a specific interpreter (e.g. conda env with `regex`), set **`globals.bench_python`** in the JSON, pass **`--bench-python`**, or export **`AXSIM_BENCH_PYTHON`**.

#### Generic workflow

1. Copy and edit the config:

```bash
cp benchmarks/runtime_config.example.json benchmarks/runtime_config.json
```

2. Fill in local paths/commands (`BLASYS`, `mockturtle`, benchmark files, etc.), then run:

```bash
python3 benchmarks/runtime_benchmark.py --config benchmarks/runtime_config.json
```

For Mockturtle integration, build helper binary once:

```bash
g++ -O3 -std=c++17 benchmarks/tools/mockturtle_mc_eval.cpp \
  /mnt/d/____Research____/mockturtle/lib/fmt/fmt/format.cc \
  /mnt/d/____Research____/mockturtle/lib/fmt/fmt/os.cc \
  -I /mnt/d/____Research____/mockturtle/include \
  -I /mnt/d/____Research____/mockturtle/lib/kitty \
  -I /mnt/d/____Research____/mockturtle/lib/lorina \
  -I /mnt/d/____Research____/mockturtle/lib/parallel_hashmap \
  -I /mnt/d/____Research____/mockturtle/lib/fmt \
  -o /mnt/d/____Research____/mockturtle/build/mockturtle_mc_eval
```

3. Results are saved under `benchmarks/results/<timestamp>/`:

- `runtime_results.csv`
- `runtime_results.json`
- `runtime_summary.md`
- per-run logs in `logs/`

**Using CMake:**

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
./axsim_main
```

To target specific GPU architectures with CMake: `-DCMAKE_CUDA_ARCHITECTURES="70;80"`. With Make: `make CUDA_ARCH=sm_80`.

#### EvoApproxLib cross-check (GPU vs README table)

On **Windows**, build and run GPU tools under **WSL** so the CUDA runtime and GPU match your Linux toolchain. If `./build/axsim_main` fails with `libcudart.so` not found, set:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

The script **`benchmarks/tools/evoapprox_gpu_check.py`** runs several **exhaustive** sweeps (counting PI patterns via `write_axpi010_counting_pi` in `pattern_io.py`) and compares **EP%**, **MAE%**, and **MSE** to the EvoApprox **`pareto_pwr_mae`** README tables:

| Default cases | Family | Golden → approx |
|---------------|--------|-----------------|
| `add8u_006`, `add8u_8LL` | `adders/8_unsigned` | `add8u_0FP` → approximate |
| `mul7u_03M`, `mul7u_069` | `multiplers/7x7_unsigned` | `mul7u_01L` → approximate |

Use **`--cases all`** to also run **`add12u_20A`** (2²⁴ patterns, long run, ~48 MB `.axpi`). **`--cases add8u_006`** runs a single id.

**ABC caveat:** some EvoApprox multipliers (e.g. **`mul8u_1JFF`**) use very wide internal buses that the bundled ABC Verilog reader rejects; those golden netlists are omitted until a compatible golden exists (e.g. Yosys-synthesized BLIF).

On Windows the script invokes **`wsl`** automatically; in WSL/Linux:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
bash benchmarks/run_evoapprox_check_wsl.sh
```

Optional: `--evoapprox-root /path/to/evoapproxlib` if the library is not a sibling of AxSim-GPU.

#### Publication-style experiments (scalability & throughput)

For **scalability** (varying `num_patterns`) and **patterns/sec**, rebuild `axsim_main` and use **`--print-timing`** (prints `EVAL_GPU_S` and `THROUGHPUT_PATTERNS_PER_S` for the GPU pair-evaluation phase). Helper: **`benchmarks/tools/gpu_scalability_sweep.py`**; methodology and reviewer checklist: **[docs/PAPER_EXPERIMENTS.md](docs/PAPER_EXPERIMENTS.md)**; circuit examples: **`benchmarks/scalability_presets.json`**.

---

## Project layout

```text
AxSim-GPU/
├── benchmarks/
│   ├── runtime_benchmark.py      # Multi-method runtime harness
│   ├── runtime_config.example.json
│   ├── runtime_config.fair.json  # Example: shared .axpi + CPU/BLASYS/GPU
│   ├── scalability_presets.json  # Example circuits for gpu_scalability_sweep.py
│   ├── tools/
│   │   ├── pattern_io.py         # AXPI010 read/write + PI packing
│   │   ├── gen_pi_patterns.py    # CLI: write .axpi for fair benchmarks
│   │   ├── gpu_scalability_sweep.py # Paper: sweep N vs EVAL_GPU_S / throughput
│   │   ├── evoapprox_gpu_check.py # EvoApprox adders/multipliers vs README (WSL on Windows)
│   │   ├── verilog_eval_runtime_native.py  # iverilog baseline
│   │   └── verilog_eval_runtime.py         # BLASYS-backed eval
│   ├── run_evoapprox_check_wsl.sh
│   └── results/                  # Created when you run the harness
├── include/axsim/
│   ├── circuit_soa.hpp    # SoA struct and flatten_from_aig()
│   ├── gpu_metrics.hpp    # run_gpu_metrics(), run_gpu_simulation_only(), config/result types
│   └── abc_interface.hpp  # soa_from_abc_file() — read AIG/BLIF/Verilog via ABC
├── src/
│   ├── circuit_soa.cpp    # CPU SoA construction from AIG list
│   ├── interface.cpp      # ABC → SoA (optional; build with AXSIM_USE_ABC and link libabc)
│   ├── main_axsim.cpp     # CLI runner: golden vs approximate netlists
│   ├── manual_metrics_test.cpp # Hand-checkable metric sanity tests
│   ├── evo_compare.cpp    # EvoApprox-style pair comparison helper
│   └── metrics.py         # Python reference: ER, MRED, MSE (truth-table)
├── cuda/
│   └── sim_kernels.cu     # Monte Carlo kernels, error/MRED/MSE reductions, host API impl
├── docs/
│   ├── GPU_SKELETON.md    # Design notes and extension points (e.g. ABC, CUB, shared memory)
│   ├── TECHNICAL_REPORT.md # Architecture & metrics (formal write-up)
│   ├── PAPER_EXPERIMENTS.md # Scalability / throughput / crossover checklist for publication
│   └── TODO.md            # Progress snapshot and remaining work
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

2. **Run Monte Carlo and get ER/MAE/MSE**:

   ```cpp
   #include "axsim/gpu_metrics.hpp"

   GpuMetricsConfig config;
   config.num_patterns = 65536;
   config.seed = 12345;

   GpuMetricsResult res = run_gpu_metrics_pair(approx_soa, golden_soa, config);
   if (!res.ok) { /* handle runtime/validation failure */ }
   // res.error_rate, res.mae_norm, res.mse
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
