# AxSim-GPU

GPU-accelerated Monte Carlo evaluation for approximate circuits.

AxSim-GPU reads a pair of circuits (golden and approximate), runs aligned random-pattern simulation on GPU, and reports error metrics:

- `error_rate`
- `mae_norm` (normalized MAE)
- `mse`

## Highlights

- CUDA bit-parallel simulation (`64` patterns per `uint64_t` lane)
- Pair-evaluation API with PI/PO compatibility checks
- Supports shared input-pattern file (`AXPI010`) via `--patterns-file`
- Works with AIG/BLIF/Verilog netlists through ABC interface

## Requirements

- C++17
- CMake >= 3.18
- CUDA Toolkit (`nvcc`)
- NVIDIA GPU (recommended compute capability >= 7.0)

Optional:

- Python 3 for helper scripts
- Yosys/Icarus/BLASYS for benchmark workflows

## Quick Start

```bash
git clone --recurse-submodules https://github.com/shitong-G/AxSim-GPU.git
cd AxSim-GPU
make
./build/axsim_main BACS/abs_diff/abs_diff.aig BACS/abs_diff/abs_diff_approx.aig
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

## Build

### Make (recommended)

```bash
make
```

Common options:

- Debug build: `make DEBUG=1`
- Specify CUDA path: `make CUDA_HOME=/path/to/cuda`
- Specify architecture: `make CUDA_ARCH=sm_80`

### CMake

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Run

```bash
./build/axsim_main <golden_netlist> <approx_netlist> [num_patterns] [seed] \
  [--patterns N] [--seed S] [--mae-normalizer X] [--lsb-first] \
  [--patterns-file <file.axpi>] [--backend gpu|cpu|both] [--print-timing]
```

Notes:

- Use `--lsb-first` for designs where `O[0]` is LSB (common in EvoApprox circuits).
- `MAE% = 100 * mae_norm`; set `--mae-normalizer` per design bit-width.
- `--patterns-file` expects AXPI010 format compatible with `scripts/pattern_io.py`.

## Pattern File Utility (AXPI010)

Generate shared PI patterns:

```bash
python scripts/gen_pi_patterns.py benchmarks/patterns/pi_n16_p4M_s42.axpi \
  --bits 16 --patterns 4000000 --seed 42
```

This is useful when you want CPU and GPU evaluations to use exactly the same input vectors.

## Repository Layout

- `src/`, `include/`, `cuda/`: core implementation
- `scripts/`: project-level helper scripts (including AXPI tools)
- `benchmarks/`: optional experiment/evaluation workflows
- `BACS/`: benchmark circuits (git submodule)

## Benchmark Workflows (Optional)

Benchmark scripts are kept under `benchmarks/` and are not required for basic build/run.

```bash
python benchmarks/runtime_benchmark.py --config benchmarks/runtime_config.fair.json
```

See `benchmarks/README.md` for details.

## License

MIT License. See `LICENSE`.

