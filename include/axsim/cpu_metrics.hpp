/**
 * CPU AIG Monte Carlo simulation and the same ER / MAE% / MSE metrics as run_gpu_metrics.
 *
 * ResubALS-style / project convention: bit-parallel levelized AND evaluation over packed
 * uint64 planes (64 patterns per SIMD word), PI randomness from the same LCG as the CUDA
 * monte_carlo_kernel (seed + block_idx * 97u per block), plus optional external PI planes
 * via GpuMetricsConfig::external_pi_packed. Single-threaded; semantics match GPU for correctness.
 */

#ifndef AXSIM_CPU_METRICS_HPP
#define AXSIM_CPU_METRICS_HPP

#include "circuit_soa.hpp"
#include "gpu_metrics.hpp"
#include <cstdint>
#include <vector>

namespace axsim {

std::vector<std::uint64_t> run_cpu_simulation_only(
    const CircuitSoA& soa,
    const GpuMetricsConfig& config = {});

GpuMetricsResult run_cpu_metrics(
    const CircuitSoA& soa,
    const std::vector<std::uint64_t>& ref_outputs,
    const GpuMetricsConfig& config = {});

GpuMetricsResult run_cpu_metrics_pair(
    const CircuitSoA& approx_soa,
    const CircuitSoA& golden_soa,
    const GpuMetricsConfig& config = {});

} // namespace axsim

#endif
