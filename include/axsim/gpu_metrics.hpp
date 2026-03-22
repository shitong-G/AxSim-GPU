/**
 * GPU Monte Carlo simulation and error metrics (ER, MRED, MSE)
 */

#ifndef AXSIM_GPU_METRICS_HPP
#define AXSIM_GPU_METRICS_HPP

#include "circuit_soa.hpp"
#include <cstddef>
#include <vector>

namespace axsim {

struct GpuMetricsConfig {
    size_t num_patterns{65536};   ///< total random patterns (multiple of 64 for uint64 bit-parallel)
    unsigned int seed{12345};
    float delta{1e-6f};           ///< for MRED denominator
};

struct GpuMetricsResult {
    float error_rate{0.f};
    float mred{0.f};
    float mse{0.f};
};

/**
 * Allocate device SoA and run Monte Carlo simulation + compute metrics.
 * - Flattened circuit is copied to device
 * - Random patterns generated on GPU (or copied from host)
 * - Simulation runs; then reference outputs (e.g. from golden netlist) are
 *   compared to get ER / MRED / MSE via reduction
 *
 * @param soa         flattened circuit (CPU)
 * @param ref_outputs reference (golden) output bits, packed: num_outputs * ceil(num_patterns/64) uint64
 * @param config      pattern count, seed, delta
 * @return error_rate, mred, mse
 */
GpuMetricsResult run_gpu_metrics(
    const CircuitSoA& soa,
    const std::vector<uint64_t>& ref_outputs,
    const GpuMetricsConfig& config = {}
);

/**
 * Only run simulation and return approximate output bits on host (for testing).
 * Packed: num_outputs * ceil(num_patterns/64) uint64_t.
 */
std::vector<uint64_t> run_gpu_simulation_only(
    const CircuitSoA& soa,
    const GpuMetricsConfig& config = {}
);

} // namespace axsim

#endif
