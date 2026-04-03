/**
 * GPU Monte Carlo simulation and error metrics (ER, MRED, MSE)
 */

#ifndef AXSIM_GPU_METRICS_HPP
#define AXSIM_GPU_METRICS_HPP

#include "circuit_soa.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace axsim {

struct GpuMetricsConfig {
    size_t num_patterns{65536};   ///< total random patterns (multiple of 64 for uint64 bit-parallel)
    unsigned int seed{12345};
    float delta{1e-6f};           ///< reserved (legacy bit-level MRED path)
    float mae_normalizer{255.0f}; ///< MAE% denominator (abs_diff_qor uses 2^8-1 = 255)
    bool outputs_msb_first{true}; ///< reconstruct integer as [o0..on-1]=[MSB..LSB] when true
    /// If non-empty, use these packed PI planes (AXPI010 / pattern_io.py) instead of GPU LCG RNG.
    /// Size must be num_pis * ceil(num_patterns/64) uint64 values.
    std::vector<std::uint64_t> external_pi_packed;
};

struct GpuMetricsResult {
    float error_rate{0.f};  ///< pattern-level: #mismatch patterns / #patterns
    float mae_norm{0.f};    ///< normalized MAE (not percent). MAE% = mae_norm * 100
    float mred{0.f};        ///< backward-compatible alias of mae_norm
    float mse{0.f};         ///< word-level mean squared error
    bool ok{false};         ///< false when input validation/CUDA runtime fails
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
 * Safe pair API for exact-vs-approx comparison.
 * Validates that both circuits have identical PI/PO counts before simulation.
 */
GpuMetricsResult run_gpu_metrics_pair(
    const CircuitSoA& approx_soa,
    const CircuitSoA& golden_soa,
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
