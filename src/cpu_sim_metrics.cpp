/**
 * CPU path: ResubALS-style bit-parallel AIG MC — same levelized AND eval and LCG PI planes
 * as cuda/sim_kernels.cu (see monte_carlo_kernel); metrics must match run_gpu_metrics when
 * both backends succeed.
 */

#include "axsim/cpu_metrics.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace axsim {
namespace {

inline bool wall_timeout(const GpuMetricsConfig& config, std::chrono::steady_clock::time_point t0) {
    if (config.max_wall_seconds <= 0.0) return false;
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() >= config.max_wall_seconds;
}

unsigned int lcg_next(unsigned int* state) {
    *state = *state * 1103515245u + 12345u;
    return *state;
}

uint64_t random_bits_64(unsigned int* state) {
    uint64_t r = 0;
    for (int i = 0; i < 64; i += 32)
        r |= static_cast<uint64_t>(lcg_next(state)) << i;
    return r;
}

void eval_one_block(
    const CircuitSoA& soa,
    int block_idx,
    int num_blocks_64,
    unsigned int seed,
    const std::vector<std::uint64_t>* pi_packed,
    std::uint64_t* row)
{
    const int num_pis = soa.num_pis;
    const int num_ands = soa.num_ands;

    if (pi_packed != nullptr) {
        for (int pi = 0; pi < num_pis; ++pi)
            row[pi] = (*pi_packed)[static_cast<size_t>(pi) * static_cast<size_t>(num_blocks_64) +
                static_cast<size_t>(block_idx)];
    } else {
        unsigned int rng = seed + static_cast<unsigned int>(block_idx) * 97u;
        for (int i = 0; i < num_pis; ++i)
            row[i] = random_bits_64(&rng);
    }

    for (int k = 0; k < num_ands; ++k) {
        const int n = num_pis + k;
        const int f0 = soa.fanin0_indices[k];
        const int f1 = soa.fanin1_indices[k];
        const uint64_t v0 = soa.is_compl0[k] ? ~row[f0] : row[f0];
        const uint64_t v1 = soa.is_compl1[k] ? ~row[f1] : row[f1];
        row[n] = v0 & v1;
    }
}

uint64_t reconstruct_output_chunk(
    const std::vector<uint64_t>& packed_bits,
    size_t num_blocks_64,
    size_t blk,
    int bit,
    int o0,
    int n,
    bool outputs_msb_first)
{
    uint64_t out = 0;
    for (int local_o = 0; local_o < n; ++local_o) {
        const size_t idx = static_cast<size_t>(o0 + local_o) * num_blocks_64 + blk;
        const uint64_t b = (packed_bits[idx] >> bit) & 1ULL;
        const int pos = outputs_msb_first ? (n - 1 - local_o) : local_o;
        out |= (b << pos);
    }
    return out;
}

bool gather_and_complement_outputs(
    const CircuitSoA& soa,
    const std::vector<uint64_t>& node_values,
    size_t num_blocks_64,
    std::vector<uint64_t>& out_packed)
{
    const int num_outputs = soa.num_outputs;
    const int num_nodes = soa.num_nodes;
    out_packed.resize(static_cast<size_t>(num_outputs) * num_blocks_64);
    for (int o = 0; o < num_outputs; ++o) {
        const int nid = soa.output_node_ids[o];
        for (size_t b = 0; b < num_blocks_64; ++b) {
            const uint64_t* row = node_values.data() + b * static_cast<size_t>(num_nodes);
            out_packed[static_cast<size_t>(o) * num_blocks_64 + b] = row[nid];
        }
    }
    if (!soa.is_output_complemented.empty() &&
        static_cast<int>(soa.is_output_complemented.size()) == num_outputs) {
        for (int o = 0; o < num_outputs; ++o) {
            if (!soa.is_output_complemented[o]) continue;
            for (size_t b = 0; b < num_blocks_64; ++b)
                out_packed[static_cast<size_t>(o) * num_blocks_64 + b] =
                    ~out_packed[static_cast<size_t>(o) * num_blocks_64 + b];
        }
    }
    return true;
}

bool compute_word_metrics(
    const std::vector<uint64_t>& h_approx,
    const std::vector<uint64_t>& ref_outputs,
    size_t num_blocks_64,
    size_t requested_patterns,
    int num_outputs,
    double normalizer,
    bool outputs_msb_first,
    const GpuMetricsConfig& config,
    std::chrono::steady_clock::time_point t0,
    GpuMetricsResult& res)
{
    size_t err_patterns = 0;
    double mae_norm_sum = 0.0;
    double mse_sum = 0.0;
    const int num_chunks = (num_outputs + 63) / 64;
    const double inv_chunks = 1.0 / static_cast<double>(num_chunks);
    for (size_t p = 0; p < requested_patterns; ++p) {
        if (config.max_wall_seconds > 0.0 && (p % 100000u) == 0u && p > 0u && wall_timeout(config, t0)) {
            std::fprintf(stderr, "[axsim] CPU wall-clock timeout during metric reduction (max_wall_seconds=%.0f)\n",
                config.max_wall_seconds);
            res.timed_out = true;
            return false;
        }
        const size_t blk = p >> 6;
        const int bit = static_cast<int>(p & 63);
        double mse_p = 0.0;
        double mae_p = 0.0;
        bool match = true;
        for (int c = 0; c < num_chunks; ++c) {
            const int o0 = c * 64;
            const int n = (num_outputs - o0 < 64) ? (num_outputs - o0) : 64;
            const uint64_t wa = reconstruct_output_chunk(
                h_approx, num_blocks_64, blk, bit, o0, n, outputs_msb_first);
            const uint64_t wr = reconstruct_output_chunk(
                ref_outputs, num_blocks_64, blk, bit, o0, n, outputs_msb_first);
            if (wa != wr) match = false;
            const double diff = static_cast<double>(wa) - static_cast<double>(wr);
            mse_p += diff * diff;
            mae_p += std::abs(diff) / normalizer;
        }
        mse_sum += mse_p * inv_chunks;
        mae_norm_sum += mae_p * inv_chunks;
        if (!match) ++err_patterns;
    }
    res.error_rate = static_cast<float>(err_patterns) / static_cast<float>(requested_patterns);
    res.mae_norm = static_cast<float>(mae_norm_sum / static_cast<double>(requested_patterns));
    res.mred = res.mae_norm;
    res.mse = static_cast<float>(mse_sum / static_cast<double>(requested_patterns));
    return true;
}

} // namespace

std::vector<std::uint64_t> run_cpu_simulation_only(
    const CircuitSoA& soa,
    const GpuMetricsConfig& config)
{
    std::vector<std::uint64_t> out;
    if (!soa.valid()) return out;
    if (config.num_patterns == 0) return out;

    size_t num_patterns = config.num_patterns;
    if (num_patterns % 64 != 0) num_patterns = (num_patterns + 63) & ~(size_t)63;
    const size_t num_blocks_64 = num_patterns / 64;
    const int num_nodes = soa.num_nodes;
    const int num_outputs = soa.num_outputs;
    const size_t expected_pi_packed = static_cast<size_t>(soa.num_pis) * num_blocks_64;

    if (!config.external_pi_packed.empty()) {
        if (config.external_pi_packed.size() != expected_pi_packed) {
            std::fprintf(stderr,
                "[axsim] run_cpu_simulation_only: external_pi_packed size mismatch: expected %zu got %zu\n",
                expected_pi_packed, config.external_pi_packed.size());
            return out;
        }
    }

    std::vector<uint64_t> node_values(num_blocks_64 * static_cast<size_t>(num_nodes), 0);
    const std::vector<std::uint64_t>* pi_ptr =
        config.external_pi_packed.empty() ? nullptr : &config.external_pi_packed;

    config.wall_timed_out = false;
    const auto t_wall0 = std::chrono::steady_clock::now();
    for (int bi = 0; bi < static_cast<int>(num_blocks_64); ++bi) {
        if (wall_timeout(config, t_wall0)) {
            config.wall_timed_out = true;
            std::fprintf(stderr, "[axsim] CPU wall-clock timeout during simulation blocks (max_wall_seconds=%.0f)\n",
                config.max_wall_seconds);
            return out;
        }
        uint64_t* row = node_values.data() + static_cast<size_t>(bi) * static_cast<size_t>(num_nodes);
        eval_one_block(soa, bi, static_cast<int>(num_blocks_64), config.seed, pi_ptr, row);
    }

    out.resize(static_cast<size_t>(num_outputs) * num_blocks_64);
    if (!gather_and_complement_outputs(soa, node_values, num_blocks_64, out)) {
        out.clear();
    }
    return out;
}

GpuMetricsResult run_cpu_metrics(
    const CircuitSoA& soa,
    const std::vector<std::uint64_t>& ref_outputs,
    const GpuMetricsConfig& config)
{
    GpuMetricsResult res;
    if (!soa.valid()) return res;
    if (config.num_patterns == 0) {
        res.ok = true;
        return res;
    }

    const size_t requested_patterns = config.num_patterns;
    size_t num_patterns_rounded = requested_patterns;
    if (num_patterns_rounded % 64 != 0) num_patterns_rounded = (num_patterns_rounded + 63) & ~(size_t)63;
    const size_t num_blocks_64 = num_patterns_rounded / 64;

    const int num_nodes = soa.num_nodes;
    const int num_outputs = soa.num_outputs;
    if (num_outputs <= 0) {
        std::fprintf(stderr, "[axsim] run_cpu_metrics requires num_outputs >= 1, got %d\n", num_outputs);
        return res;
    }

    const size_t expected_ref_size = static_cast<size_t>(num_outputs) * num_blocks_64;
    if (ref_outputs.size() != expected_ref_size) {
        std::fprintf(stderr, "[axsim] run_cpu_metrics ref_outputs size mismatch: expected=%zu got=%zu\n",
            expected_ref_size, ref_outputs.size());
        return res;
    }

    const size_t expected_pi_packed = static_cast<size_t>(soa.num_pis) * num_blocks_64;
    if (!config.external_pi_packed.empty() &&
        config.external_pi_packed.size() != expected_pi_packed) {
        std::fprintf(stderr,
            "[axsim] run_cpu_metrics external_pi_packed size mismatch: expected %zu got %zu\n",
            expected_pi_packed, config.external_pi_packed.size());
        return res;
    }

    std::vector<uint64_t> node_values(num_blocks_64 * static_cast<size_t>(num_nodes), 0);
    const std::vector<std::uint64_t>* pi_ptr =
        config.external_pi_packed.empty() ? nullptr : &config.external_pi_packed;

    const auto t_wall0 = std::chrono::steady_clock::now();
    for (int bi = 0; bi < static_cast<int>(num_blocks_64); ++bi) {
        if (wall_timeout(config, t_wall0)) {
            res.timed_out = true;
            std::fprintf(stderr, "[axsim] CPU wall-clock timeout during simulation blocks (max_wall_seconds=%.0f)\n",
                config.max_wall_seconds);
            return res;
        }
        uint64_t* row = node_values.data() + static_cast<size_t>(bi) * static_cast<size_t>(num_nodes);
        eval_one_block(soa, bi, static_cast<int>(num_blocks_64), config.seed, pi_ptr, row);
    }

    std::vector<uint64_t> h_approx;
    if (!gather_and_complement_outputs(soa, node_values, num_blocks_64, h_approx)) {
        return res;
    }

    const double normalizer =
        (config.mae_normalizer > 0.0f) ? static_cast<double>(config.mae_normalizer) : 1.0;
    if (!compute_word_metrics(h_approx, ref_outputs, num_blocks_64, requested_patterns, num_outputs,
            normalizer, config.outputs_msb_first, config, t_wall0, res)) {
        return res;
    }
    res.ok = true;
    return res;
}

GpuMetricsResult run_cpu_metrics_pair(
    const CircuitSoA& approx_soa,
    const CircuitSoA& golden_soa,
    const GpuMetricsConfig& config)
{
    GpuMetricsResult res;
    if (!approx_soa.valid() || !golden_soa.valid()) return res;
    if (approx_soa.num_pis != golden_soa.num_pis || approx_soa.num_outputs != golden_soa.num_outputs) {
        std::fprintf(stderr,
            "[axsim] circuit-pair mismatch: approx(PIs=%d, POs=%d) vs golden(PIs=%d, POs=%d)\n",
            approx_soa.num_pis, approx_soa.num_outputs, golden_soa.num_pis, golden_soa.num_outputs);
        return res;
    }
    const std::vector<uint64_t> ref_outputs = run_cpu_simulation_only(golden_soa, config);
    if (config.wall_timed_out) {
        res.timed_out = true;
        return res;
    }
    if (ref_outputs.empty() && config.num_patterns != 0) return res;
    return run_cpu_metrics(approx_soa, ref_outputs, config);
}

} // namespace axsim
