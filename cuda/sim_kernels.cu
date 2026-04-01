/**
 * GPU Monte Carlo simulation and metric reduction kernels
 *
 * Build: nvcc -std=c++17 -O3 -arch=sm_70 ... or use CMake with FindCUDAToolkit
 * Optional: CUB for reduction (e.g. -I/path/to/cub or CUDA toolkit CUB)
 */

#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// A. Device-side SoA (filled by host after cudaMemcpy)
// -----------------------------------------------------------------------------

struct DeviceCircuitSoA {
    const int* fanin0;
    const int* fanin1;
    const uint8_t* is_compl0;
    const uint8_t* is_compl1;
    int num_pis;
    int num_ands;
    int num_nodes;
};

// -----------------------------------------------------------------------------
// Simple LCG for random bits (replace with curand if needed)
// -----------------------------------------------------------------------------

__device__ unsigned int lcg_next(unsigned int* state) {
    *state = *state * 1103515245u + 12345u;
    return *state;
}

__device__ uint64_t random_bits_64(unsigned int* state) {
    uint64_t r = 0;
    for (int i = 0; i < 64; i += 32) {
        r |= (uint64_t)lcg_next(state) << i;
    }
    return r;
}

// -----------------------------------------------------------------------------
// B. Monte Carlo simulation kernel (one uint64_t per thread = 64 patterns)
// -----------------------------------------------------------------------------

__global__ void monte_carlo_kernel(
    const DeviceCircuitSoA soa,
    uint64_t* __restrict__ node_values,  // [num_nodes * num_blocks_64]
    int num_blocks_64,                    // num_patterns / 64
    unsigned int seed,
    int start_block)
{
    // Block handles a slice of pattern blocks; each thread handles one uint64_t
    int block_idx = start_block + blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= num_blocks_64) return;

    unsigned int rng = seed + block_idx * 97u;

    // Each thread owns one "row" (64 patterns packed in uint64_t).
    // Layout: node_values[block_idx * num_nodes + node_id].
    uint64_t* row = node_values + (size_t)block_idx * soa.num_nodes;

    // Generate random bits for PIs (this thread owns this row)
    #pragma unroll 4
    for (int i = 0; i < soa.num_pis; i++)
        row[i] = random_bits_64(&rng);

    // Shared-memory tiling for AND descriptors:
    // all threads in a block evaluate the same k in lockstep, so cache fanins/complements once per tile.
    constexpr int AND_TILE = 256;  // must be <= blockDim.x used at launch (BLOCK=256)
    __shared__ int sh_f0[AND_TILE];
    __shared__ int sh_f1[AND_TILE];
    __shared__ uint8_t sh_c0[AND_TILE];
    __shared__ uint8_t sh_c1[AND_TILE];

    for (int base = 0; base < soa.num_ands; base += AND_TILE) {
        int tile = soa.num_ands - base;
        if (tile > AND_TILE) tile = AND_TILE;

        if (threadIdx.x < tile) {
            const int gk = base + (int)threadIdx.x;
            sh_f0[threadIdx.x] = soa.fanin0[gk];
            sh_f1[threadIdx.x] = soa.fanin1[gk];
            sh_c0[threadIdx.x] = soa.is_compl0[gk];
            sh_c1[threadIdx.x] = soa.is_compl1[gk];
        }
        __syncthreads();

        #pragma unroll 4
        for (int t = 0; t < tile; ++t) {
            const int n = soa.num_pis + base + t;
            const uint64_t v0 = sh_c0[t] ? ~row[sh_f0[t]] : row[sh_f0[t]];
            const uint64_t v1 = sh_c1[t] ? ~row[sh_f1[t]] : row[sh_f1[t]];
            row[n] = v0 & v1;
        }
        __syncthreads();
    }
}

// Optional: use shared memory to cache PI row for this block (reduce global loads for ANDs that fanin from PIs)
// #define USE_SHARED_PI 1

// Overload: single block of threads each does one uint64_t (num_blocks_64 threads total per grid)
__global__ void monte_carlo_kernel_simple(
    const DeviceCircuitSoA soa,
    uint64_t* __restrict__ node_values,
    int num_blocks_64,
    unsigned int seed,
    int start_block)
{
    int block_idx = start_block + blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= num_blocks_64) return;

    unsigned int rng = seed + block_idx * 97u;
    uint64_t* row = node_values + (size_t)block_idx * soa.num_nodes;

    for (int i = 0; i < soa.num_pis; i++) {
        if (i == 0) row[i] = random_bits_64(&rng);
        else row[i] = random_bits_64(&rng);  // TODO: coalesce PI generation with other threads
    }

    for (int k = 0; k < soa.num_ands; k++) {
        int n = soa.num_pis + k;
        int f0 = soa.fanin0[k], f1 = soa.fanin1[k];
        uint64_t v0 = soa.is_compl0[k] ? ~row[f0] : row[f0];
        uint64_t v1 = soa.is_compl1[k] ? ~row[f1] : row[f1];
        row[n] = v0 & v1;
    }
}

// -----------------------------------------------------------------------------
// C. Metric reductions
// -----------------------------------------------------------------------------

// Error rate: sum of popc(g ^ f) over all output bits, then divide by (num_outputs * num_patterns)
__global__ void error_count_kernel(
    const uint64_t* __restrict__ approx,   // [num_outputs * num_blocks_64]
    const uint64_t* __restrict__ reference,
    int num_outputs,
    int num_blocks_64,
    uint64_t* __restrict__ block_sums)     // one sum per block, then reduce again on host or second kernel
{
    __shared__ uint64_t sh[256];
    uint64_t local = 0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_outputs * num_blocks_64;
    for (int i = idx; i < total; i += gridDim.x * blockDim.x) {
        local += __popcll(approx[i] ^ reference[i]);
    }
    sh[threadIdx.x] = local;
    __syncthreads();

    // Block reduce (simple tree)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) block_sums[blockIdx.x] = sh[0];
}

// MRED: mean over all output bits of |g-f|/max(|f|,delta). Each packed bit is one pattern (boolean 0/1).
__global__ void mred_sum_kernel(
    const uint64_t* __restrict__ approx,
    const uint64_t* __restrict__ reference,
    int num_outputs,
    int num_blocks_64,
    float delta,
    double* __restrict__ block_sums)
{
    __shared__ double sh[256];
    double local = 0.0;
    double d = (double)delta;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_outputs * num_blocks_64;
    for (int i = idx; i < total; i += gridDim.x * blockDim.x) {
        uint64_t a = approx[i];
        uint64_t r = reference[i];
        uint64_t diff = a ^ r;
        if (diff == 0) continue;
        for (int bit = 0; bit < 64; ++bit) {
            uint64_t mask = 1ull << bit;
            if ((diff & mask) == 0) continue;
            double fbit = (r & mask) ? 1.0 : 0.0;
            double denom = fmax(fbit, d);  // max(|f|, delta) for f in {0,1}
            local += 1.0 / denom;
        }
    }
    sh[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) block_sums[blockIdx.x] = sh[0];
}

// Apply output complement: for each output o with is_compl[o]==1, flip all bits in that output row
__global__ void apply_output_complement_kernel(
    uint64_t* __restrict__ out_bits,
    const uint8_t* __restrict__ is_compl,
    int num_outputs,
    int num_blocks_64)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_outputs * num_blocks_64;
    if (idx >= total) return;
    int o = idx / num_blocks_64;
    if (is_compl[o]) out_bits[idx] = ~out_bits[idx];
}

// MSE: (1/N)*sum (g-f)^2. For 0/1: (g-f)^2 is 0 or 1, so sum of squared errors = popc(g^f).
__global__ void mse_sum_kernel(
    const uint64_t* __restrict__ approx,
    const uint64_t* __restrict__ reference,
    int num_outputs,
    int num_blocks_64,
    uint64_t* __restrict__ block_sums)
{
    __shared__ uint64_t sh[256];
    uint64_t local = 0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_outputs * num_blocks_64;
    for (int i = idx; i < total; i += gridDim.x * blockDim.x)
        local += __popcll(approx[i] ^ reference[i]);  // (g-f)^2 for 0/1 is 0 or 1
    sh[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) block_sums[blockIdx.x] = sh[0];
}

// -----------------------------------------------------------------------------
// Host API: launch simulation and metrics (implement in .cu or .cpp with CUDA)
// -----------------------------------------------------------------------------

#include "axsim/circuit_soa.hpp"
#include "axsim/gpu_metrics.hpp"
#include <vector>
#include <cstring>
#include <cstdio>

namespace axsim {

namespace {

const int BLOCK = 256;
const int MAX_GRID_X = 65535;

bool cuda_ok(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[axsim][cuda] %s failed: %s\n", what, cudaGetErrorString(err));
        return false;
    }
    return true;
}

bool launch_monte_carlo_checked(
    const DeviceCircuitSoA& d_soa,
    uint64_t* d_node_values,
    int num_blocks_64,
    unsigned int seed)
{
    if (num_blocks_64 <= 0) return true;
    const int max_blocks_per_launch = MAX_GRID_X * BLOCK;
    for (int start = 0; start < num_blocks_64; start += max_blocks_per_launch) {
        const int remaining = num_blocks_64 - start;
        const int this_launch_threads = (remaining < max_blocks_per_launch) ? remaining : max_blocks_per_launch;
        const int nblk = (this_launch_threads + BLOCK - 1) / BLOCK;
        monte_carlo_kernel<<<nblk, BLOCK>>>(d_soa, d_node_values, num_blocks_64, seed, start);
        if (!cuda_ok(cudaGetLastError(), "launch monte_carlo_kernel")) return false;
    }
    return cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize after monte_carlo_kernel");
}

inline uint64_t reconstruct_output_word(
    const std::vector<uint64_t>& packed_bits,
    size_t num_blocks_64,
    size_t blk,
    int bit,
    int num_outputs,
    bool outputs_msb_first)
{
    uint64_t out = 0;
    for (int o = 0; o < num_outputs; ++o) {
        const size_t idx = (size_t)o * num_blocks_64 + blk;
        const uint64_t b = (packed_bits[idx] >> bit) & 1ULL;
        const int pos = outputs_msb_first ? (num_outputs - 1 - o) : o;
        out |= (b << pos);
    }
    return out;
}
}

GpuMetricsResult run_gpu_metrics(
    const CircuitSoA& soa,
    const std::vector<uint64_t>& ref_outputs,
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

    const int num_ands = soa.num_ands;
    const int num_nodes = soa.num_nodes;
    const int num_pis = soa.num_pis;
    const int num_outputs = soa.num_outputs;
    if (num_outputs <= 0 || num_outputs > 64) {
        std::fprintf(stderr, "[axsim] run_gpu_metrics requires 1..64 outputs, got %d\n", num_outputs);
        return res;
    }

    const size_t expected_ref_size = (size_t)num_outputs * num_blocks_64;
    if (ref_outputs.size() != expected_ref_size) {
        std::fprintf(stderr, "[axsim] ref_outputs size mismatch: expected=%zu got=%zu\n",
            expected_ref_size, ref_outputs.size());
        return res;
    }

    const size_t node_values_bytes = num_blocks_64 * (size_t)num_nodes * sizeof(uint64_t);
    const size_t ref_bytes = expected_ref_size * sizeof(uint64_t);

    int* d_f0 = nullptr;
    int* d_f1 = nullptr;
    uint8_t* d_c0 = nullptr;
    uint8_t* d_c1 = nullptr;
    uint8_t* d_compl = nullptr;
    uint64_t* d_node_values = nullptr;
    uint64_t* d_approx_out = nullptr;

    bool success = false;
    do {
        if (!cuda_ok(cudaMalloc(&d_f0, num_ands * sizeof(int)), "cudaMalloc d_f0")) break;
        if (!cuda_ok(cudaMalloc(&d_f1, num_ands * sizeof(int)), "cudaMalloc d_f1")) break;
        if (!cuda_ok(cudaMalloc(&d_c0, num_ands * sizeof(uint8_t)), "cudaMalloc d_c0")) break;
        if (!cuda_ok(cudaMalloc(&d_c1, num_ands * sizeof(uint8_t)), "cudaMalloc d_c1")) break;
        if (!cuda_ok(cudaMalloc(&d_node_values, node_values_bytes), "cudaMalloc d_node_values")) break;
        if (!cuda_ok(cudaMalloc(&d_approx_out, ref_bytes), "cudaMalloc d_approx_out")) break;

        if (!cuda_ok(cudaMemcpy(d_f0, soa.fanin0_indices.data(), num_ands * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy d_f0")) break;
        if (!cuda_ok(cudaMemcpy(d_f1, soa.fanin1_indices.data(), num_ands * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy d_f1")) break;
        if (!cuda_ok(cudaMemcpy(d_c0, soa.is_compl0.data(), num_ands * sizeof(uint8_t), cudaMemcpyHostToDevice), "cudaMemcpy d_c0")) break;
        if (!cuda_ok(cudaMemcpy(d_c1, soa.is_compl1.data(), num_ands * sizeof(uint8_t), cudaMemcpyHostToDevice), "cudaMemcpy d_c1")) break;

        DeviceCircuitSoA d_soa;
        d_soa.fanin0 = d_f0;
        d_soa.fanin1 = d_f1;
        d_soa.is_compl0 = d_c0;
        d_soa.is_compl1 = d_c1;
        d_soa.num_pis = num_pis;
        d_soa.num_ands = num_ands;
        d_soa.num_nodes = num_nodes;

        if (!launch_monte_carlo_checked(d_soa, d_node_values, (int)num_blocks_64, config.seed)) break;

        for (int o = 0; o < num_outputs; ++o) {
            const int nid = soa.output_node_ids[o];
            uint64_t* src = d_node_values + nid;
            uint64_t* dst = d_approx_out + (size_t)o * num_blocks_64;
            if (!cuda_ok(cudaMemcpy2D(
                    dst, sizeof(uint64_t),
                    src, num_nodes * sizeof(uint64_t),
                    sizeof(uint64_t), (int)num_blocks_64,
                    cudaMemcpyDeviceToDevice),
                "cudaMemcpy2D gather outputs")) break;
        }
        if (!cuda_ok(cudaGetLastError(), "post-gather error check")) break;

        if (!soa.is_output_complemented.empty() && (int)soa.is_output_complemented.size() == num_outputs) {
            if (!cuda_ok(cudaMalloc(&d_compl, num_outputs * sizeof(uint8_t)), "cudaMalloc d_compl")) break;
            if (!cuda_ok(cudaMemcpy(d_compl, soa.is_output_complemented.data(), num_outputs * sizeof(uint8_t), cudaMemcpyHostToDevice), "cudaMemcpy d_compl")) break;
            const int nflip = num_outputs * (int)num_blocks_64;
            apply_output_complement_kernel<<<(nflip + BLOCK - 1) / BLOCK, BLOCK>>>(d_approx_out, d_compl, num_outputs, (int)num_blocks_64);
            if (!cuda_ok(cudaGetLastError(), "launch apply_output_complement_kernel")) break;
            if (!cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize after apply_output_complement_kernel")) break;
        }

        std::vector<uint64_t> h_approx(expected_ref_size);
        if (!cuda_ok(cudaMemcpy(h_approx.data(), d_approx_out, ref_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy d_approx_out->host")) break;

        size_t err_patterns = 0;
        double mae_norm_sum = 0.0;
        double mse_sum = 0.0;
        const double normalizer = (config.mae_normalizer > 0.0f) ? (double)config.mae_normalizer : 1.0;
        for (size_t p = 0; p < requested_patterns; ++p) {
            const size_t blk = p >> 6;
            const int bit = (int)(p & 63);
            const uint64_t approx_word = reconstruct_output_word(
                h_approx, num_blocks_64, blk, bit, num_outputs, config.outputs_msb_first);
            const uint64_t ref_word = reconstruct_output_word(
                ref_outputs, num_blocks_64, blk, bit, num_outputs, config.outputs_msb_first);

            if (approx_word != ref_word) ++err_patterns;
            const double diff = (double)approx_word - (double)ref_word;
            mse_sum += diff * diff;
            mae_norm_sum += std::abs(diff) / normalizer;
        }

        res.error_rate = (float)err_patterns / (float)requested_patterns;
        res.mae_norm = (float)(mae_norm_sum / (double)requested_patterns);
        res.mred = res.mae_norm;  // backward-compatible alias
        res.mse = (float)(mse_sum / (double)requested_patterns);
        success = true;
    } while (false);

    cudaFree(d_compl);
    cudaFree(d_approx_out);
    cudaFree(d_node_values);
    cudaFree(d_c1);
    cudaFree(d_c0);
    cudaFree(d_f1);
    cudaFree(d_f0);

    res.ok = success;
    return res;
}

std::vector<uint64_t> run_gpu_simulation_only(
    const CircuitSoA& soa,
    const GpuMetricsConfig& config)
{
    std::vector<uint64_t> out;
    if (!soa.valid()) return out;

    if (config.num_patterns == 0) return out;

    size_t num_patterns = config.num_patterns;
    if (num_patterns % 64 != 0) num_patterns = (num_patterns + 63) & ~(size_t)63;
    const size_t num_blocks_64 = num_patterns / 64;
    const int num_outputs = soa.num_outputs;
    const size_t out_size = (size_t)num_outputs * num_blocks_64;

    const int num_ands = soa.num_ands;
    const int num_nodes = soa.num_nodes;
    const int num_pis = soa.num_pis;

    int* d_f0 = nullptr;
    int* d_f1 = nullptr;
    uint8_t* d_c0 = nullptr;
    uint8_t* d_c1 = nullptr;
    uint64_t* d_node_values = nullptr;
    bool success = false;

    do {
        if (!cuda_ok(cudaMalloc(&d_f0, num_ands * sizeof(int)), "cudaMalloc d_f0")) break;
        if (!cuda_ok(cudaMalloc(&d_f1, num_ands * sizeof(int)), "cudaMalloc d_f1")) break;
        if (!cuda_ok(cudaMalloc(&d_c0, num_ands * sizeof(uint8_t)), "cudaMalloc d_c0")) break;
        if (!cuda_ok(cudaMalloc(&d_c1, num_ands * sizeof(uint8_t)), "cudaMalloc d_c1")) break;
        if (!cuda_ok(cudaMalloc(&d_node_values, num_blocks_64 * (size_t)num_nodes * sizeof(uint64_t)), "cudaMalloc d_node_values")) break;

        if (!cuda_ok(cudaMemcpy(d_f0, soa.fanin0_indices.data(), num_ands * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy d_f0")) break;
        if (!cuda_ok(cudaMemcpy(d_f1, soa.fanin1_indices.data(), num_ands * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy d_f1")) break;
        if (!cuda_ok(cudaMemcpy(d_c0, soa.is_compl0.data(), num_ands * sizeof(uint8_t), cudaMemcpyHostToDevice), "cudaMemcpy d_c0")) break;
        if (!cuda_ok(cudaMemcpy(d_c1, soa.is_compl1.data(), num_ands * sizeof(uint8_t), cudaMemcpyHostToDevice), "cudaMemcpy d_c1")) break;

        DeviceCircuitSoA d_soa;
        d_soa.fanin0 = d_f0;
        d_soa.fanin1 = d_f1;
        d_soa.is_compl0 = d_c0;
        d_soa.is_compl1 = d_c1;
        d_soa.num_pis = num_pis;
        d_soa.num_ands = num_ands;
        d_soa.num_nodes = num_nodes;

        if (!launch_monte_carlo_checked(d_soa, d_node_values, (int)num_blocks_64, config.seed)) break;

        out.resize(out_size);
        for (int o = 0; o < num_outputs; ++o) {
            const int nid = soa.output_node_ids[o];
            uint64_t* src = d_node_values + nid;
            uint64_t* dst = out.data() + (size_t)o * num_blocks_64;
            if (!cuda_ok(cudaMemcpy2D(
                    dst, sizeof(uint64_t),
                    src, num_nodes * sizeof(uint64_t),
                    sizeof(uint64_t), (int)num_blocks_64,
                    cudaMemcpyDeviceToHost),
                "cudaMemcpy2D gather outputs to host")) break;
        }
        if (!cuda_ok(cudaGetLastError(), "post-gather host copy error check")) break;

        if (!soa.is_output_complemented.empty() && (int)soa.is_output_complemented.size() == num_outputs) {
            for (int o = 0; o < num_outputs; ++o) {
                if (!soa.is_output_complemented[o]) continue;
                for (size_t b = 0; b < num_blocks_64; ++b) {
                    out[(size_t)o * num_blocks_64 + b] = ~out[(size_t)o * num_blocks_64 + b];
                }
            }
        }
        success = true;
    } while (false);

    if (!success) out.clear();
    cudaFree(d_node_values);
    cudaFree(d_c1);
    cudaFree(d_c0);
    cudaFree(d_f1);
    cudaFree(d_f0);
    return out;
}

GpuMetricsResult run_gpu_metrics_pair(
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
    const std::vector<uint64_t> ref_outputs = run_gpu_simulation_only(golden_soa, config);
    if (ref_outputs.empty() && config.num_patterns != 0) return res;
    return run_gpu_metrics(approx_soa, ref_outputs, config);
}

} // namespace axsim
