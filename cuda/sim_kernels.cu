/**
 * GPU Monte Carlo simulation and metric reduction kernels
 *
 * Build: nvcc -std=c++17 -O3 -arch=sm_70 ... or use CMake with FindCUDAToolkit
 * Optional: CUB for reduction (e.g. -I/path/to/cub or CUDA toolkit CUB)
 */

#include <cstdint>
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
    unsigned int seed)
{
    // Block handles a slice of pattern blocks; each thread handles one uint64_t
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= num_blocks_64) return;

    unsigned int rng = seed + block_idx * 97u;

    // Shared memory: load PI values for this block (all threads in block use same PIs for same block_idx if you pack by block)
    // Layout: node_values[block_idx * num_nodes + node_id] for this block's patterns
    // So we need node_values of size num_nodes * num_blocks_64; thread block_idx writes to offset block_idx * num_nodes.
    uint64_t* row = node_values + (size_t)block_idx * soa.num_nodes;

    // Generate random bits for PIs (this thread owns this row)
    for (int i = 0; i < soa.num_pis; i++)
        row[i] = random_bits_64(&rng);
    __syncthreads();

    // Evaluate AND nodes in order (each node depends only on earlier nodes)
    for (int k = 0; k < soa.num_ands; k++) {
        int n = soa.num_pis + k;
        int f0 = soa.fanin0[k];
        int f1 = soa.fanin1[k];
        uint64_t v0 = row[f0];
        uint64_t v1 = row[f1];
        if (soa.is_compl0[k]) v0 = ~v0;
        if (soa.is_compl1[k]) v1 = ~v1;
        row[n] = v0 & v1;
    }
}

// Optional: use shared memory to cache PI row for this block (reduce global loads for ANDs that fanin from PIs)
// #define USE_SHARED_PI 1

// Overload: single block of threads each does one uint64_t (num_blocks_64 threads total per grid)
__global__ void monte_carlo_kernel_simple(
    const DeviceCircuitSoA soa,
    uint64_t* __restrict__ node_values,
    int num_blocks_64,
    unsigned int seed)
{
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
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

// MRED: per-pattern |g-f|/max(|f|, delta). For 0/1 outputs, |g-f| is 0 or 1; interpret as float.
// Pack: we have bits, so "value" = popc of output word for multi-output, or use single output.
// Skeleton: compute per-thread sum of relative errors, then block reduce.
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_outputs * num_blocks_64;
    for (int i = idx; i < total; i += gridDim.x * blockDim.x) {
        uint64_t a = approx[i];
        uint64_t r = reference[i];
        uint64_t diff = a ^ r;
        int c = __popcll(diff);
        if (c == 0) continue;
        // For boolean: each differing bit contributes 1.0 / max(1, delta) or 1.0 when f=1
        // Simplified: relative error = |a-r|/max(|r|, delta). For bits, |r| is 0 or 1 -> 1/max(1,delta) or 1/1
        double denom = fmax((double)__popcll(r), (double)delta);
        local += (double)c / denom;
    }
    sh[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) block_sums[blockIdx.x] = sh[0];
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

namespace axsim {

namespace {

const int BLOCK = 256;
const int MAX_BLOCKS = 1024;

}

GpuMetricsResult run_gpu_metrics(
    const CircuitSoA& soa,
    const std::vector<uint64_t>& ref_outputs,
    const GpuMetricsConfig& config)
{
    GpuMetricsResult res;
    if (!soa.valid()) return res;

    size_t num_patterns = config.num_patterns;
    if (num_patterns % 64 != 0) num_patterns = (num_patterns + 63) & ~(size_t)63;
    size_t num_blocks_64 = num_patterns / 64;

    int num_ands = soa.num_ands;
    int num_nodes = soa.num_nodes;
    int num_pis = soa.num_pis;
    int num_outputs = soa.num_outputs;

    size_t node_values_bytes = num_blocks_64 * num_nodes * sizeof(uint64_t);
    size_t ref_bytes = num_outputs * num_blocks_64 * sizeof(uint64_t);
    if (ref_outputs.size() < num_outputs * num_blocks_64) return res;

    // Allocate device
    int *d_f0, *d_f1;
    uint8_t *d_c0, *d_c1;
    uint64_t *d_node_values, *d_ref, *d_approx_out;
    uint64_t *d_err_blocks;
    double* d_mred_blocks;

    cudaMalloc(&d_f0, num_ands * sizeof(int));
    cudaMalloc(&d_f1, num_ands * sizeof(int));
    cudaMalloc(&d_c0, num_ands * sizeof(uint8_t));
    cudaMalloc(&d_c1, num_ands * sizeof(uint8_t));
    cudaMalloc(&d_node_values, node_values_bytes);
    cudaMalloc(&d_ref, ref_bytes);
    cudaMalloc(&d_approx_out, ref_bytes);
    cudaMalloc(&d_err_blocks, MAX_BLOCKS * sizeof(uint64_t));
    cudaMalloc(&d_mred_blocks, MAX_BLOCKS * sizeof(double));

    cudaMemcpy(d_f0, soa.fanin0_indices.data(), num_ands * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_f1, soa.fanin1_indices.data(), num_ands * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c0, soa.is_compl0.data(), num_ands * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c1, soa.is_compl1.data(), num_ands * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ref, ref_outputs.data(), ref_bytes, cudaMemcpyHostToDevice);

    DeviceCircuitSoA d_soa;
    d_soa.fanin0 = d_f0;
    d_soa.fanin1 = d_f1;
    d_soa.is_compl0 = d_c0;
    d_soa.is_compl1 = d_c1;
    d_soa.num_pis = num_pis;
    d_soa.num_ands = num_ands;
    d_soa.num_nodes = num_nodes;

    // Launch Monte Carlo (one thread per block of 64 patterns)
    int nblk = (int)((num_blocks_64 + BLOCK - 1) / BLOCK);
    if (nblk > 65535) nblk = 65535;
    monte_carlo_kernel_simple<<<nblk, BLOCK>>>(d_soa, d_node_values, (int)num_blocks_64, config.seed);

    // Extract output node values into d_approx_out [num_outputs * num_blocks_64]
    // TODO: kernel that gathers row[soa.output_node_ids[o]] for each output o and block_idx
    // For skeleton: assume single output at last node
    {
        size_t pitch = num_blocks_64 * sizeof(uint64_t);
        for (int o = 0; o < num_outputs; o++) {
            int nid = soa.output_node_ids[o];
            uint64_t* src = d_node_values + nid;  // strided: node_values[block_idx*num_nodes + nid]
            uint64_t* dst = d_approx_out + o * num_blocks_64;
            // Copy strided to contiguous: need a small kernel or cudaMemcpy2D
            cudaMemcpy2D(dst, sizeof(uint64_t), src, num_nodes * sizeof(uint64_t), sizeof(uint64_t), (int)num_blocks_64, cudaMemcpyDeviceToDevice);
        }
    }

    // Error rate reduction (launch enough blocks to cover output size)
    int nred = (int)(num_outputs * num_blocks_64 + BLOCK - 1) / BLOCK;
    if (nred > MAX_BLOCKS) nred = MAX_BLOCKS;
    if (nred < 1) nred = 1;
    error_count_kernel<<<nred, BLOCK>>>(d_approx_out, d_ref, num_outputs, (int)num_blocks_64, d_err_blocks);
    std::vector<uint64_t> h_err(nred);
    cudaMemcpy(h_err.data(), d_err_blocks, nred * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    uint64_t total_err = 0;
    for (int i = 0; i < nred; i++) total_err += h_err[i];
    res.error_rate = (float)total_err / (float)(num_outputs * num_patterns);

    // MRED reduction
    mred_sum_kernel<<<nred, BLOCK>>>(d_approx_out, d_ref, num_outputs, (int)num_blocks_64, config.delta, d_mred_blocks);
    std::vector<double> h_mred(nred);
    cudaMemcpy(h_mred.data(), d_mred_blocks, nred * sizeof(double), cudaMemcpyDeviceToHost);
    double mred_sum = 0;
    for (int i = 0; i < nred; i++) mred_sum += h_mred[i];
    res.mred = (float)(mred_sum / (double)(num_outputs * num_patterns));

    // MSE = same as error count for 0/1, then mean
    res.mse = res.error_rate;  // for boolean outputs ER == MSE

    cudaFree(d_mred_blocks);
    cudaFree(d_err_blocks);
    cudaFree(d_approx_out);
    cudaFree(d_ref);
    cudaFree(d_node_values);
    cudaFree(d_c1);
    cudaFree(d_c0);
    cudaFree(d_f1);
    cudaFree(d_f0);

    return res;
}

std::vector<uint64_t> run_gpu_simulation_only(
    const CircuitSoA& soa,
    const GpuMetricsConfig& config)
{
    std::vector<uint64_t> out;
    if (!soa.valid()) return out;

    size_t num_patterns = config.num_patterns;
    if (num_patterns % 64 != 0) num_patterns = (num_patterns + 63) & ~(size_t)63;
    size_t num_blocks_64 = num_patterns / 64;
    int num_outputs = soa.num_outputs;
    size_t out_size = num_outputs * num_blocks_64;

    int num_ands = soa.num_ands;
    int num_nodes = soa.num_nodes;
    int num_pis = soa.num_pis;

    int *d_f0, *d_f1;
    uint8_t *d_c0, *d_c1;
    uint64_t* d_node_values;
    cudaMalloc(&d_f0, num_ands * sizeof(int));
    cudaMalloc(&d_f1, num_ands * sizeof(int));
    cudaMalloc(&d_c0, num_ands * sizeof(uint8_t));
    cudaMalloc(&d_c1, num_ands * sizeof(uint8_t));
    cudaMalloc(&d_node_values, num_blocks_64 * num_nodes * sizeof(uint64_t));

    cudaMemcpy(d_f0, soa.fanin0_indices.data(), num_ands * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_f1, soa.fanin1_indices.data(), num_ands * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c0, soa.is_compl0.data(), num_ands * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c1, soa.is_compl1.data(), num_ands * sizeof(uint8_t), cudaMemcpyHostToDevice);

    DeviceCircuitSoA d_soa;
    d_soa.fanin0 = d_f0; d_soa.fanin1 = d_f1;
    d_soa.is_compl0 = d_c0; d_soa.is_compl1 = d_c1;
    d_soa.num_pis = num_pis; d_soa.num_ands = num_ands; d_soa.num_nodes = num_nodes;

    int nblk = (int)((num_blocks_64 + BLOCK - 1) / BLOCK);
    if (nblk > 65535) nblk = 65535;
    monte_carlo_kernel_simple<<<nblk, BLOCK>>>(d_soa, d_node_values, (int)num_blocks_64, config.seed);

    out.resize(out_size);
    for (int o = 0; o < num_outputs; o++) {
        int nid = soa.output_node_ids[o];
        uint64_t* src = d_node_values + nid;
        uint64_t* dst = out.data() + o * num_blocks_64;
        cudaMemcpy2D(dst, sizeof(uint64_t), src, num_nodes * sizeof(uint64_t), sizeof(uint64_t), (int)num_blocks_64, cudaMemcpyDeviceToHost);
    }

    cudaFree(d_node_values);
    cudaFree(d_c1);
    cudaFree(d_c0);
    cudaFree(d_f1);
    cudaFree(d_f0);
    return out;
}

} // namespace axsim
