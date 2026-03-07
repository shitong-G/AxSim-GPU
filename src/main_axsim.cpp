/**
 * Example: build a tiny AIG, flatten to SoA, run GPU sim + metrics
 *
 * Replace the manual AIG construction with your ABC parser when integrating.
 */

#include "axsim/circuit_soa.hpp"
#include "axsim/gpu_metrics.hpp"
#include <cstdio>
#include <vector>

int main() {
    // --- A. Build SoA (here: minimal 2-input AND gate as AIG)
    //     Node 0,1 = PIs, Node 2 = AND(0, 1). One output = node 2.
    const int num_pis = 2;
    std::vector<std::tuple<int, int, bool, bool>> and_nodes;
    and_nodes.push_back({0, 1, false, false});  // node 2 = and(node0, node1)
    std::vector<int> out_ids = {2};

    axsim::CircuitSoA soa = axsim::flatten_from_aig(num_pis, and_nodes, out_ids);
    if (!soa.valid()) {
        fprintf(stderr, "SoA invalid\n");
        return 1;
    }

    // --- Reference outputs: for 2 inputs, 2^2=4 patterns. We use 64 patterns (1 block) for demo.
    //     Golden: out = in0 & in1. Packed as 1 output * 1 uint64 = 64 bits (we only care first 4 for exact match).
    size_t num_patterns = 64;
    size_t num_blocks_64 = 1;
    std::vector<uint64_t> ref_outputs(soa.num_outputs * num_blocks_64);
    // Reference: pattern 0,1,2,3 -> (0,0),(1,0),(0,1),(1,1) -> out 0,0,0,1 -> bits 0,0,0,1
    ref_outputs[0] = 1u;  // bit0 = pattern 0, bit1 = pattern 1, ... so pattern 3 = bit3 = 1

    axsim::GpuMetricsConfig config;
    config.num_patterns = num_patterns;
    config.seed = 42;

    // --- B+C. Run GPU simulation and metrics
    axsim::GpuMetricsResult res = axsim::run_gpu_metrics(soa, ref_outputs, config);

    printf("Error rate: %.6f\n", res.error_rate);
    printf("MRED:       %.6f\n", res.mred);
    printf("MSE:        %.6f\n", res.mse);

    return 0;
}
