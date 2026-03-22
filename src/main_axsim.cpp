/**
 * Example: treat AND as reference (golden) and OR as approximate — same PIs / same seed
 * so patterns align; run_gpu_metrics(approx_soa, ref, config) is the intended API shape.
 *
 * OR = ~(~a & ~b): one AND with both fanins complemented + PO complement.
 */

#include "axsim/circuit_soa.hpp"
#include "axsim/gpu_metrics.hpp"
#include <cstdio>
#include <tuple>
#include <vector>

int main() {
    const int num_pis = 2;

    // Golden: y = a & b
    std::vector<std::tuple<int, int, bool, bool>> and_only;
    and_only.push_back({0, 1, false, false});
    axsim::CircuitSoA soa_and = axsim::flatten_from_aig(num_pis, and_only, {2});

    // Approximate (for this demo): y = a | b  (De Morgan on AIG)
    std::vector<std::tuple<int, int, bool, bool>> or_as_aig;
    or_as_aig.push_back({0, 1, true, true});
    axsim::CircuitSoA soa_or = axsim::flatten_from_aig(num_pis, or_as_aig, {2});
    soa_or.is_output_complemented = {1};

    if (!soa_and.valid() || !soa_or.valid()) {
        fprintf(stderr, "SoA invalid\n");
        return 1;
    }

    printf("Golden (ref): AND(in0,in1)  |  Approx (simulated): OR(in0,in1)\n");
    printf("Same seed -> same random PI vectors; metrics = mismatch between the two circuits.\n\n");

    axsim::GpuMetricsConfig config;
    config.num_patterns = 64;
    config.seed = 42;
    // Boolean outputs: golden f=0 makes |g-f|/max(|f|,delta) = 1/delta unless delta is O(1); use 1.0 to match ER scale.
    config.delta = 1.0f;

    std::vector<uint64_t> ref_outputs = axsim::run_gpu_simulation_only(soa_and, config);
    axsim::GpuMetricsResult res = axsim::run_gpu_metrics(soa_or, ref_outputs, config);

    printf("Error rate: %.6f\n", res.error_rate);
    printf("MRED:       %.6f\n", res.mred);
    printf("MSE:        %.6f\n", res.mse);

    return 0;
}
