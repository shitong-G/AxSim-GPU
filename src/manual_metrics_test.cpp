#include "axsim/circuit_soa.hpp"
#include "axsim/gpu_metrics.hpp"

#include <cmath>
#include <cstdio>
#include <tuple>
#include <vector>

namespace {

axsim::CircuitSoA build_and_1bit() {
    // PIs: 0=a, 1=b; AND node id=2; output id=2.
    std::vector<std::tuple<int, int, bool, bool>> and_nodes = {
        {0, 1, false, false}
    };
    std::vector<int> out_ids = {2};
    return axsim::flatten_from_aig(2, and_nodes, out_ids);
}

axsim::CircuitSoA build_2bit_and_or() {
    // PIs: 0=a, 1=b
    // Node2 = a&b
    // Node3 = ~a & ~b
    // Outputs (MSB->LSB by index):
    //   o0: Node2                 => AND
    //   o1: ~Node3                => OR
    std::vector<std::tuple<int, int, bool, bool>> and_nodes = {
        {0, 1, false, false}, // node 2: a&b
        {0, 1, true,  true }, // node 3: ~a&~b
    };
    std::vector<int> out_ids = {2, 3};
    axsim::CircuitSoA soa = axsim::flatten_from_aig(2, and_nodes, out_ids);
    soa.is_output_complemented = {0, 1}; // second output is OR
    return soa;
}

bool nearly_equal(float a, float b, float eps = 1e-6f) {
    return std::fabs(a - b) <= eps;
}

void print_case(
    const char* name,
    const axsim::GpuMetricsResult& got,
    float exp_er,
    float exp_mae_norm,
    float exp_mse)
{
    std::printf("[%s]\n", name);
    std::printf("  got: ER=%.6f MAE_NORM=%.6f MSE=%.6f ok=%d\n",
        got.error_rate, got.mae_norm, got.mse, (int)got.ok);
    std::printf("  exp: ER=%.6f MAE_NORM=%.6f MSE=%.6f\n",
        exp_er, exp_mae_norm, exp_mse);

    const bool pass =
        got.ok &&
        nearly_equal(got.error_rate, exp_er) &&
        nearly_equal(got.mae_norm, exp_mae_norm) &&
        nearly_equal(got.mse, exp_mse);
    std::printf("  result: %s\n\n", pass ? "PASS" : "FAIL");
}

} // namespace

int main() {
    axsim::GpuMetricsConfig cfg;
    cfg.num_patterns = 1 << 20;
    cfg.seed = 42;
    cfg.mae_normalizer = 1.0f;   // makes MAE_NORM for 1-bit NAND exactly 1.0
    cfg.outputs_msb_first = true;

    const axsim::CircuitSoA golden = build_and_1bit();

    // Case 1: same circuit => all zero error metrics.
    const axsim::CircuitSoA approx_same = build_and_1bit();
    const axsim::GpuMetricsResult r_same = axsim::run_gpu_metrics_pair(approx_same, golden, cfg);
    print_case("same", r_same, 0.0f, 0.0f, 0.0f);

    // Case 2: NAND = ~(a&b), always opposite of AND => all ones metrics for 1-bit output.
    axsim::CircuitSoA approx_nand = build_and_1bit();
    approx_nand.is_output_complemented = {1};
    const axsim::GpuMetricsResult r_nand = axsim::run_gpu_metrics_pair(approx_nand, golden, cfg);
    print_case("nand_vs_and", r_nand, 1.0f, 1.0f, 1.0f);

    // Case 3: 2-bit output, flip only LSB (OR -> NOR), MSB unchanged.
    // With outputs_msb_first=true and normalizer=3:
    // every pattern differs by exactly 1 -> ER=1, MAE_NORM=1/3, MSE=1.
    axsim::GpuMetricsConfig cfg2 = cfg;
    cfg2.mae_normalizer = 3.0f;   // 2-bit full-scale = 3
    cfg2.outputs_msb_first = true;
    const axsim::CircuitSoA golden2 = build_2bit_and_or();
    axsim::CircuitSoA approx2 = build_2bit_and_or();
    approx2.is_output_complemented = {0, 0}; // second output becomes NOR
    const axsim::GpuMetricsResult r_flip_lsb =
        axsim::run_gpu_metrics_pair(approx2, golden2, cfg2);
    print_case("2bit_flip_lsb", r_flip_lsb, 1.0f, 1.0f / 3.0f, 1.0f);

    // Case 4: Same pair as case 3, but interpret output order as LSB-first.
    // Then the flipped bit is weight-2 -> |diff|=2 always.
    // ER=1, MAE_NORM=2/3, MSE=4.
    cfg2.outputs_msb_first = false;
    const axsim::GpuMetricsResult r_flip_msb_weight =
        axsim::run_gpu_metrics_pair(approx2, golden2, cfg2);
    print_case("2bit_lsb_first_weight_check", r_flip_msb_weight, 1.0f, 2.0f / 3.0f, 4.0f);

    return 0;
}
