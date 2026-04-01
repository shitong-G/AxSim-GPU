#include "axsim/abc_interface.hpp"
#include "axsim/gpu_metrics.hpp"

#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "Usage: %s <golden_netlist> <approx_netlist> [num_patterns] [mae_normalizer] [seed] [outputs_msb_first]\n",
            argv[0]);
        return 2;
    }

    const char* golden_path = argv[1];
    const char* approx_path = argv[2];

    axsim::GpuMetricsConfig cfg;
    cfg.num_patterns = (argc >= 4) ? (size_t)std::strtoull(argv[3], nullptr, 10) : (size_t)(1u << 20);
    cfg.mae_normalizer = (argc >= 5) ? std::strtof(argv[4], nullptr) : 65535.0f;
    cfg.seed = (argc >= 6) ? (unsigned int)std::strtoul(argv[5], nullptr, 10) : 42u;
    cfg.outputs_msb_first = (argc >= 7) ? (std::atoi(argv[6]) != 0) : true;

    axsim::CircuitSoA golden = axsim::soa_from_abc_file(golden_path);
    axsim::CircuitSoA approx = axsim::soa_from_abc_file(approx_path);
    if (!golden.valid() || !approx.valid()) {
        std::fprintf(stderr, "Failed to load/convert netlists.\n");
        return 1;
    }

    axsim::GpuMetricsResult r = axsim::run_gpu_metrics_pair(approx, golden, cfg);
    if (!r.ok) {
        std::fprintf(stderr, "Metric computation failed.\n");
        return 1;
    }

    std::printf("golden: %s\n", golden_path);
    std::printf("approx: %s\n", approx_path);
    std::printf("patterns: %zu seed: %u mae_normalizer: %.1f outputs_msb_first: %d\n",
        cfg.num_patterns, cfg.seed, cfg.mae_normalizer, (int)cfg.outputs_msb_first);
    std::printf("EP%%(error_rate): %.6f\n", 100.0f * r.error_rate);
    std::printf("MAE%%:            %.6f\n", 100.0f * r.mae_norm);
    std::printf("MSE:              %.6f\n", r.mse);
    return 0;
}
