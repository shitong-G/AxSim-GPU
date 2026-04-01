/**
 * Compare two circuits (golden vs approximate) loaded from netlist files.
 * run_gpu_metrics_pair validates and compares (golden vs approximate)
 * under the same random PI vectors and seed.
 */

#include "axsim/circuit_soa.hpp"
#include "axsim/gpu_metrics.hpp"
#include "axsim/abc_interface.hpp"
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <string>

namespace {

void print_usage(const char* prog) {
    std::fprintf(stderr,
        "Usage:\n"
        "  %s <golden_netlist> <approx_netlist> [num_patterns] [seed]\n"
        "     [--patterns N] [--seed S] [--mae-normalizer X] [--lsb-first]\n"
        "\n"
        "Examples:\n"
        "  %s BACS/abs_diff/abs_diff.aig BACS/abs_diff/abs_diff_approx.aig\n"
        "  %s exact.v approx.v 4000000 42 --mae-normalizer 255\n"
        "  %s exact.blif approx.blif --patterns 2000000 --seed 7 --lsb-first\n"
        "\n"
        "Notes:\n"
        "  - File format is auto-detected by extension via ABC (supports .aig/.v/.blif/...)\n"
        "  - --lsb-first sets outputs_msb_first=false for integer reconstruction.\n",
        prog, prog, prog, prog);
}

bool starts_with_dash(const char* s) {
    return s && s[0] == '-';
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 2) {
        const std::string only_arg = argv[1];
        if (only_arg == "--help" || only_arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }
    if (argc < 3) {
        print_usage(argv[0]);
        return 2;
    }

    const char* golden_path = argv[1];
    const char* approx_path = argv[2];

    axsim::GpuMetricsConfig config;
    config.num_patterns = 256000;
    config.seed = 42;
    config.mae_normalizer = 255.0f;

    int argi = 3;
    if (argi < argc && !starts_with_dash(argv[argi])) {
        config.num_patterns = static_cast<size_t>(std::strtoull(argv[argi], nullptr, 10));
        ++argi;
    }
    if (argi < argc && !starts_with_dash(argv[argi])) {
        config.seed = static_cast<unsigned int>(std::strtoul(argv[argi], nullptr, 10));
        ++argi;
    }

    for (; argi < argc; ++argi) {
        std::string opt = argv[argi];
        if (opt == "--patterns" || opt == "-p") {
            if (argi + 1 >= argc) {
                std::fprintf(stderr, "Missing value for %s\n", opt.c_str());
                return 2;
            }
            config.num_patterns = static_cast<size_t>(std::strtoull(argv[++argi], nullptr, 10));
        } else if (opt == "--seed" || opt == "-s") {
            if (argi + 1 >= argc) {
                std::fprintf(stderr, "Missing value for %s\n", opt.c_str());
                return 2;
            }
            config.seed = static_cast<unsigned int>(std::strtoul(argv[++argi], nullptr, 10));
        } else if (opt == "--mae-normalizer" || opt == "-n") {
            if (argi + 1 >= argc) {
                std::fprintf(stderr, "Missing value for %s\n", opt.c_str());
                return 2;
            }
            config.mae_normalizer = std::strtof(argv[++argi], nullptr);
        } else if (opt == "--lsb-first") {
            config.outputs_msb_first = false;
        } else if (opt == "--msb-first") {
            config.outputs_msb_first = true;
        } else if (opt == "--help" || opt == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "Unknown option: %s\n", opt.c_str());
            print_usage(argv[0]);
            return 2;
        }
    }

    axsim::CircuitSoA soa1 = axsim::soa_from_abc_file(golden_path);
    if (!soa1.valid()) {
        std::fprintf(stderr, "SoA invalid: golden circuit (%s)\n", golden_path);
        return 1;
    }
    
    axsim::CircuitSoA soa2 = axsim::soa_from_abc_file(approx_path);
    if (!soa2.valid()) {
        std::fprintf(stderr, "SoA invalid: approximate circuit (%s)\n", approx_path);
        return 1;
    }

    std::cout << soa1.num_pis << " " << soa1.num_nodes << " " << soa1.num_ands << " " << soa1.num_outputs << std::endl;
    std::cout << soa2.num_pis << " " << soa2.num_nodes << " " << soa2.num_ands << " " << soa2.num_outputs << std::endl;

    std::printf("Golden (ref): %s\n", golden_path);
    std::printf("Approx (simulated): %s\n", approx_path);
    std::printf("Same seed -> same random PI vectors; metrics compare the two circuits.\n");
    std::printf("Config: patterns=%zu seed=%u mae_normalizer=%.6f outputs_msb_first=%d\n\n",
        config.num_patterns, config.seed, config.mae_normalizer, (int)config.outputs_msb_first);

    axsim::GpuMetricsResult res = axsim::run_gpu_metrics_pair(soa2, soa1, config);
    if (!res.ok) {
        std::fprintf(stderr, "GPU metric evaluation failed.\n");
        return 1;
    }

    std::printf("Error rate: %.6f\n", res.error_rate);
    std::printf("EP%%:        %.6f\n", 100.0f * res.error_rate);
    std::printf("MAE%%:       %.6f\n", 100.0f * res.mae_norm);
    std::printf("MSE:         %.6f\n", res.mse);

    return 0;
}
