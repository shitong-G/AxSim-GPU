/**
 * Compare two circuits (golden vs approximate) loaded from netlist files.
 * run_gpu_metrics_pair validates and compares (golden vs approximate)
 * under the same random PI vectors and seed.
 */

#include "axsim/circuit_soa.hpp"
#include "axsim/cpu_metrics.hpp"
#include "axsim/gpu_metrics.hpp"
#include "axsim/abc_interface.hpp"
#include "axsim/pattern_file.hpp"
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <string>
#include <cmath>

namespace {

enum class BackendMode {
    Gpu,
    Cpu,
    Both,
};

bool metrics_close(float a, float b, float tol) {
    if (std::isnan(a) && std::isnan(b)) return true;
    return std::fabs(a - b) <= tol;
}

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
        "  - Default mae_normalizer=255 fits ~8b-output benchmarks; for w-bit unsigned words use\n"
        "    about (2^w-1), e.g. 22-bit product -> --mae-normalizer 4194303, or MAE%% explodes.\n"
        "  - EvoApprox Verilog: use --lsb-first (O[0] is LSB); else integer output is wrong.\n"
        "  - --patterns-file <.axpi> uses shared PI planes (see scripts/pattern_io.py).\n"
        "  - --print-timing prints EVAL_*_S and THROUGHPUT (pair eval).\n"
        "  - --timeout-seconds N : wall-clock limit (0=off). CPU checks between 64-pattern blocks;\n"
        "    GPU checks between kernel grid chunks. Exit 124 on timeout (cannot stop mid-kernel).\n"
        "  - --backend gpu|cpu|both : gpu=CUDA (default); cpu=host AIG MC like ResubALS-style sim;\n"
        "    both=run CPU+GPU and compare metrics (should match) and report speedup.\n",
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
    const char* patterns_file = nullptr;
    bool print_timing = false;
    BackendMode backend_mode = BackendMode::Gpu;

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
        } else if (opt == "--patterns-file") {
            if (argi + 1 >= argc) {
                std::fprintf(stderr, "Missing value for %s\n", opt.c_str());
                return 2;
            }
            patterns_file = argv[++argi];
        } else if (opt == "--print-timing") {
            print_timing = true;
        } else if (opt == "--timeout-seconds" || opt == "--max-wall-seconds") {
            if (argi + 1 >= argc) {
                std::fprintf(stderr, "Missing value for %s\n", opt.c_str());
                return 2;
            }
            config.max_wall_seconds = std::strtod(argv[++argi], nullptr);
        } else if (opt == "--backend") {
            if (argi + 1 >= argc) {
                std::fprintf(stderr, "Missing value for %s\n", opt.c_str());
                return 2;
            }
            const std::string b = argv[++argi];
            if (b == "gpu")
                backend_mode = BackendMode::Gpu;
            else if (b == "cpu")
                backend_mode = BackendMode::Cpu;
            else if (b == "both")
                backend_mode = BackendMode::Both;
            else {
                std::fprintf(stderr, "--backend must be gpu, cpu, or both (got %s)\n", b.c_str());
                return 2;
            }
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
    if (patterns_file) {
        if (!axsim::load_axpi010_file(patterns_file, soa1.num_pis, config.num_patterns, config.external_pi_packed)) {
            return 1;
        }
        std::printf("PI patterns: shared file %s (aligned with verilog_eval --patterns-file)\n", patterns_file);
    } else {
        std::printf("PI patterns: LCG from seed (same on GPU/CPU; not verilog_eval random.Random)\n");
    }
    const char* backend_str =
        (backend_mode == BackendMode::Gpu) ? "gpu" :
        (backend_mode == BackendMode::Cpu) ? "cpu" : "both";
    std::printf("Config: backend=%s patterns=%zu seed=%u mae_normalizer=%.6f outputs_msb_first=%d max_wall_seconds=%.0f\n\n",
        backend_str, config.num_patterns, config.seed, config.mae_normalizer, (int)config.outputs_msb_first,
        config.max_wall_seconds);

    auto print_metrics = [](const char* tag, const axsim::GpuMetricsResult& r) {
        if (tag && tag[0])
            std::printf("%s", tag);
        std::printf("Error rate: %.6f\n", r.error_rate);
        std::printf("EP%%:        %.6f\n", 100.0f * r.error_rate);
        std::printf("MAE%%:       %.6f\n", 100.0f * r.mae_norm);
        std::printf("MSE:         %.6f\n", r.mse);
    };

    if (backend_mode == BackendMode::Gpu) {
        const auto t_eval0 = std::chrono::steady_clock::now();
        axsim::GpuMetricsResult res = axsim::run_gpu_metrics_pair(soa2, soa1, config);
        const auto t_eval1 = std::chrono::steady_clock::now();
        const double eval_s = std::chrono::duration<double>(t_eval1 - t_eval0).count();

        if (!res.ok) {
            if (res.timed_out)
                std::fprintf(stderr, "GPU metric evaluation stopped (wall-clock timeout).\n");
            else
                std::fprintf(stderr, "GPU metric evaluation failed.\n");
            if (print_timing)
                std::printf("EVAL_GPU_S=%.9f\n", eval_s);
            return res.timed_out ? 124 : 1;
        }

        print_metrics("", res);

        if (print_timing) {
            std::printf("EVAL_GPU_S=%.9f\n", eval_s);
            if (eval_s > 0.0)
                std::printf("THROUGHPUT_PATTERNS_PER_S=%.6f\n", static_cast<double>(config.num_patterns) / eval_s);
            else
                std::printf("THROUGHPUT_PATTERNS_PER_S=inf\n");
        }
        return 0;
    }

    if (backend_mode == BackendMode::Cpu) {
        const auto t_eval0 = std::chrono::steady_clock::now();
        axsim::GpuMetricsResult res = axsim::run_cpu_metrics_pair(soa2, soa1, config);
        const auto t_eval1 = std::chrono::steady_clock::now();
        const double eval_s = std::chrono::duration<double>(t_eval1 - t_eval0).count();

        if (!res.ok) {
            if (res.timed_out)
                std::fprintf(stderr, "CPU metric evaluation stopped (wall-clock timeout).\n");
            else
                std::fprintf(stderr, "CPU metric evaluation failed.\n");
            if (print_timing)
                std::printf("EVAL_CPU_S=%.9f\n", eval_s);
            return res.timed_out ? 124 : 1;
        }

        print_metrics("", res);

        if (print_timing) {
            std::printf("EVAL_CPU_S=%.9f\n", eval_s);
            if (eval_s > 0.0)
                std::printf("THROUGHPUT_PATTERNS_PER_S=%.6f\n", static_cast<double>(config.num_patterns) / eval_s);
            else
                std::printf("THROUGHPUT_PATTERNS_PER_S=inf\n");
        }
        return 0;
    }

    // both
    const auto t_gpu0 = std::chrono::steady_clock::now();
    axsim::GpuMetricsResult res_gpu = axsim::run_gpu_metrics_pair(soa2, soa1, config);
    const auto t_gpu1 = std::chrono::steady_clock::now();
    const double eval_gpu_s = std::chrono::duration<double>(t_gpu1 - t_gpu0).count();

    const auto t_cpu0 = std::chrono::steady_clock::now();
    axsim::GpuMetricsResult res_cpu = axsim::run_cpu_metrics_pair(soa2, soa1, config);
    const auto t_cpu1 = std::chrono::steady_clock::now();
    const double eval_cpu_s = std::chrono::duration<double>(t_cpu1 - t_cpu0).count();

    if (!res_gpu.ok) {
        if (res_gpu.timed_out)
            std::fprintf(stderr, "GPU metric evaluation stopped (wall-clock timeout).\n");
        else
            std::fprintf(stderr, "GPU metric evaluation failed.\n");
        if (print_timing)
            std::printf("EVAL_GPU_S=%.9f\nEVAL_CPU_S=%.9f\n", eval_gpu_s, eval_cpu_s);
        return res_gpu.timed_out ? 124 : 1;
    }
    if (!res_cpu.ok) {
        if (res_cpu.timed_out)
            std::fprintf(stderr, "CPU metric evaluation stopped (wall-clock timeout).\n");
        else
            std::fprintf(stderr, "CPU metric evaluation failed.\n");
        if (print_timing)
            std::printf("EVAL_GPU_S=%.9f\nEVAL_CPU_S=%.9f\n", eval_gpu_s, eval_cpu_s);
        return res_cpu.timed_out ? 124 : 1;
    }

    const float tol = 1e-5f;
    const bool ok_er = metrics_close(res_gpu.error_rate, res_cpu.error_rate, tol);
    const bool ok_mae = metrics_close(res_gpu.mae_norm, res_cpu.mae_norm, tol);
    const float mse_tol = std::max(1e-4f, 1e-6f * std::fabs(res_gpu.mse));
    const bool ok_mse = metrics_close(res_gpu.mse, res_cpu.mse, mse_tol);

    std::printf("[GPU]\n");
    print_metrics("", res_gpu);
    std::printf("[CPU]\n");
    print_metrics("", res_cpu);
    std::printf("METRICS_CPU_GPU_MATCH=%d\n", (ok_er && ok_mae && ok_mse) ? 1 : 0);
    if (!ok_er || !ok_mae || !ok_mse) {
        std::fprintf(stderr,
            "Mismatch (GPU vs CPU): ER %.9g vs %.9g, MAE_norm %.9g vs %.9g, MSE %.9g vs %.9g\n",
            res_gpu.error_rate, res_cpu.error_rate,
            res_gpu.mae_norm, res_cpu.mae_norm,
            res_gpu.mse, res_cpu.mse);
    }

    if (print_timing) {
        std::printf("EVAL_GPU_S=%.9f\n", eval_gpu_s);
        std::printf("EVAL_CPU_S=%.9f\n", eval_cpu_s);
        if (eval_cpu_s > 0.0 && eval_gpu_s > 0.0)
            std::printf("SPEEDUP_GPU_VS_CPU=%.6f\n", eval_cpu_s / eval_gpu_s);
        if (eval_gpu_s > 0.0)
            std::printf("THROUGHPUT_GPU_PATTERNS_PER_S=%.6f\n", static_cast<double>(config.num_patterns) / eval_gpu_s);
        if (eval_cpu_s > 0.0)
            std::printf("THROUGHPUT_CPU_PATTERNS_PER_S=%.6f\n", static_cast<double>(config.num_patterns) / eval_cpu_s);
    }

    return (ok_er && ok_mae && ok_mse) ? 0 : 3;
}
