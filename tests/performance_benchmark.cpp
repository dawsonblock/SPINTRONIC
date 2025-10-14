/*
 * Performance Benchmark Suite for Pseudomode Framework
 * Comprehensive profiling of computational hotspots
 * 
 * Copyright (c) 2025 Aetheron Research
 * Licensed under Apache License 2.0
 */

#include "pseudomode_solver.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cmath>
#include <thread>

using namespace PseudomodeSolver;
using namespace std::chrono;

// Benchmark result structure
struct BenchmarkResult {
    std::string name;
    double mean_ms;
    double std_ms;
    double min_ms;
    double max_ms;
    int iterations;
    double throughput; // items/second
};

// Timer utility
class Timer {
public:
    Timer() : start_(high_resolution_clock::now()) {}
    
    double elapsed_ms() const {
        auto end = high_resolution_clock::now();
        return duration_cast<microseconds>(end - start_).count() / 1000.0;
    }
    
    void reset() {
        start_ = high_resolution_clock::now();
    }
    
private:
    time_point<high_resolution_clock> start_;
};

// Benchmark runner
template<typename Func>
BenchmarkResult run_benchmark(const std::string& name, Func f, int iterations = 100, int n_items = 1) {
    std::vector<double> times;
    times.reserve(iterations);
    
    // Warmup
    for (int i = 0; i < 5; ++i) {
        f();
    }
    
    // Actual benchmark
    for (int i = 0; i < iterations; ++i) {
        Timer t;
        f();
        times.push_back(t.elapsed_ms());
    }
    
    // Statistics
    double sum = 0.0;
    for (double t : times) sum += t;
    double mean = sum / times.size();
    
    double sq_sum = 0.0;
    for (double t : times) {
        double diff = t - mean;
        sq_sum += diff * diff;
    }
    double std = std::sqrt(sq_sum / times.size());
    
    auto minmax = std::minmax_element(times.begin(), times.end());
    
    BenchmarkResult result;
    result.name = name;
    result.mean_ms = mean;
    result.std_ms = std;
    result.min_ms = *minmax.first;
    result.max_ms = *minmax.second;
    result.iterations = iterations;
    result.throughput = (n_items * 1000.0) / mean; // items/second
    
    return result;
}

void print_result(const BenchmarkResult& r) {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  " << std::setw(45) << std::left << r.name 
              << " | " << std::setw(8) << std::right << r.mean_ms << " ms"
              << " ± " << std::setw(6) << r.std_ms << " ms"
              << " [" << std::setw(7) << r.min_ms << " - " << std::setw(7) << r.max_ms << "]";
    if (r.throughput > 0) {
        std::cout << " | " << std::scientific << std::setprecision(2) << r.throughput << " ops/s";
    }
    std::cout << "\n";
}

//=============================================================================
// Benchmark 1: Spectral Density Computation
//=============================================================================
void benchmark_spectral_density() {
    std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  SPECTRAL DENSITY BENCHMARKS                                   ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";
    
    // Test different grid sizes
    std::vector<int> grid_sizes = {100, 1000, 10000, 100000};
    
    for (int N : grid_sizes) {
        std::vector<double> omega;
        for (int i = 0; i < N; ++i) {
            omega.push_back(i * 0.0001);
        }
        
        auto result = run_benchmark(
            "Acoustic (" + std::to_string(N) + " points)",
            [&omega]() {
                auto J = SpectralDensity2D::acoustic(omega, 1.0, 0.5, 1.5);
            },
            50,
            N
        );
        print_result(result);
    }
    
    // Compare different phonon types
    std::vector<double> omega_std;
    for (int i = 0; i < 10000; ++i) {
        omega_std.push_back(i * 0.0001);
    }
    
    std::cout << "\n";
    
    auto r1 = run_benchmark("Acoustic (10K)", 
        [&omega_std]() { auto J = SpectralDensity2D::acoustic(omega_std, 1.0, 0.5, 1.5); },
        100, 10000);
    print_result(r1);
    
    auto r2 = run_benchmark("Flexural (10K)",
        [&omega_std]() { auto J = SpectralDensity2D::flexural(omega_std, 1.0, 0.5, 1.5); },
        100, 10000);
    print_result(r2);
    
    // Note: Lorentzian may not be a standalone function
    // auto r3 = run_benchmark("Lorentzian (10K)",
    //     [&omega_std]() { auto J = SpectralDensity2D::lorentzian(omega_std, 0.05, 0.01, 0.001); },
    //     100, 10000);
    // print_result(r3);
    
    // Material database
    std::cout << "\n";
    
    auto r4 = run_benchmark("MoS2 Material (10K)",
        [&omega_std]() { auto J = SpectralDensity2D::build_material_spectrum(omega_std, "MoS2"); },
        100, 10000);
    print_result(r4);
    
    auto r5 = run_benchmark("WSe2 Material (10K)",
        [&omega_std]() { auto J = SpectralDensity2D::build_material_spectrum(omega_std, "WSe2"); },
        100, 10000);
    print_result(r5);
}

//=============================================================================
// Benchmark 2: Quantum State Operations
//=============================================================================
void benchmark_quantum_state() {
    std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  QUANTUM STATE BENCHMARKS                                      ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";
    
    // Test different system sizes
    std::vector<std::tuple<int, int, int>> configs = {
        {2, 1, 3},   // Small: dim=8
        {2, 2, 4},   // Medium: dim=32
        {2, 3, 5},   // Large: dim=108
        {2, 4, 5}    // Very large: dim=200
    };
    
    for (const auto& [sys_dim, n_modes, n_max] : configs) {
        int total_dim = sys_dim * std::pow(n_max, n_modes);
        
        // State creation
        auto r1 = run_benchmark(
            "Create state (dim=" + std::to_string(total_dim) + ")",
            [sys_dim, n_modes, n_max]() {
                QuantumState state(sys_dim, n_modes, n_max);
            },
            100
        );
        print_result(r1);
        
        // Normalization
        QuantumState state(sys_dim, n_modes, n_max);
        state.set_initial_state("plus");
        
        auto r2 = run_benchmark(
            "Normalize (dim=" + std::to_string(total_dim) + ")",
            [&state]() {
                state.normalize();
            },
            1000
        );
        print_result(r2);
        
        // Trace computation
        auto r3 = run_benchmark(
            "Trace (dim=" + std::to_string(total_dim) + ")",
            [&state]() {
                auto t = state.trace();
            },
            1000
        );
        print_result(r3);
    }
}

//=============================================================================
// Benchmark 3: Prony Fitting
//=============================================================================
void benchmark_prony_fitting() {
    std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  PRONY FITTING BENCHMARKS                                      ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";
    
    // Test different data sizes
    std::vector<int> data_sizes = {50, 100, 500, 1000};
    
    for (int N : data_sizes) {
        std::vector<double> times;
        std::vector<Complex> C_data;
        
        for (int i = 0; i < N; ++i) {
            double t = i * 0.1;
            times.push_back(t);
            C_data.push_back(0.001 * std::exp(Complex(0.0, 0.02 * t - 0.01 * t)));
        }
        
        auto result = run_benchmark(
            "Fit (N=" + std::to_string(N) + ", K=2)",
            [&C_data, &times]() {
                auto fit = PronyFitter::fit_correlation(C_data, times, 2, 300.0);
            },
            10  // Fewer iterations for expensive operation
        );
        print_result(result);
    }
    
    // Test different K values
    std::cout << "\n";
    
    std::vector<double> times_std;
    std::vector<Complex> C_data_std;
    for (int i = 0; i < 200; ++i) {
        double t = i * 0.1;
        times_std.push_back(t);
        C_data_std.push_back(0.001 * std::exp(Complex(0.0, 0.02 * t - 0.01 * t)));
    }
    
    for (int K = 1; K <= 5; ++K) {
        auto result = run_benchmark(
            "Fit (N=200, K=" + std::to_string(K) + ")",
            [&C_data_std, &times_std, K]() {
                auto fit = PronyFitter::fit_correlation(C_data_std, times_std, K, 300.0);
            },
            10
        );
        print_result(result);
    }
}

//=============================================================================
// Benchmark 4: Lindblad Evolution
//=============================================================================
void benchmark_lindblad_evolution() {
    std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  LINDBLAD EVOLUTION BENCHMARKS                                 ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";
    
    // Test different system sizes
    std::vector<std::tuple<int, int>> configs = {
        {1, 3},  // 1 mode, n_max=3: dim=6
        {2, 3},  // 2 modes, n_max=3: dim=18
        {3, 3},  // 3 modes, n_max=3: dim=54
        {3, 4}   // 3 modes, n_max=4: dim=128
    };
    
    for (const auto& [n_modes, n_max] : configs) {
        // Setup
        System2DParams system;
        system.temperature_K = 300.0;
        
        std::vector<PseudomodeParams> modes;
        for (int k = 0; k < n_modes; ++k) {
            PseudomodeParams mode;
            mode.omega_eV = 0.02 + k * 0.01;
            mode.gamma_eV = 0.005;
            mode.g_eV = 0.001;
            mode.mode_type = "acoustic";
            modes.push_back(mode);
        }
        
        SimulationConfig config;
        config.adaptive_n_max = n_max;
        config.use_gpu = false;
        
        LindbladEvolution evolution(system, modes, config);
        QuantumState initial_state(2, n_modes, n_max);
        initial_state.set_initial_state("plus");
        
        int total_dim = 2 * std::pow(n_max, n_modes);
        
        // Single step
        std::vector<double> times_single = {0.0, 0.1};
        auto r1 = run_benchmark(
            "Single step (dim=" + std::to_string(total_dim) + ")",
            [&evolution, &initial_state, &times_single]() {
                auto result = evolution.evolve(initial_state, times_single);
            },
            20
        );
        print_result(r1);
        
        // 10 steps
        std::vector<double> times_10;
        for (int i = 0; i < 10; ++i) times_10.push_back(i * 0.1);
        
        auto r2 = run_benchmark(
            "10 steps (dim=" + std::to_string(total_dim) + ")",
            [&evolution, &initial_state, &times_10]() {
                auto result = evolution.evolve(initial_state, times_10);
            },
            10
        );
        print_result(r2);
        
        // 100 steps
        std::vector<double> times_100;
        for (int i = 0; i < 100; ++i) times_100.push_back(i * 0.1);
        
        auto r3 = run_benchmark(
            "100 steps (dim=" + std::to_string(total_dim) + ")",
            [&evolution, &initial_state, &times_100]() {
                auto result = evolution.evolve(initial_state, times_100);
            },
            5
        );
        print_result(r3);
        
        std::cout << "\n";
    }
}

//=============================================================================
// Benchmark 5: Full Workflow End-to-End
//=============================================================================
void benchmark_full_workflow() {
    std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  FULL WORKFLOW BENCHMARKS                                      ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";
    
    std::vector<std::string> materials = {"MoS2", "WSe2", "graphene"};
    
    for (const auto& material : materials) {
        SimulationConfig config;
        config.max_pseudomodes = 2;
        config.adaptive_n_max = 3;
        config.total_time_ps = 1.0;
        config.time_step_ps = 0.1;
        config.use_gpu = false;
        
        PseudomodeFramework2D framework(config);
        
        System2DParams system;
        system.temperature_K = 300.0;
        
        std::vector<double> omega;
        for (int i = 1; i < 1000; ++i) {
            omega.push_back(i * 0.0001);
        }
        
        std::vector<double> times;
        for (double t = 0.0; t <= 1.0; t += 0.1) {
            times.push_back(t);
        }
        
        auto result = run_benchmark(
            material + " Full Simulation",
            [&framework, &material, &system, &omega, &times]() {
                auto sim_result = framework.simulate_material(material, system, omega, times);
            },
            3  // Very few iterations for full workflow
        );
        print_result(result);
    }
}

//=============================================================================
// Benchmark 6: Memory Allocation Patterns
//=============================================================================
void benchmark_memory_allocation() {
    std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  MEMORY ALLOCATION BENCHMARKS                                  ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";
    
    // Large vector allocations
    auto r1 = run_benchmark(
        "Allocate 1M doubles",
        []() {
            std::vector<double> v(1000000);
        },
        100,
        1000000
    );
    print_result(r1);
    
    auto r2 = run_benchmark(
        "Allocate 1M Complex",
        []() {
            std::vector<Complex> v(1000000);
        },
        100,
        1000000
    );
    print_result(r2);
    
    // Quantum state memory
    std::vector<int> dims = {100, 1000, 10000};
    for (int dim : dims) {
        int n_max = std::pow(dim, 1.0/3.0) + 1;
        int n_modes = 3;
        
        auto r = run_benchmark(
            "QuantumState alloc (dim≈" + std::to_string(dim) + ")",
            [n_max, n_modes]() {
                QuantumState state(2, n_modes, n_max);
            },
            50
        );
        print_result(r);
    }
}

//=============================================================================
// Benchmark 7: Parallel Scaling
//=============================================================================
void benchmark_parallel_scaling() {
    std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  PARALLEL SCALING BENCHMARKS                                   ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";
    
    // Batch simulation with different thread counts
    std::vector<int> thread_counts = {1, 2, 4};
    
    for (int n_threads : thread_counts) {
        SimulationConfig config;
        config.max_pseudomodes = 2;
        config.adaptive_n_max = 2;
        config.total_time_ps = 0.5;
        config.time_step_ps = 0.1;
        config.use_gpu = false;
        
        PseudomodeFramework2D framework(config);
        
        std::vector<std::string> materials = {"MoS2", "MoS2", "MoS2", "MoS2"};
        std::vector<System2DParams> systems(4);
        for (auto& sys : systems) sys.temperature_K = 300.0;
        
        auto result = run_benchmark(
            "Batch 4 sims (" + std::to_string(n_threads) + " threads)",
            [&framework, &materials, &systems, n_threads]() {
                auto results = framework.batch_simulate(materials, systems, n_threads);
            },
            2
        );
        print_result(result);
    }
}

//=============================================================================
// Main
//=============================================================================
int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                                                                ║\n";
    std::cout << "║        PSEUDOMODE FRAMEWORK PERFORMANCE BENCHMARKS             ║\n";
    std::cout << "║                                                                ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
    
    // System info
    std::cout << "\nSystem Information:\n";
    std::cout << "  C++ Standard: " << __cplusplus << "\n";
    std::cout << "  Hardware Concurrency: " << std::thread::hardware_concurrency() << " threads\n";
    #ifdef USE_CUDA
    std::cout << "  CUDA: Enabled\n";
    #else
    std::cout << "  CUDA: Disabled\n";
    #endif
    std::cout << "\n";
    
    // Run all benchmarks
    benchmark_spectral_density();
    benchmark_quantum_state();
    benchmark_prony_fitting();
    benchmark_lindblad_evolution();
    benchmark_full_workflow();
    benchmark_memory_allocation();
    benchmark_parallel_scaling();
    
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  BENCHMARKING COMPLETE                                         ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
    
    return 0;
}
