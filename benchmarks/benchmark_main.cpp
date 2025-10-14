/**
 * @file benchmark_main.cpp
 * @brief Performance benchmarks for pseudomode framework
 * 
 * Uses Google Benchmark library to profile critical operations.
 */

#include <benchmark/benchmark.h>
#include "../include/pseudomode_solver.h"
#include <random>

using namespace PseudomodeSolver;

// Benchmark spectral density computation
static void BM_SpectralDensity_Acoustic(benchmark::State& state) {
    int n_points = state.range(0);
    std::vector<double> omega(n_points);
    for (int i = 0; i < n_points; ++i) {
        omega[i] = i * 10.0 / n_points;
    }
    
    for (auto _ : state) {
        auto J = SpectralDensity2D::acoustic(omega, 1.0, 1.0, 1.5);
        benchmark::DoNotOptimize(J);
    }
    
    state.SetComplexityN(n_points);
}
BENCHMARK(BM_SpectralDensity_Acoustic)
    ->RangeMultiplier(10)
    ->Range(100, 100000)
    ->Complexity();

// Benchmark Prony fitting
static void BM_PronyFitting(benchmark::State& state) {
    int n_modes = state.range(0);
    int n_points = 1000;
    
    // Generate synthetic correlation function
    std::vector<double> t_grid(n_points);
    std::vector<Complex> C_data(n_points);
    
    for (int i = 0; i < n_points; ++i) {
        t_grid[i] = i * 0.01;
        C_data[i] = std::exp(Complex(-0.1, -1.0) * t_grid[i]);
    }
    
    for (auto _ : state) {
        auto result = PronyFitter::fit_correlation(C_data, t_grid, n_modes, 300.0);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_PronyFitting)
    ->Arg(1)
    ->Arg(2)
    ->Arg(3)
    ->Arg(5)
    ->Arg(10);

// Benchmark quantum state operations
static void BM_QuantumState_Normalization(benchmark::State& state) {
    int n_pseudomodes = state.range(0);
    int system_dim = 2;
    int n_max = 3;
    
    QuantumState qstate(system_dim, n_pseudomodes, n_max);
    qstate.set_initial_state("ground");
    
    for (auto _ : state) {
        qstate.normalize();
        benchmark::DoNotOptimize(qstate);
    }
}
BENCHMARK(BM_QuantumState_Normalization)
    ->Arg(1)
    ->Arg(2)
    ->Arg(3)
    ->Arg(5);

// Benchmark sparse matrix operations
static void BM_SparseMatrixVectorMult(benchmark::State& state) {
    int dim = state.range(0);
    
    // Create sparse matrix (tridiagonal for simplicity)
    SparseMatrix mat(dim, dim);
    mat.values.reserve(3 * dim);
    mat.col_indices.reserve(3 * dim);
    
    for (int i = 0; i < dim; ++i) {
        mat.row_ptrs[i] = mat.values.size();
        
        // Diagonal
        mat.values.push_back(Complex(2.0, 0.0));
        mat.col_indices.push_back(i);
        
        // Off-diagonals
        if (i > 0) {
            mat.values.push_back(Complex(-1.0, 0.0));
            mat.col_indices.push_back(i - 1);
        }
        if (i < dim - 1) {
            mat.values.push_back(Complex(-1.0, 0.0));
            mat.col_indices.push_back(i + 1);
        }
    }
    mat.row_ptrs[dim] = mat.values.size();
    mat.nnz = mat.values.size();
    
    ComplexVector input(dim, Complex(1.0, 0.0));
    ComplexVector output(dim);
    
    for (auto _ : state) {
        // Simulate sparse matvec
        for (int i = 0; i < dim; ++i) {
            Complex sum(0.0, 0.0);
            for (int j = mat.row_ptrs[i]; j < mat.row_ptrs[i+1]; ++j) {
                sum += mat.values[j] * input[mat.col_indices[j]];
            }
            output[i] = sum;
        }
        benchmark::DoNotOptimize(output);
    }
    
    state.SetComplexityN(dim);
}
BENCHMARK(BM_SparseMatrixVectorMult)
    ->RangeMultiplier(2)
    ->Range(8, 2048)
    ->Complexity();

// Benchmark memory allocation patterns
static void BM_MemoryAllocation(benchmark::State& state) {
    int n_pseudomodes = state.range(0);
    int system_dim = 2;
    int n_max = 5;
    
    for (auto _ : state) {
        QuantumState qstate(system_dim, n_pseudomodes, n_max);
        benchmark::DoNotOptimize(qstate);
    }
}
BENCHMARK(BM_MemoryAllocation)
    ->Arg(1)
    ->Arg(2)
    ->Arg(3)
    ->Arg(5)
    ->Arg(10);

// Benchmark FFT operations
static void BM_FFT_CorrelationToSpectrum(benchmark::State& state) {
    int n_points = state.range(0);
    
    std::vector<Complex> correlation(n_points);
    std::vector<Complex> spectrum(n_points);
    
    // Initialize with decaying correlation function
    for (int i = 0; i < n_points; ++i) {
        double t = i * 0.01;
        correlation[i] = std::exp(Complex(-0.1, -1.0) * t);
    }
    
    for (auto _ : state) {
        Utils::fft_correlation_to_spectrum(correlation, spectrum);
        benchmark::DoNotOptimize(spectrum);
    }
    
    state.SetComplexityN(n_points);
}
BENCHMARK(BM_FFT_CorrelationToSpectrum)
    ->RangeMultiplier(2)
    ->Range(128, 16384)
    ->Complexity();

// Benchmark complete simulation workflow
static void BM_CompleteSimulation(benchmark::State& state) {
    int n_pseudomodes = state.range(0);
    
    SimulationConfig config;
    config.max_pseudomodes = n_pseudomodes;
    config.adaptive_n_max = 3;
    config.total_time_ps = 10.0;
    config.time_step_ps = 0.1;
    config.use_gpu = false; // CPU benchmark
    
    PseudomodeFramework2D framework(config);
    
    System2DParams system;
    system.omega0_eV = 1.4;
    system.temperature_K = 300.0;
    
    std::vector<double> omega_grid(100);
    for (size_t i = 0; i < omega_grid.size(); ++i) {
        omega_grid[i] = i * 0.05;
    }
    
    std::vector<double> time_grid(100);
    for (size_t i = 0; i < time_grid.size(); ++i) {
        time_grid[i] = i * 0.1;
    }
    
    for (auto _ : state) {
        auto result = framework.simulate_material(
            "graphene", system, omega_grid, time_grid
        );
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_CompleteSimulation)
    ->Arg(1)
    ->Arg(2)
    ->Arg(3)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
