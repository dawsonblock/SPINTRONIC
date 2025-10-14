/*
 * Utility Functions Implementation  
 * Apache License 2.0 - Copyright (c) 2025 Aetheron Research
 */

#include "pseudomode_solver.h"
#include <fftw3.h>
#include <chrono>
#include <cmath>
#include <algorithm>

namespace PseudomodeSolver {

namespace Utils {

void fft_correlation_to_spectrum(
    const std::vector<double>& J_omega,
    std::vector<Complex>& correlation) {

    const int N = J_omega.size();
    correlation.resize(N);

    // Allocate FFTW arrays
    fftw_complex* in = fftw_alloc_complex(N);
    fftw_complex* out = fftw_alloc_complex(N);

    // Create FFTW plan
    fftw_plan plan = fftw_plan_dft_1d(N, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);

    // Fill input array (spectral density is real)
    for (int i = 0; i < N; ++i) {
        in[i][0] = J_omega[i]; // Real part
        in[i][1] = 0.0;        // Imaginary part
    }

    // Execute FFT
    fftw_execute(plan);

    // Extract correlation function
    for (int i = 0; i < N; ++i) {
        correlation[i] = Complex(out[i][0], out[i][1]);
    }

    // Cleanup
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);
}

int compute_adaptive_n_max(
    const std::vector<PseudomodeParams>& modes,
    double temperature_K,
    double occupation_threshold) {

    const double kB = PhysicalConstants::KB_EV;
    int max_n_max = 2; // Minimum truncation

    for (const auto& mode : modes) {
        if (mode.omega_eV > 0.0 && temperature_K > 0.0) {
            double beta_omega = mode.omega_eV / (kB * temperature_K);
            double n_thermal = 1.0 / (std::exp(beta_omega) - 1.0);

            // Estimate required truncation: n_max ≈ 3 * ⟨n⟩ + 5
            int required_n_max = static_cast<int>(3.0 * n_thermal + 5.0);
            max_n_max = std::max(max_n_max, required_n_max);
        }
    }

    // Cap at reasonable maximum to prevent memory explosion
    return std::min(max_n_max, 15);
}

size_t estimate_memory_usage(
    int system_dim,
    int n_pseudomodes,
    int n_max) {

    // State vector size
    int total_dim = system_dim * std::pow(n_max, n_pseudomodes);
    size_t state_memory = total_dim * sizeof(std::complex<double>);

    // Sparse matrices (rough estimate)
    size_t sparse_memory = total_dim * total_dim * 0.01 * sizeof(std::complex<double>); // Assume 1% sparsity

    // Temporary arrays and workspace
    size_t temp_memory = 5 * state_memory;

    return state_memory + sparse_memory + temp_memory;
}

Timer::Timer(const std::string& name) : name_(name) {
    start_ = std::chrono::high_resolution_clock::now();
}

Timer::~Timer() {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_);
    std::cout << "[Timer] " << name_ << ": " << duration.count() << " ms" << std::endl;
}

} // namespace Utils

} // namespace PseudomodeSolver
