/*
 * Utilities Implementation
 * Apache License 2.0 - Copyright (c) 2025 Aetheron Research
 */

#include "pseudomode_solver_complete.h"
#include <fstream>
#include <sstream>

namespace PseudomodeFramework {
namespace Utils {

size_t estimate_memory_bytes(int system_dim, int n_modes, int n_max) {
    // State vector size
    size_t bath_dim = 1;
    for (int i = 0; i < n_modes; ++i) {
        bath_dim *= n_max;
    }
    size_t total_dim = system_dim * bath_dim;

    // State vector
    size_t state_memory = total_dim * sizeof(std::complex<double>);

    // Hamiltonian (dense matrix)
    size_t hamiltonian_memory = total_dim * total_dim * sizeof(std::complex<double>);

    // Lindblad operators (assume ~5 operators)
    size_t lindblad_memory = 5 * total_dim * total_dim * sizeof(std::complex<double>);

    // Temporary arrays
    size_t temp_memory = 4 * state_memory; // RK4 requires multiple vectors

    return state_memory + hamiltonian_memory + lindblad_memory + temp_memory;
}

int compute_adaptive_n_max(
    const std::vector<PseudomodeParams>& modes,
    double temperature_K,
    double threshold) {

    const double kB = PhysConstants::KB_EV;
    int max_n_max = 2; // Minimum

    for (const auto& mode : modes) {
        if (mode.omega_eV > 0 && temperature_K > 0) {
            double beta_omega = mode.omega_eV / (kB * temperature_K);
            if (beta_omega < 50) {
                double n_thermal = 1.0 / (std::exp(beta_omega) - 1.0);

                // Need n_max such that sum_{n=n_max}^∞ P(n) < threshold
                // For thermal distribution: P(n) = (1-e^{-βω}) (e^{-βω})^n
                // Geometric series tail: sum_{n=N}^∞ P(n) = (e^{-βω})^N

                int required_n_max = static_cast<int>(-std::log(threshold) / beta_omega) + 1;
                max_n_max = std::max(max_n_max, required_n_max);
            }
        }
    }

    return std::min(max_n_max, 15); // Cap to prevent memory explosion
}

void save_vector(const std::string& filename, const RealVector& data) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    file << std::scientific << std::setprecision(12);
    for (const auto& value : data) {
        file << value << "\n";
    }
}

void save_complex_vector(const std::string& filename, const ComplexVector& data) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    file << std::scientific << std::setprecision(12);
    for (const auto& value : data) {
        file << value.real() << "\t" << value.imag() << "\n";
    }
}

void check_dimensions(int expected, int actual, const std::string& context) {
    if (expected != actual) {
        std::ostringstream oss;
        oss << "Dimension mismatch in " << context 
            << ": expected " << expected << ", got " << actual;
        throw std::invalid_argument(oss.str());
    }
}

#ifdef USE_CUDA
void check_cuda_error(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA error at " << file << ":" << line 
            << " - " << cudaGetErrorString(error);
        throw std::runtime_error(oss.str());
    }
}
#endif

} // namespace Utils
} // namespace PseudomodeFramework
