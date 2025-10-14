/*
 * Complete 2D Pseudomode Framework - Production C++/CUDA Implementation
 * Apache License 2.0 - Copyright (c) 2025 Aetheron Research
 * 
 * Based on validated Python implementation, this provides 10-100x speedup
 * while maintaining numerical accuracy and physical correctness.
 */

#ifndef PSEUDOMODE_SOLVER_H
#define PSEUDOMODE_SOLVER_H

#include <complex>
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <chrono>

// Math libraries
#include <cmath>
#include <algorithm>
#include <numeric>

// External dependencies
#ifdef USE_EIGEN
#include <Eigen/Dense>
#include <Eigen/Sparse>
#endif

#ifdef USE_FFTW
#include <fftw3.h>
#endif

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cusparse.h>
#include <cublas_v2.h>
#endif

// OpenMP for CPU parallelization
#ifdef USE_OPENMP
#include <omp.h>
#endif

// Physical constants
namespace PhysConstants {
    constexpr double HBAR_EVS = 6.582119569e-16;  // eV⋅s
    constexpr double KB_EV = 8.617333262e-5;      // eV/K
    constexpr double PI = 3.14159265358979323846;
}

namespace PseudomodeFramework {

// Forward declarations
class MaterialDatabase;
class PronyFitter;
class LindbladSolver;
class QuantumState;

// Type definitions
using Complex = std::complex<double>;
using RealVector = std::vector<double>;
using ComplexVector = std::vector<Complex>;
using RealMatrix = std::vector<std::vector<double>>;
using ComplexMatrix = std::vector<std::vector<Complex>>;

// Configuration structures
struct PseudomodeParams {
    double omega_eV;     // Frequency (eV)
    double gamma_eV;     // Decay rate (eV)
    double g_eV;         // Coupling strength (eV)
    std::string type;    // "acoustic", "flexural", "optical"
    double n_thermal;    // Thermal occupation

    bool is_valid() const {
        return gamma_eV > 0.0 && g_eV >= 0.0 && std::isfinite(omega_eV);
    }
};

struct SystemParams {
    double omega0_eV = 1.6;      // System frequency
    double temperature_K = 300.0; // Temperature
    int n_max = 3;               // Oscillator truncation
    std::string coupling = "sigma_z"; // Coupling operator
};

struct SimulationConfig {
    int max_modes = 4;           // Maximum pseudomodes
    double time_step_ps = 0.01;  // Integration step
    double total_time_ps = 50.0; // Simulation time
    double tolerance = 1e-8;     // Convergence tolerance
    bool use_gpu = false;        // Enable CUDA
    int n_threads = 0;           // OpenMP threads (0=auto)
};

// Results structure
struct CoherenceTimes {
    double T1_ps = 0.0;
    double T2_star_ps = 0.0;
    double T2_echo_ps = 0.0;
    bool valid = false;
};

struct SimulationResult {
    std::vector<PseudomodeParams> modes;
    CoherenceTimes coherence_times;
    double computation_time_s = 0.0;
    double fit_rmse = 0.0;
    double fit_bic = 0.0;
    std::string status = "initialized";
    bool success = false;
};

// Material database class
class MaterialDatabase {
public:
    static RealVector build_spectral_density(
        const RealVector& omega_grid,
        const std::string& material
    );

    static RealVector acoustic_2d(
        const RealVector& omega,
        double alpha = 0.01,
        double omega_c = 0.04,
        double q = 1.5
    );

    static RealVector flexural_2d(
        const RealVector& omega,
        double alpha_f = 0.005,
        double omega_f = 0.02,
        double s_f = 0.3,
        double q = 2.0
    );

    static RealVector optical_peak(
        const RealVector& omega,
        double Omega_j = 0.048,
        double lambda_j = 0.002,
        double Gamma_j = 0.001
    );

private:
    static std::unordered_map<std::string, std::unordered_map<std::string, double>> 
        material_params_;
};

// Prony fitting class
class PronyFitter {
public:
    struct FitResult {
        std::vector<PseudomodeParams> modes;
        double rmse = 0.0;
        double bic = 0.0;
        bool converged = false;
        std::string message;
    };

    PronyFitter(int max_modes = 4, double regularization = 1e-6);

    // Convert spectral density to correlation function
    ComplexVector spectrum_to_correlation(
        const RealVector& J_omega,
        const RealVector& omega_grid,
        const RealVector& t_grid,
        double temperature_K
    );

    // Fit correlation function to pseudomodes
    FitResult fit_correlation(
        const ComplexVector& C_data,
        const RealVector& t_grid,
        double temperature_K
    );

private:
    int max_modes_;
    double regularization_;

    FitResult fit_single_K(
        const ComplexVector& C_data,
        const RealVector& t_grid,
        int K,
        double temperature_K
    );

    std::vector<Complex> companion_matrix_roots(
        const ComplexVector& coefficients
    );

    void apply_physical_constraints(
        std::vector<PseudomodeParams>& modes,
        double temperature_K
    );

    double compute_bic(const FitResult& result, int n_data);
};

// Quantum state class
class QuantumState {
public:
    QuantumState(int system_dim, int n_modes, int n_max);
    ~QuantumState();

    // State manipulation
    void set_initial_state(const std::string& state_type);
    void normalize();
    Complex trace() const;
    double purity() const;

    // Access state data
    const ComplexVector& get_state() const { return state_vector_; }
    ComplexVector& get_state() { return state_vector_; }

    // Partial trace to system
    std::unique_ptr<QuantumState> partial_trace_system() const;

    // Expectation values
    Complex expectation_pauli_x() const;
    Complex expectation_pauli_z() const;

private:
    int sys_dim_, n_modes_, n_max_, total_dim_;
    ComplexVector state_vector_;

    struct BasisState {
        int system_state;
        std::vector<int> bath_states;
    };

    std::vector<BasisState> basis_states_;
    std::unordered_map<std::string, int> state_index_map_;

    void build_basis();
    std::string encode_state(int sys, const std::vector<int>& bath) const;
};

// Lindblad master equation solver
class LindbladSolver {
public:
    LindbladSolver(
        const SystemParams& system,
        const std::vector<PseudomodeParams>& modes,
        const SimulationConfig& config
    );

    ~LindbladSolver();

    // Time evolution
    std::vector<std::unique_ptr<QuantumState>> evolve(
        const QuantumState& initial_state,
        const RealVector& times
    );

    // Extract coherence times from evolution
    CoherenceTimes extract_coherence_times(
        const std::vector<std::unique_ptr<QuantumState>>& evolution,
        const RealVector& times
    ) const;

private:
    SystemParams system_params_;
    std::vector<PseudomodeParams> pseudomodes_;
    SimulationConfig config_;

    int total_dim_;
    ComplexMatrix hamiltonian_;
    std::vector<ComplexMatrix> lindblad_operators_;

    void build_hamiltonian();
    void build_lindblad_operators();

    // RK4 integration step
    ComplexVector compute_rhs(const ComplexVector& state) const;
    void rk4_step(ComplexVector& state, double dt) const;

    // Matrix operations
    ComplexVector matrix_vector_mult(
        const ComplexMatrix& matrix,
        const ComplexVector& vector
    ) const;

    void add_commutator(
        ComplexVector& result,
        const ComplexMatrix& H,
        const ComplexVector& state
    ) const;

    void add_dissipator(
        ComplexVector& result,
        const ComplexMatrix& L,
        const ComplexVector& state
    ) const;

#ifdef USE_CUDA
    // GPU acceleration
    void setup_gpu();
    void cleanup_gpu();
    void evolve_gpu_step(ComplexVector& state, double dt);

    cuDoubleComplex* d_state_;
    cuDoubleComplex* d_temp_;
    cusparseHandle_t cusparse_handle_;
    cublasHandle_t cublas_handle_;
#endif
};

// High-level framework interface
class PseudomodeFramework {
public:
    PseudomodeFramework(const SimulationConfig& config = SimulationConfig{});
    ~PseudomodeFramework();

    // Main simulation interface
    SimulationResult simulate_material(
        const std::string& material,
        const SystemParams& system_params,
        const RealVector& omega_grid = {},
        const RealVector& time_grid = {}
    );

    // Batch processing
    std::vector<SimulationResult> batch_simulate(
        const std::vector<std::string>& materials,
        const std::vector<SystemParams>& systems
    );

    // Export results
    void export_results(
        const SimulationResult& result,
        const std::string& filename,
        const std::string& format = "json"
    ) const;

    // Configuration
    void set_config(const SimulationConfig& config) { config_ = config; }
    const SimulationConfig& get_config() const { return config_; }

private:
    SimulationConfig config_;
    std::unique_ptr<MaterialDatabase> materials_;
    std::unique_ptr<PronyFitter> fitter_;

    // Default grids
    RealVector create_default_omega_grid() const;
    RealVector create_default_time_grid() const;

    // Timing utilities
    class Timer {
    public:
        Timer(const std::string& name);
        ~Timer();
    private:
        std::string name_;
        std::chrono::high_resolution_clock::time_point start_;
    };
};

// Utility functions
namespace Utils {
    // Memory estimation
    size_t estimate_memory_bytes(int system_dim, int n_modes, int n_max);

    // Adaptive truncation
    int compute_adaptive_n_max(
        const std::vector<PseudomodeParams>& modes,
        double temperature_K,
        double threshold = 0.01
    );

    // File I/O
    void save_vector(const std::string& filename, const RealVector& data);
    void save_complex_vector(const std::string& filename, const ComplexVector& data);

    // Error handling
    void check_cuda_error(cudaError_t error, const char* file, int line);
    void check_dimensions(int expected, int actual, const std::string& context);
}

// Macro for CUDA error checking
#ifdef USE_CUDA
#define CUDA_CHECK(call) Utils::check_cuda_error((call), __FILE__, __LINE__)
#else
#define CUDA_CHECK(call) do {} while(0)
#endif

} // namespace PseudomodeFramework

#endif // PSEUDOMODE_SOLVER_H
