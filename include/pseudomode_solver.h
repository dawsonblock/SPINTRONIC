/*
 * Pseudomode Quantum Dynamics Solver - C++/CUDA Implementation
 * Copyright (c) 2025 Aetheron Research
 * Licensed under Apache License 2.0
 * 
 * High-performance implementation of finite-dimensional memory embedding
 * for non-Markovian quantum dynamics in 2D materials.
 * 
 * Key improvements over Python/QuTiP version:
 * - Apache-2 license (GPL contamination removed)
 * - CUDA GPU acceleration for large Hilbert spaces
 * - Sparse matrix operations (CSR format)
 * - Adaptive truncation based on physics
 * - Memory-mapped file I/O for large datasets
 * - OpenMP parallelization for CPU fallback
 */

#ifndef PSEUDOMODE_SOLVER_H
#define PSEUDOMODE_SOLVER_H

#include <complex>
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <chrono>
#include <utility>
#include <Eigen/Dense>

// CUDA support (optional compilation)
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cusparse.h>
#include <cublas_v2.h>
#endif

// Physical constants
namespace PhysicalConstants {
    constexpr double HBAR_EVS = 6.582119569e-16;  // eV⋅s
    constexpr double KB_EV = 8.617333262e-5;      // eV/K
    constexpr double C_LIGHT = 2.998e8;           // m/s
}

namespace PseudomodeSolver {

// Forward declarations
class SpectralDensity2D;
class PseudomodeSystem;
class QuantumState;
class LindblAdEvolution;

// Complex number type
using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;
using ComplexMatrix = std::vector<std::vector<Complex>>;

// Sparse matrix structure (CSR format)
struct SparseMatrix {
    std::vector<Complex> values;
    std::vector<int> col_indices;
    std::vector<int> row_ptrs;
    int rows, cols, nnz;

    SparseMatrix(int r, int c) : rows(r), cols(c), nnz(0) {
        row_ptrs.resize(rows + 1, 0);
    }
};

// Pseudomode parameters (single mode)
struct PseudomodeParams {
    double omega_eV;      // frequency (eV)
    double gamma_eV;      // decay rate (eV) 
    double g_eV;          // coupling strength (eV)
    std::string mode_type; // acoustic, flexural, optical, etc.

    // Validation
    bool is_valid() const {
        return gamma_eV > 0.0 && g_eV >= 0.0;
    }
};

// System parameters for 2D materials
struct System2DParams {
    double omega0_eV = 1.4;      // system frequency (eV)
    double alpha_R_eV = 0.0;     // Rashba coupling (eV⋅Å)
    double beta_D_eV = 0.0;      // Dresselhaus coupling (eV⋅Å) 
    double Delta_v_eV = 0.0;     // valley splitting (eV)
    double temperature_K = 300.0; // temperature (K)
};

// Simulation parameters
struct SimulationConfig {
    int max_pseudomodes = 6;      // maximum K
    int adaptive_n_max = 5;       // oscillator truncation
    double time_step_ps = 0.01;   // integration time step
    double total_time_ps = 100.0; // simulation duration
    double convergence_tol = 1e-8; // convergence tolerance
    bool use_gpu = true;          // enable CUDA if available
    int gpu_device_id = 0;        // GPU device to use
    std::string coupling_operator = "sigma_z"; // system-bath coupling
};

// Spectral density builder for 2D materials
class SpectralDensity2D {
public:
    // Acoustic phonon spectral density: J(ω) = α ω exp(-(ω/ωc)^q)
    static std::vector<double> acoustic(
        const std::vector<double>& omega,
        double alpha,
        double omega_c,
        double q = 1.5
    );

    // Flexural (ZA) phonon spectral density: J(ω) = αf ω^sf exp(-(ω/ωf)^q) 
    static std::vector<double> flexural(
        const std::vector<double>& omega,
        double alpha_f,
        double omega_f,
        double s_f = 0.5,
        double q = 2.0
    );

    // Discrete vibronic/magnon peak: Lorentzian
    static std::vector<double> lorentzian_peak(
        const std::vector<double>& omega,
        double Omega_j,
        double lambda_j,
        double Gamma_j
    );

    // Material-specific spectral densities
    static std::vector<double> build_material_spectrum(
        const std::vector<double>& omega,
        const std::string& material,
        const std::unordered_map<std::string, double>& params = {}
    );
};

// Prony fitting for correlation function decomposition
class PronyFitter {
public:
    struct FitResult {
        std::vector<PseudomodeParams> modes;
        double rmse;
        double bic;
        bool converged;
        std::string message;
    };

    // Fit correlation function C(t) ≈ Σ ηk exp(-(γk + iΩk)t)
    static FitResult fit_correlation(
        const std::vector<Complex>& C_data,
        const std::vector<double>& t_grid,
        int max_modes,
        double temperature_K
    );

private:
    // Hankel matrix construction for linear prediction
    static std::vector<std::vector<Complex>> create_hankel_matrix(
        const std::vector<Complex>& data,
        int K
    );

    // Initial Prony fit
    static FitResult fit_prony_initial(
        const std::vector<Complex>& C_data,
        const std::vector<double>& t_grid,
        int K
    );

    // Find polynomial roots (for Prony method)
    static std::vector<Complex> find_polynomial_roots(
        const Eigen::VectorXcd& coeffs
    );

    // Constrained refinement using Levenberg-Marquardt
    static FitResult refine_parameters(
        const std::vector<PseudomodeParams>& initial_params,
        const std::vector<Complex>& C_data,
        const std::vector<double>& t_grid,
        double temperature_K
    );

    // Compute residuals and Jacobian for refinement
    static std::pair<Eigen::VectorXd, Eigen::MatrixXd> compute_residuals_and_jacobian(
        const Eigen::VectorXd& theta,
        const std::vector<Complex>& C_data,
        const std::vector<double>& t_grid,
        double temperature_K
    );

    // Add soft constraint penalties to residuals
    static void add_constraint_penalties(
        const Eigen::VectorXd& theta,
        Eigen::VectorXd& residuals,
        double temperature_K
    );

    // Compute Jacobian matrix for refinement
    static Eigen::MatrixXd compute_jacobian(
        const Eigen::VectorXd& theta,
        const std::vector<double>& t_grid,
        double temperature_K
    );

    // Project parameters onto physical constraints
    static void project_onto_constraints(
        Eigen::VectorXd& theta,
        int K
    );

    // BIC model selection
    static double compute_bic(
        const FitResult& fit,
        int n_data_points
    );
};

// Quantum state representation  
class QuantumState {
public:
    QuantumState(int system_dim, int n_pseudomodes, int n_max);
    ~QuantumState();

    // State manipulation
    void set_initial_state(const std::string& state_type);
    void normalize();
    Complex trace() const;
    double purity() const;

    // Expectation values
    Complex expectation_value(const SparseMatrix& observable) const;

    // Partial trace over pseudomodes (extract system density matrix)
    std::unique_ptr<QuantumState> partial_trace_system() const;

    // Accessors for evolution (needed by LindbladEvolution)
    ComplexVector& get_state_vector() { return state_vector_; }
    const ComplexVector& get_state_vector() const { return state_vector_; }
    int get_total_dim() const { return total_dim_; }

#ifdef USE_CUDA
    cuDoubleComplex* get_gpu_data() { return d_state_vector_; }
    const cuDoubleComplex* get_gpu_data() const { return d_state_vector_; }
#endif

private:
    int sys_dim_, n_modes_, n_max_, total_dim_;
    ComplexVector state_vector_;

#ifdef USE_CUDA
    cuDoubleComplex* d_state_vector_;
    bool on_gpu_;
    void copy_to_gpu();
    void copy_from_gpu();
#endif
};

// Lindbladian evolution engine
class LindbladEvolution {
public:
    LindbladEvolution(
        const System2DParams& system,
        const std::vector<PseudomodeParams>& modes,
        const SimulationConfig& config
    );

    ~LindbladEvolution();

    // Time evolution
    std::vector<std::unique_ptr<QuantumState>> evolve(
        const QuantumState& initial_state,
        const std::vector<double>& times
    );

    // Coherence time extraction
    struct CoherenceTimes {
        double T1_ps;
        double T2_star_ps;
        double T2_echo_ps;
    };

    CoherenceTimes extract_coherence_times(
        const std::vector<std::unique_ptr<QuantumState>>& evolution
    ) const;

private:
    System2DParams system_params_;
    std::vector<PseudomodeParams> pseudomodes_;
    SimulationConfig config_;

    // Operators
    std::unique_ptr<SparseMatrix> hamiltonian_;
    std::vector<std::unique_ptr<SparseMatrix>> lindblad_ops_;

    // GPU resources
#ifdef USE_CUDA
    cusparseHandle_t cusparse_handle_;
    cublasHandle_t cublas_handle_;
    cuDoubleComplex* d_temp_vector_;
    void setup_gpu_resources();
    void cleanup_gpu_resources();

    // CUDA kernels for Lindbladian evolution
    void evolve_step_gpu(
        cuDoubleComplex* state,
        double dt
    );
#endif

    // CPU fallback
    void evolve_step_cpu(
        ComplexVector& state,
        double dt
    ) const;

    // Hamiltonian construction
    void build_hamiltonian();
    void build_lindblad_operators();

    // Sparse matrix operations
    void sparse_matrix_vector_mult(
        const SparseMatrix& matrix,
        const ComplexVector& input,
        ComplexVector& output
    ) const;

    // Lindbladian action computation
    void compute_lindbladian_action(
        const ComplexVector& state,
        ComplexVector& lindblad_state
    ) const;

    // Helper functions for operator construction
    int get_pseudomode_occupation(int state_index, int mode_index, int n_max) const;
    
    std::unique_ptr<SparseMatrix> build_annihilation_operator(
        const PseudomodeParams& mode,
        double prefactor
    ) const;
    
    std::unique_ptr<SparseMatrix> build_creation_operator(
        const PseudomodeParams& mode,
        double prefactor
    ) const;
    
    void build_pauli_operators(
        SparseMatrix& sigma_x,
        SparseMatrix& sigma_y,
        SparseMatrix& sigma_z
    ) const;

    // Coherence time fitting
    double extract_exponential_decay_time(const std::vector<Complex>& observable_vals) const;
    double extract_gaussian_decay_time(const std::vector<Complex>& observable_vals) const;
};

// High-level interface
class PseudomodeFramework2D {
public:
    PseudomodeFramework2D(const SimulationConfig& config = SimulationConfig{});

    // Complete workflow: spectrum → pseudomodes → dynamics → coherence
    struct SimulationResult {
        std::vector<PseudomodeParams> fitted_modes;
        LindbladEvolution::CoherenceTimes coherence_times;
        std::vector<std::unique_ptr<QuantumState>> time_evolution;
        double computation_time_seconds;
        std::string status;
    };

    SimulationResult simulate_material(
        const std::string& material_name,
        const System2DParams& system_params,
        const std::vector<double>& omega_grid,
        const std::vector<double>& time_grid
    );

    // Batch processing for materials screening
    std::vector<SimulationResult> batch_simulate(
        const std::vector<std::string>& materials,
        const std::vector<System2DParams>& systems,
        int n_parallel_jobs = -1  // -1 for auto-detect CPU cores
    );

    // Export results to various formats
    void export_results(
        const SimulationResult& result,
        const std::string& filename,
        const std::string& format = "json"  // json, hdf5, csv
    );

private:
    SimulationConfig config_;

#ifdef USE_CUDA
    bool cuda_available_;
    int cuda_device_count_;
    void detect_cuda_capabilities();
#endif
};

// Utility functions
namespace Utils {
    // Fast Fourier Transform (correlation function ↔ spectral density)
    void fft_correlation_to_spectrum(
        const std::vector<Complex>& correlation,
        std::vector<Complex>& spectrum
    );
    
    void fft_spectrum_to_correlation(
        const std::vector<double>& J_omega,
        std::vector<Complex>& correlation
    );

    // Adaptive truncation based on occupation numbers
    int compute_adaptive_n_max(
        const std::vector<PseudomodeParams>& modes,
        double temperature_K,
        double occupation_threshold = 0.01
    );

    // Memory estimation
    size_t estimate_memory_usage(
        int system_dim,
        int n_pseudomodes, 
        int n_max
    );

    // Performance profiling
    class Timer {
    public:
        Timer(const std::string& name);
        ~Timer();
    private:
        std::string name_;
        std::chrono::high_resolution_clock::time_point start_;
    };
}

} // namespace PseudomodeSolver

#endif // PSEUDOMODE_SOLVER_H
