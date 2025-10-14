/*
 * Lindblad Evolution Implementation
 * Apache License 2.0 - Copyright (c) 2025 Aetheron Research
 */

#include "pseudomode_solver.h"
#include <omp.h>
#include <chrono>
#include <iostream>
#include <cmath>

namespace PseudomodeSolver {

LindbladEvolution::LindbladEvolution(
    const System2DParams& system,
    const std::vector<PseudomodeParams>& modes,
    const SimulationConfig& config)
    : system_params_(system), pseudomodes_(modes), config_(config) {

    // Validate pseudomodes
    for (const auto& mode : modes) {
        if (!mode.is_valid()) {
            throw std::invalid_argument("Invalid pseudomode parameters");
        }
    }

#ifdef USE_CUDA
    if (config_.use_gpu) {
        setup_gpu_resources();
    }
#endif

    build_hamiltonian();
    build_lindblad_operators();
}

LindbladEvolution::~LindbladEvolution() {
#ifdef USE_CUDA
    if (config_.use_gpu) {
        cleanup_gpu_resources();
    }
#endif
}

std::vector<std::unique_ptr<QuantumState>> LindbladEvolution::evolve(
    const QuantumState& initial_state,
    const std::vector<double>& times) {

    Utils::Timer timer("LindbladEvolution::evolve");

    std::vector<std::unique_ptr<QuantumState>> evolution;
    evolution.reserve(times.size());

    // Copy initial state
    auto current_state = std::make_unique<QuantumState>(initial_state);
    evolution.push_back(std::make_unique<QuantumState>(*current_state));

    double current_time = 0.0;

    for (size_t i = 1; i < times.size(); ++i) {
        double target_time = times[i];
        double remaining_time = target_time - current_time;

        // Adaptive time stepping
        while (remaining_time > 1e-12) {
            double dt = std::min(config_.time_step_ps, remaining_time);

#ifdef USE_CUDA
            if (config_.use_gpu) {
                evolve_step_gpu(current_state->get_gpu_data(), dt); // QuantumState must define get_gpu_data()
            } else {
                evolve_step_cpu(current_state->get_state_vector(), dt); // QuantumState must define get_state_vector()
            }
#else
            evolve_step_cpu(current_state->get_state_vector(), dt);
#endif

            current_time += dt;
            remaining_time -= dt;
        }

        // Store state at this time point
        evolution.push_back(std::make_unique<QuantumState>(*current_state));

        // Progress reporting
        if (i % 10 == 0) {
            std::cout << "Evolution progress: " << i << "/" << times.size() 
                      << " (" << (100.0 * i / times.size()) << "%)" << std::endl;
        }
    }

    return evolution;
}

void LindbladEvolution::evolve_step_cpu(
    ComplexVector& state,
    double dt) const {

    const int dim = state.size();
    ComplexVector k1(dim), k2(dim), k3(dim), k4(dim);
    ComplexVector temp_state(dim);

    // 4th-order Runge-Kutta integration
    // k1 = L(ρ_n)
    compute_lindbladian_action(state, k1);

    // k2 = L(ρ_n + dt/2 * k1)  
    for (int i = 0; i < dim; ++i) {
        temp_state[i] = state[i] + 0.5 * dt * k1[i];
    }
    compute_lindbladian_action(temp_state, k2);

    // k3 = L(ρ_n + dt/2 * k2)
    for (int i = 0; i < dim; ++i) {
        temp_state[i] = state[i] + 0.5 * dt * k2[i];
    }
    compute_lindbladian_action(temp_state, k3);

    // k4 = L(ρ_n + dt * k3)
    for (int i = 0; i < dim; ++i) {
        temp_state[i] = state[i] + dt * k3[i];
    }
    compute_lindbladian_action(temp_state, k4);

    // Final update: ρ_{n+1} = ρ_n + dt/6 * (k1 + 2k2 + 2k3 + k4)
    #pragma omp parallel for
    for (int i = 0; i < dim; ++i) {
        state[i] += (dt / 6.0) * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);
    }
}

void LindbladEvolution::compute_lindbladian_action(
    const ComplexVector& state,
    ComplexVector& lindblad_state) const {

    const int dim = state.size();
    std::fill(lindblad_state.begin(), lindblad_state.end(), Complex(0.0, 0.0));

    // Hamiltonian evolution: -i[H, ρ]
    ComplexVector hamiltonian_action(dim);
    sparse_matrix_vector_mult(*hamiltonian_, state, hamiltonian_action);

    #pragma omp parallel for
    for (int i = 0; i < dim; ++i) {
        lindblad_state[i] += Complex(0.0, -1.0) * hamiltonian_action[i];
    }

    // Dissipator terms: Σ_k γ_k D[L_k] ρ
    for (const auto& lindblad_op : lindblad_ops_) {
        ComplexVector dissipator_action(dim);

        // D[L] ρ = L ρ L† - 1/2 {L† L, ρ}
        sparse_matrix_vector_mult(*lindblad_op, state, dissipator_action);

        #pragma omp parallel for
        for (int i = 0; i < dim; ++i) {
            lindblad_state[i] += dissipator_action[i];
        }
    }
}

void LindbladEvolution::build_hamiltonian() {
    const int sys_dim = 2; // qubit
    const int n_modes = pseudomodes_.size();
    const int n_max = config_.adaptive_n_max;
    const int total_dim = sys_dim * std::pow(n_max, n_modes);

    hamiltonian_ = std::make_unique<SparseMatrix>(total_dim, total_dim);

    // System Hamiltonian: H_S = ω₀/2 σ_z + H_SOC
    // This is a simplified version - full implementation would construct
    // the tensor product structure systematically

    std::cout << "Building Hamiltonian for " << n_modes << " pseudomodes, "
              << "total dimension: " << total_dim << std::endl;

    // Sparse matrix construction would go here
    // For brevity, showing the structure only

    for (int i = 0; i < total_dim; ++i) {
        // Diagonal system energy
        double system_energy = 0.5 * system_params_.omega0_eV;

        // Pseudomode energies: Σ_k Ω_k a_k† a_k
        for (size_t k = 0; k < pseudomodes_.size(); ++k) {
            int occupation = get_pseudomode_occupation(i, k, n_max);
            system_energy += pseudomodes_[k].omega_eV * occupation;
        }

        // Add diagonal element
        hamiltonian_->values.push_back(Complex(system_energy, 0.0));
        hamiltonian_->col_indices.push_back(i);
        hamiltonian_->row_ptrs[i + 1] = hamiltonian_->values.size();
    }

    hamiltonian_->nnz = hamiltonian_->values.size();
}

void LindbladEvolution::build_lindblad_operators() {
    const double kB = PhysicalConstants::KB_EV;

    lindblad_ops_.clear();

    for (const auto& mode : pseudomodes_) {
        // Thermal occupation number
        double n_k = 0.0;
        if (mode.omega_eV > 0.0 && system_params_.temperature_K > 0.0) {
            double beta_omega = mode.omega_eV / (kB * system_params_.temperature_K);
            n_k = 1.0 / (std::exp(beta_omega) - 1.0);
        }

        // Cooling operator: sqrt(γ(n+1)) a_k
        if (mode.gamma_eV > 0.0) {
            double rate_cool = mode.gamma_eV * (n_k + 1.0);
            if (rate_cool > 1e-12) {
                auto cooling_op = build_annihilation_operator(mode, std::sqrt(rate_cool));
                lindblad_ops_.push_back(std::move(cooling_op));
            }
        }

        // Heating operator: sqrt(γn) a_k†
        if (mode.gamma_eV > 0.0 && n_k > 1e-12) {
            double rate_heat = mode.gamma_eV * n_k;
            auto heating_op = build_creation_operator(mode, std::sqrt(rate_heat));
            lindblad_ops_.push_back(std::move(heating_op));
        }
    }

    std::cout << "Built " << lindblad_ops_.size() << " Lindblad operators" << std::endl;
}

LindbladEvolution::CoherenceTimes LindbladEvolution::extract_coherence_times(
    const std::vector<std::unique_ptr<QuantumState>>& evolution) const {

    CoherenceTimes times;

    // Extract system observables
    std::vector<Complex> sigma_x_vals, sigma_y_vals, sigma_z_vals;
    std::vector<double> purity_vals;

    for (const auto& state : evolution) {
        auto system_state = state->partial_trace_system();

        // Pauli operators (2x2 matrices)
        SparseMatrix sigma_x(2, 2), sigma_y(2, 2), sigma_z(2, 2);
        build_pauli_operators(sigma_x, sigma_y, sigma_z);

        sigma_x_vals.push_back(system_state->expectation_value(sigma_x));
        sigma_y_vals.push_back(system_state->expectation_value(sigma_y));
        sigma_z_vals.push_back(system_state->expectation_value(sigma_z));
        purity_vals.push_back(system_state->purity());
    }

    // Extract T₁ from exponential decay of |⟨σ_z⟩|
    times.T1_ps = extract_exponential_decay_time(sigma_z_vals);

    // Extract T₂* from Gaussian decay of |⟨σ_x⟩|
    times.T2_star_ps = extract_gaussian_decay_time(sigma_x_vals);

    // T₂ echo (simplified - would need echo simulation)
    times.T2_echo_ps = 2.0 * times.T2_star_ps; // Rough estimate

    return times;
}

#ifdef USE_CUDA
void LindbladEvolution::setup_gpu_resources() {
    cudaError_t err = cudaSetDevice(config_.gpu_device_id);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        config_.use_gpu = false;
        return;
    }

    // Initialize CUDA libraries
    cusparseCreate(&cusparse_handle_);
    cublasCreate(&cublas_handle_);

    std::cout << "GPU resources initialized successfully" << std::endl;
}

void LindbladEvolution::cleanup_gpu_resources() {
    if (d_temp_vector_) {
        cudaFree(d_temp_vector_);
    }

    cusparseDestroy(cusparse_handle_);
    cublasDestroy(cublas_handle_);
}
#endif

// Helper function implementations (stubs for complete build)

int LindbladEvolution::get_pseudomode_occupation(int state_index, int mode_index, int n_max) const {
    // Extract occupation number for a given pseudomode from composite state index
    // In tensor product basis |sys⟩ ⊗ |n₁⟩ ⊗ |n₂⟩ ⊗ ... ⊗ |nₖ⟩
    // This is a simplified implementation
    // Assume system is 2-dimensional (spin-1/2 or valley states)
    int sys_dim = 2;
    int bath_index = state_index / sys_dim;
    int divisor = 1;
    for (int k = 0; k < mode_index; ++k) {
        divisor *= n_max;
    }
    return (bath_index / divisor) % n_max;
}

std::unique_ptr<SparseMatrix> LindbladEvolution::build_annihilation_operator(
    const PseudomodeParams& mode,
    double prefactor) const {
    // Build annihilation operator a_k for a specific pseudomode
    // a|n⟩ = √n |n-1⟩
    int n_max = config_.adaptive_n_max;
    auto op = std::make_unique<SparseMatrix>(n_max, n_max);
    
    for (int n = 1; n < n_max; ++n) {
        double matrix_element = prefactor * std::sqrt(static_cast<double>(n));
        op->values.push_back(Complex(matrix_element, 0.0));
        op->col_indices.push_back(n);
        op->row_ptrs[n] = op->values.size();
    }
    op->nnz = op->values.size();
    
    return op;
}

std::unique_ptr<SparseMatrix> LindbladEvolution::build_creation_operator(
    const PseudomodeParams& mode,
    double prefactor) const {
    // Build creation operator a_k† for a specific pseudomode
    // a†|n⟩ = √(n+1) |n+1⟩
    int n_max = config_.adaptive_n_max;
    auto op = std::make_unique<SparseMatrix>(n_max, n_max);
    
    for (int n = 0; n < n_max - 1; ++n) {
        double matrix_element = prefactor * std::sqrt(static_cast<double>(n + 1));
        op->values.push_back(Complex(matrix_element, 0.0));
        op->col_indices.push_back(n + 1);
        op->row_ptrs[n] = op->values.size();
    }
    op->nnz = op->values.size();
    
    return op;
}

void LindbladEvolution::build_pauli_operators(
    SparseMatrix& sigma_x,
    SparseMatrix& sigma_y,
    SparseMatrix& sigma_z) const {
    // Pauli matrices in sparse format
    
    // σ_x = |0⟩⟨1| + |1⟩⟨0|
    sigma_x.values = {Complex(1.0, 0.0), Complex(1.0, 0.0)};
    sigma_x.col_indices = {1, 0};
    sigma_x.row_ptrs = {0, 1, 2};
    sigma_x.nnz = 2;
    
    // σ_y = -i|0⟩⟨1| + i|1⟩⟨0|
    sigma_y.values = {Complex(0.0, -1.0), Complex(0.0, 1.0)};
    sigma_y.col_indices = {1, 0};
    sigma_y.row_ptrs = {0, 1, 2};
    sigma_y.nnz = 2;
    
    // σ_z = |0⟩⟨0| - |1⟩⟨1|
    sigma_z.values = {Complex(1.0, 0.0), Complex(-1.0, 0.0)};
    sigma_z.col_indices = {0, 1};
    sigma_z.row_ptrs = {0, 1, 2};
    sigma_z.nnz = 2;
}

double LindbladEvolution::extract_exponential_decay_time(
    const std::vector<Complex>& observable_vals) const {
    // Fit exponential decay: |⟨O(t)⟩| ≈ O₀ exp(-t/T₁)
    // Simple least-squares fit in log space
    
    if (observable_vals.size() < 3) {
        return 0.0;
    }
    
    std::vector<double> log_vals;
    std::vector<double> times;
    
    for (size_t i = 0; i < observable_vals.size(); ++i) {
        double mag = std::abs(observable_vals[i]);
        if (mag > 1e-10) {
            log_vals.push_back(std::log(mag));
            times.push_back(i * config_.time_step_ps);
        }
    }
    
    if (log_vals.size() < 2) {
        return 0.0;
    }
    
    // Linear regression: log|O| = a - t/T₁
    double sum_t = 0.0, sum_log = 0.0, sum_t2 = 0.0, sum_t_log = 0.0;
    int n = log_vals.size();
    
    for (int i = 0; i < n; ++i) {
        sum_t += times[i];
        sum_log += log_vals[i];
        sum_t2 += times[i] * times[i];
        sum_t_log += times[i] * log_vals[i];
    }
    
    double slope = (n * sum_t_log - sum_t * sum_log) / (n * sum_t2 - sum_t * sum_t);
    
    // slope = -1/T₁, so T₁ = -1/slope (convert fs to ps)
    return (slope < 0.0) ? -1000.0 / slope : 0.0;
}

double LindbladEvolution::extract_gaussian_decay_time(
    const std::vector<Complex>& observable_vals) const {
    // Fit Gaussian decay: |⟨O(t)⟩| ≈ O₀ exp(-(t/T₂*)²)
    // Simple least-squares fit in log space
    
    if (observable_vals.size() < 3) {
        return 0.0;
    }
    
    std::vector<double> log_vals;
    std::vector<double> times_squared;
    
    for (size_t i = 0; i < observable_vals.size(); ++i) {
        double mag = std::abs(observable_vals[i]);
        if (mag > 1e-10) {
            log_vals.push_back(std::log(mag));
            double t = i * config_.time_step_ps;
            times_squared.push_back(t * t);
        }
    }
    
    if (log_vals.size() < 2) {
        return 0.0;
    }
    
    // Linear regression: log|O| = a - t²/T₂*²
    double sum_t2 = 0.0, sum_log = 0.0, sum_t4 = 0.0, sum_t2_log = 0.0;
    int n = log_vals.size();
    
    for (int i = 0; i < n; ++i) {
        sum_t2 += times_squared[i];
        sum_log += log_vals[i];
        sum_t4 += times_squared[i] * times_squared[i];
        sum_t2_log += times_squared[i] * log_vals[i];
    }
    
    double slope = (n * sum_t2_log - sum_t2 * sum_log) / (n * sum_t4 - sum_t2 * sum_t2);
    
    // slope = -1/T₂*², so T₂* = sqrt(-1/slope) (convert fs to ps)
    return (slope < 0.0) ? 1000.0 * std::sqrt(-1.0 / slope) : 0.0;
}

void LindbladEvolution::sparse_matrix_vector_mult(
    const SparseMatrix& A,
    const ComplexVector& x,
    ComplexVector& y) const {
    // Sparse matrix-vector multiplication: y = A * x
    // CSR (Compressed Sparse Row) format
    
    const int n_rows = A.rows;
    y.resize(n_rows);
    std::fill(y.begin(), y.end(), Complex(0.0, 0.0));
    
    #pragma omp parallel for
    for (int i = 0; i < n_rows; ++i) {
        Complex sum(0.0, 0.0);
        const int row_start = A.row_ptrs[i];
        const int row_end = A.row_ptrs[i + 1];
        
        for (int idx = row_start; idx < row_end; ++idx) {
            int j = A.col_indices[idx];
            sum += A.values[idx] * x[j];
        }
        
        y[i] = sum;
    }
}

} // namespace PseudomodeSolver
