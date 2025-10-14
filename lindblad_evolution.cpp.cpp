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

} // namespace PseudomodeSolver
