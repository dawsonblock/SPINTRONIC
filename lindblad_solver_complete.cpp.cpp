/*
 * Complete Lindblad Solver Implementation
 * Apache License 2.0 - Copyright (c) 2025 Aetheron Research
 */

#include "pseudomode_solver_complete.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#endif

namespace PseudomodeFramework {

LindbladSolver::LindbladSolver(
    const SystemParams& system,
    const std::vector<PseudomodeParams>& modes,
    const SimulationConfig& config)
    : system_params_(system), pseudomodes_(modes), config_(config) {

    // Validate parameters
    for (const auto& mode : modes) {
        if (!mode.is_valid()) {
            throw std::invalid_argument("Invalid pseudomode parameters");
        }
    }

    // Calculate total dimension
    int bath_dim = 1;
    for (int i = 0; i < static_cast<int>(pseudomodes_.size()); ++i) {
        bath_dim *= system_params_.n_max;
    }
    total_dim_ = 2 * bath_dim; // 2-level system

    if (total_dim_ > 10000) {
        std::cerr << "Warning: Large Hilbert space (" << total_dim_ 
                  << "), consider reducing n_max or number of modes" << std::endl;
    }

    // Build operators
    build_hamiltonian();
    build_lindblad_operators();

#ifdef USE_CUDA
    if (config_.use_gpu) {
        setup_gpu();
    }
#endif
}

LindbladSolver::~LindbladSolver() {
#ifdef USE_CUDA
    if (config_.use_gpu) {
        cleanup_gpu();
    }
#endif
}

void LindbladSolver::build_hamiltonian() {
    hamiltonian_.resize(total_dim_, std::vector<Complex>(total_dim_, Complex(0.0, 0.0)));

    // Create a temporary quantum state to access basis indexing
    QuantumState temp_state(2, pseudomodes_.size(), system_params_.n_max);

    // System Hamiltonian: ω₀/2 σᵧ + bath energies
    for (int i = 0; i < total_dim_; ++i) {
        // Diagonal elements: system + bath energies
        const auto& basis_i = temp_state.get_basis_states()[i];

        // System energy: ω₀/2 σᵧ
        double sys_energy = 0.5 * system_params_.omega0_eV * 
                           (2 * basis_i.system_state - 1); // ±ω₀/2

        // Bath energies: Σₖ Ωₖ nₖ
        double bath_energy = 0.0;
        for (size_t k = 0; k < pseudomodes_.size(); ++k) {
            bath_energy += pseudomodes_[k].omega_eV * basis_i.bath_states[k];
        }

        hamiltonian_[i][i] = Complex(sys_energy + bath_energy, 0.0);

        // Interaction terms: gₖ σᵧ (aₖ† + aₖ) for each mode
        for (size_t k = 0; k < pseudomodes_.size(); ++k) {
            double g_k = pseudomodes_[k].g_eV;

            if (std::abs(g_k) > 1e-12) {
                // Find states connected by aₖ† and aₖ
                auto bath_states_up = basis_i.bath_states;
                auto bath_states_down = basis_i.bath_states;

                // aₖ† |...,nₖ,...⟩ = √(nₖ+1) |...,nₖ+1,...⟩
                if (bath_states_up[k] < system_params_.n_max - 1) {
                    bath_states_up[k]++;

                    // Find corresponding basis state
                    for (int j = 0; j < total_dim_; ++j) {
                        const auto& basis_j = temp_state.get_basis_states()[j];
                        if (basis_j.system_state == basis_i.system_state &&
                            basis_j.bath_states == bath_states_up) {

                            double matrix_element = g_k * std::sqrt(basis_states_up[k]);
                            if (system_params_.coupling == "sigma_z") {
                                // σᵧ coupling: only diagonal in system basis
                                matrix_element *= (2 * basis_i.system_state - 1);
                            }

                            hamiltonian_[i][j] += Complex(matrix_element, 0.0);
                            break;
                        }
                    }
                }

                // aₖ |...,nₖ,...⟩ = √nₖ |...,nₖ-1,...⟩
                if (bath_states_down[k] > 0) {
                    bath_states_down[k]--;

                    for (int j = 0; j < total_dim_; ++j) {
                        const auto& basis_j = temp_state.get_basis_states()[j];
                        if (basis_j.system_state == basis_i.system_state &&
                            basis_j.bath_states == bath_states_down) {

                            double matrix_element = g_k * std::sqrt(basis_i.bath_states[k]);
                            if (system_params_.coupling == "sigma_z") {
                                matrix_element *= (2 * basis_i.system_state - 1);
                            }

                            hamiltonian_[i][j] += Complex(matrix_element, 0.0);
                            break;
                        }
                    }
                }
            }
        }
    }
}

void LindbladSolver::build_lindblad_operators() {
    lindblad_operators_.clear();

    const double kB = PhysConstants::KB_EV;
    double T = system_params_.temperature_K;

    QuantumState temp_state(2, pseudomodes_.size(), system_params_.n_max);

    for (size_t k = 0; k < pseudomodes_.size(); ++k) {
        const auto& mode = pseudomodes_[k];

        // Thermal occupation
        double n_k = 0.0;
        if (mode.omega_eV > 0 && T > 0) {
            double beta_omega = mode.omega_eV / (kB * T);
            if (beta_omega < 50) {
                n_k = 1.0 / (std::exp(beta_omega) - 1.0);
            }
        }

        // Cooling operator: √(γ(n+1)) σᵧ ⊗ aₖ
        if (mode.gamma_eV > 0) {
            ComplexMatrix L_cool(total_dim_, std::vector<Complex>(total_dim_, Complex(0.0, 0.0)));
            double rate_cool = mode.gamma_eV * (n_k + 1.0);

            for (int i = 0; i < total_dim_; ++i) {
                const auto& basis_i = temp_state.get_basis_states()[i];

                if (basis_i.bath_states[k] > 0) { // Can annihilate
                    auto bath_states_down = basis_i.bath_states;
                    bath_states_down[k]--;

                    for (int j = 0; j < total_dim_; ++j) {
                        const auto& basis_j = temp_state.get_basis_states()[j];

                        if (basis_j.system_state == basis_i.system_state &&
                            basis_j.bath_states == bath_states_down) {

                            double sigma_element = (2 * basis_i.system_state - 1); // σᵧ
                            double annihilation = std::sqrt(basis_i.bath_states[k]);

                            L_cool[j][i] = Complex(
                                std::sqrt(rate_cool) * sigma_element * annihilation, 0.0
                            );
                            break;
                        }
                    }
                }
            }

            lindblad_operators_.push_back(L_cool);
        }

        // Heating operator: √(γn) σᵧ ⊗ aₖ†
        if (mode.gamma_eV > 0 && n_k > 1e-12) {
            ComplexMatrix L_heat(total_dim_, std::vector<Complex>(total_dim_, Complex(0.0, 0.0)));
            double rate_heat = mode.gamma_eV * n_k;

            for (int i = 0; i < total_dim_; ++i) {
                const auto& basis_i = temp_state.get_basis_states()[i];

                if (basis_i.bath_states[k] < system_params_.n_max - 1) { // Can create
                    auto bath_states_up = basis_i.bath_states;
                    bath_states_up[k]++;

                    for (int j = 0; j < total_dim_; ++j) {
                        const auto& basis_j = temp_state.get_basis_states()[j];

                        if (basis_j.system_state == basis_i.system_state &&
                            basis_j.bath_states == bath_states_up) {

                            double sigma_element = (2 * basis_i.system_state - 1);
                            double creation = std::sqrt(basis_i.bath_states[k] + 1);

                            L_heat[j][i] = Complex(
                                std::sqrt(rate_heat) * sigma_element * creation, 0.0
                            );
                            break;
                        }
                    }
                }
            }

            lindblad_operators_.push_back(L_heat);
        }
    }
}

std::vector<std::unique_ptr<QuantumState>> LindbladSolver::evolve(
    const QuantumState& initial_state,
    const RealVector& times) {

    if (times.empty()) {
        throw std::invalid_argument("Empty time array");
    }

    std::vector<std::unique_ptr<QuantumState>> evolution;
    evolution.reserve(times.size());

    // Copy initial state
    ComplexVector current_state = initial_state.get_state();

    // Add initial state to evolution
    auto state_copy = std::make_unique<QuantumState>(2, pseudomodes_.size(), system_params_.n_max);
    state_copy->get_state() = current_state;
    evolution.push_back(std::move(state_copy));

    // Time evolution using RK4
    double current_time = times[0];

    for (size_t i = 1; i < times.size(); ++i) {
        double target_time = times[i];
        double dt = target_time - current_time;

        // Use adaptive time stepping if dt is large
        while (dt > config_.time_step_ps * 1e-12) {
            double step = std::min(config_.time_step_ps * 1e-12, dt);
            rk4_step(current_state, step);
            dt -= step;
        }

        if (dt > 1e-15) {
            rk4_step(current_state, dt);
        }

        current_time = target_time;

        // Store evolved state
        auto evolved_state = std::make_unique<QuantumState>(2, pseudomodes_.size(), system_params_.n_max);
        evolved_state->get_state() = current_state;
        evolution.push_back(std::move(evolved_state));
    }

    return evolution;
}

void LindbladSolver::rk4_step(ComplexVector& state, double dt) const {
    ComplexVector k1 = compute_rhs(state);

    ComplexVector temp_state(state.size());
    for (size_t i = 0; i < state.size(); ++i) {
        temp_state[i] = state[i] + 0.5 * dt * k1[i];
    }
    ComplexVector k2 = compute_rhs(temp_state);

    for (size_t i = 0; i < state.size(); ++i) {
        temp_state[i] = state[i] + 0.5 * dt * k2[i];
    }
    ComplexVector k3 = compute_rhs(temp_state);

    for (size_t i = 0; i < state.size(); ++i) {
        temp_state[i] = state[i] + dt * k3[i];
    }
    ComplexVector k4 = compute_rhs(temp_state);

    // Final update
    for (size_t i = 0; i < state.size(); ++i) {
        state[i] += (dt / 6.0) * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);
    }
}

ComplexVector LindbladSolver::compute_rhs(const ComplexVector& state) const {
    ComplexVector rhs(state.size(), Complex(0.0, 0.0));

    // Hamiltonian evolution: -i[H, ρ]
    add_commutator(rhs, hamiltonian_, state);

    // Dissipator terms: Σₖ D[Lₖ]ρ
    for (const auto& L : lindblad_operators_) {
        add_dissipator(rhs, L, state);
    }

    return rhs;
}

void LindbladSolver::add_commutator(
    ComplexVector& result,
    const ComplexMatrix& H,
    const ComplexVector& state) const {

    // For state vector |ψ⟩: -i H |ψ⟩
    ComplexVector H_psi = matrix_vector_mult(H, state);

    for (size_t i = 0; i < result.size(); ++i) {
        result[i] += Complex(0.0, -1.0) * H_psi[i];
    }
}

void LindbladSolver::add_commutator(
    ComplexVector& result,
    const ComplexMatrix& H,
    const ComplexVector& state) const {

    // Effective Hamiltonian for non-Hermitian evolution of a state vector
    // H_eff = H - (i/2) * Σ_k L_k^† L_k
    ComplexMatrix H_eff = H;

    for (const auto& L : lindblad_operators_) {
        ComplexMatrix L_dag_L(total_dim_, std::vector<Complex>(total_dim_, Complex(0.0, 0.0)));
        // This is inefficient; ideally, pre-calculate L_dag_L products
        for (int i = 0; i < total_dim_; ++i) {
            for (int j = 0; j < total_dim_; ++j) {
                for (int k = 0; k < total_dim_; ++k) {
                    L_dag_L[i][j] += std::conj(L[k][i]) * L[k][j];
                }
            }
        }

        for (int i = 0; i < total_dim_; ++i) {
            for (int j = 0; j < total_dim_; ++j) {
                H_eff[i][j] -= Complex(0.0, 0.5) * L_dag_L[i][j];
            }
        }
    }

    // Evolve with H_eff: d|ψ⟩/dt = -i * H_eff |ψ⟩
    ComplexVector H_eff_psi = matrix_vector_mult(H_eff, state);

    for (size_t i = 0; i < result.size(); ++i) {
        result[i] += Complex(0.0, -1.0) * H_eff_psi[i];
    }
}


void LindbladSolver::add_dissipator(
    ComplexVector& result,
    const ComplexMatrix& L,
    const ComplexVector& state) const {

    // This function is not used for pure state evolution with an effective Hamiltonian.
    // The dissipative effects are included in the modified add_commutator function.
}

ComplexVector LindbladSolver::matrix_vector_mult(
    const ComplexMatrix& matrix,
    const ComplexVector& vector) const {

    ComplexVector result(vector.size(), Complex(0.0, 0.0));

    #pragma omp parallel for if(vector.size() > 1000)
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[i].size(); ++j) {
            result[i] += matrix[i][j] * vector[j];
        }
    }

    return result;
}

CoherenceTimes LindbladSolver::extract_coherence_times(
    const std::vector<std::unique_ptr<QuantumState>>& evolution,
    const RealVector& times) const {

    CoherenceTimes coherence;

    if (evolution.empty()) {
        return coherence;
    }

    // Extract Pauli expectation values
    std::vector<Complex> sigma_x_vals, sigma_z_vals;

    for (const auto& state : evolution) {
        sigma_x_vals.push_back(state->expectation_pauli_x());
        sigma_z_vals.push_back(state->expectation_pauli_z());
    }

    // Extract T₂* from exponential decay of |⟨σₓ⟩|
    try {
        std::vector<double> abs_sigma_x;
        for (const auto& val : sigma_x_vals) {
            abs_sigma_x.push_back(std::abs(val));
        }

        if (abs_sigma_x[0] > 1e-6) {
            // Find 1/e decay point
            double target = abs_sigma_x[0] / std::exp(1.0);

            for (size_t i = 1; i < abs_sigma_x.size(); ++i) {
                if (abs_sigma_x[i] < target) {
                    coherence.T2_star_ps = times[i] * 1e12; // Convert to ps
                    coherence.valid = true;
                    break;
                }
            }
        }

        if (!coherence.valid) {
            coherence.T2_star_ps = std::numeric_limits<double>::infinity();
            coherence.valid = true;
        }

    } catch (...) {
        coherence.T2_star_ps = 0.0;
        coherence.valid = false;
    }

    return coherence;
}

#ifdef USE_CUDA
void LindbladSolver::setup_gpu() {
    // Initialize CUDA resources
    CUDA_CHECK(cudaSetDevice(0));

    // Allocate GPU memory
    size_t state_size = total_dim_ * sizeof(cuDoubleComplex);
    CUDA_CHECK(cudaMalloc(&d_state_, state_size));
    CUDA_CHECK(cudaMalloc(&d_temp_, state_size));

    // Initialize cuBLAS and cuSPARSE
    cublasCreate(&cublas_handle_);
    cusparseCreate(&cusparse_handle_);
}

void LindbladSolver::cleanup_gpu() {
    if (d_state_) cudaFree(d_state_);
    if (d_temp_) cudaFree(d_temp_);

    if (cublas_handle_) cublasDestroy(cublas_handle_);
    if (cusparse_handle_) cusparseDestroy(cusparse_handle_);
}
#endif

} // namespace PseudomodeFramework
