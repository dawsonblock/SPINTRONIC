/*
 * Complete Quantum State Implementation
 * Apache License 2.0 - Copyright (c) 2025 Aetheron Research
 */

#include "pseudomode_solver_complete.h"
#include <sstream>
#include <iomanip>

namespace PseudomodeFramework {

QuantumState::QuantumState(int system_dim, int n_modes, int n_max)
    : sys_dim_(system_dim), n_modes_(n_modes), n_max_(n_max) {

    if (system_dim <= 0 || n_modes < 0 || n_max <= 0) {
        throw std::invalid_argument("Invalid dimensions");
    }

    // Calculate total dimension
    int bath_dim = 1;
    for (int i = 0; i < n_modes; ++i) {
        bath_dim *= n_max;
    }
    total_dim_ = system_dim * bath_dim;

    // Initialize state vector
    state_vector_.resize(total_dim_, Complex(0.0, 0.0));

    // Build basis
    build_basis();
}

QuantumState::~QuantumState() {
    // Automatic cleanup through RAII
}

void QuantumState::build_basis() {
    basis_states_.clear();
    state_index_map_.clear();

    // Generate all basis states |s,n1,n2,...,nK⟩
    for (int s = 0; s < sys_dim_; ++s) {
        std::function<void(std::vector<int>&, int)> generate_bath_states = 
            [&](std::vector<int>& bath_config, int mode_index) {
                if (mode_index == n_modes_) {
                    // Complete bath configuration
                    BasisState state;
                    state.system_state = s;
                    state.bath_states = bath_config;

                    int index = basis_states_.size();
                    basis_states_.push_back(state);

                    std::string key = encode_state(s, bath_config);
                    state_index_map_[key] = index;
                    return;
                }

                // Recursively generate configurations for remaining modes
                for (int n = 0; n < n_max_; ++n) {
                    bath_config[mode_index] = n;
                    generate_bath_states(bath_config, mode_index + 1);
                }
            };

        std::vector<int> bath_config(n_modes_);
        generate_bath_states(bath_config, 0);
    }
}

std::string QuantumState::encode_state(int sys, const std::vector<int>& bath) const {
    std::ostringstream oss;
    oss << sys;
    for (int n : bath) {
        oss << "," << n;
    }
    return oss.str();
}

void QuantumState::set_initial_state(const std::string& state_type) {
    std::fill(state_vector_.begin(), state_vector_.end(), Complex(0.0, 0.0));

    if (state_type == "ground") {
        // |0⟩ ⊗ |0,0,...,0⟩
        std::vector<int> vacuum(n_modes_, 0);
        std::string key = encode_state(0, vacuum);
        auto it = state_index_map_.find(key);
        if (it != state_index_map_.end()) {
            state_vector_[it->second] = Complex(1.0, 0.0);
        }

    } else if (state_type == "excited") {
        // |1⟩ ⊗ |0,0,...,0⟩
        std::vector<int> vacuum(n_modes_, 0);
        std::string key = encode_state(1, vacuum);
        auto it = state_index_map_.find(key);
        if (it != state_index_map_.end()) {
            state_vector_[it->second] = Complex(1.0, 0.0);
        }

    } else if (state_type == "plus") {
        // |+⟩ = (|0⟩ + |1⟩)/√2 ⊗ |0,0,...,0⟩
        std::vector<int> vacuum(n_modes_, 0);
        double norm = 1.0 / std::sqrt(2.0);

        std::string key0 = encode_state(0, vacuum);
        std::string key1 = encode_state(1, vacuum);

        auto it0 = state_index_map_.find(key0);
        auto it1 = state_index_map_.find(key1);

        if (it0 != state_index_map_.end()) {
            state_vector_[it0->second] = Complex(norm, 0.0);
        }
        if (it1 != state_index_map_.end()) {
            state_vector_[it1->second] = Complex(norm, 0.0);
        }

    } else if (state_type == "minus") {
        // |−⟩ = (|0⟩ - |1⟩)/√2 ⊗ |0,0,...,0⟩
        std::vector<int> vacuum(n_modes_, 0);
        double norm = 1.0 / std::sqrt(2.0);

        std::string key0 = encode_state(0, vacuum);
        std::string key1 = encode_state(1, vacuum);

        auto it0 = state_index_map_.find(key0);
        auto it1 = state_index_map_.find(key1);

        if (it0 != state_index_map_.end()) {
            state_vector_[it0->second] = Complex(norm, 0.0);
        }
        if (it1 != state_index_map_.end()) {
            state_vector_[it1->second] = Complex(-norm, 0.0);
        }

    } else {
        throw std::invalid_argument("Unknown state type: " + state_type);
    }

    normalize();
}

void QuantumState::normalize() {
    double norm_sq = 0.0;
    for (const auto& amp : state_vector_) {
        norm_sq += std::norm(amp);
    }

    if (norm_sq > 1e-12) {
        double norm = std::sqrt(norm_sq);
        for (auto& amp : state_vector_) {
            amp /= norm;
        }
    }
}

Complex QuantumState::trace() const {
    // For pure states, trace is always 1
    // For mixed states, would need density matrix representation
    return Complex(1.0, 0.0);
}

double QuantumState::purity() const {
    // For pure states: Tr[ρ²] = 1
    // For this state vector: purity = |Σ |ψᵢ|⁴|
    double purity = 0.0;
    for (const auto& amp : state_vector_) {
        purity += std::pow(std::abs(amp), 4);
    }
    return purity;
}

std::unique_ptr<QuantumState> QuantumState::partial_trace_system() const {
    // Create system-only state (2×2 density matrix)
    auto system_state = std::make_unique<QuantumState>(sys_dim_, 0, 1);

    // The "state" will actually be a flattened density matrix
    system_state->state_vector_.resize(sys_dim_ * sys_dim_, Complex(0.0, 0.0));

    // Compute partial trace: ρₛ = Trᵦ[|ψ⟩⟨ψ|]
    for (int i = 0; i < sys_dim_; ++i) {
        for (int j = 0; j < sys_dim_; ++j) {
            Complex matrix_element = 0.0;

            // Sum over all bath configurations
            for (const auto& basis_state : basis_states_) {
                if (basis_state.system_state == i) {
                    // Find corresponding |j,bath⟩ state
                    std::string key_j = encode_state(j, basis_state.bath_states);
                    auto it = state_index_map_.find(key_j);

                    if (it != state_index_map_.end()) {
                        int idx_i = state_index_map_.at(encode_state(i, basis_state.bath_states));
                        int idx_j = it->second;

                        matrix_element += std::conj(state_vector_[idx_i]) * state_vector_[idx_j];
                    }
                }
            }

            system_state->state_vector_[i * sys_dim_ + j] = matrix_element;
        }
    }

    return system_state;
}

Complex QuantumState::expectation_pauli_x() const {
    auto system_dm = partial_trace_system();

    // ⟨σₓ⟩ = Tr[ρ σₓ] = ρ₀₁ + ρ₁₀
    if (system_dm->state_vector_.size() >= 4) {
        return system_dm->state_vector_[1] + system_dm->state_vector_[2]; // ρ₀₁ + ρ₁₀
    }
    return Complex(0.0, 0.0);
}

Complex QuantumState::expectation_pauli_z() const {
    auto system_dm = partial_trace_system();

    // ⟨σᵧ⟩ = Tr[ρ σᵧ] = ρ₀₀ - ρ₁₁
    if (system_dm->state_vector_.size() >= 4) {
        return system_dm->state_vector_[0] - system_dm->state_vector_[3]; // ρ₀₀ - ρ₁₁
    }
    return Complex(0.0, 0.0);
}

} // namespace PseudomodeFramework
