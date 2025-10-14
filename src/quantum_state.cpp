/*
 * Quantum State Implementation
 * Apache License 2.0 - Copyright (c) 2025 Aetheron Research
 */

#include "pseudomode_solver.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace PseudomodeSolver {

QuantumState::QuantumState(int system_dim, int n_pseudomodes, int n_max)
    : sys_dim_(system_dim), n_modes_(n_pseudomodes), n_max_(n_max) {

    auto int_pow = [](int base, int exp) -> int {
        if (base <= 0 || exp < 0) return 0;
        int result = 1;
        while (exp > 0) {
            if (exp & 1) {
                if (result > std::numeric_limits<int>::max() / base) {
                    throw std::overflow_error("Dimension overflow in quantum state size computation");
                }
                result *= base;
            }
            if (exp > 1) {
                if (base > 0 && base > std::numeric_limits<int>::max() / base) {
                    throw std::overflow_error("Dimension overflow in quantum state size computation");
                }
                base *= base;
            }
            exp >>= 1;
        }
        return result;
    };
    total_dim_ = sys_dim_ * int_pow(n_max_, n_modes_);
    if (total_dim_ <= 0) {
        throw std::overflow_error("Computed total_dim_ is invalid");
    }
    state_vector_.resize(total_dim_, Complex(0.0, 0.0));

#ifdef USE_CUDA
    on_gpu_ = false;
    d_state_vector_ = nullptr;
#endif

    // safe integer exponentiation
    auto int_pow = [](int base, int exp) {
        int result = 1;
        while (exp > 0) {
            if (exp & 1) {
                if (result > std::numeric_limits<int>::max() / base) {
                    throw std::overflow_error("Index computation overflow in excited_index");
                }
                result *= base;
            }
            exp >>= 1;
            if (exp) {
                if (base > 0 && base > std::numeric_limits<int>::max() / base) {
                    throw std::overflow_error("Index computation overflow in excited_index");
                }
                base *= base;
            }
        }
        return result;
    };
    int excited_index = int_pow(n_max_, n_modes_);
    if (excited_index < 0 || excited_index >= total_dim_) {
        throw std::out_of_range("excited_index out of bounds");
    }
    state_vector_[excited_index] = Complex(1.0, 0.0);
              << ", total_dim=" << total_dim_ << std::endl;
}

QuantumState::~QuantumState() {
#ifdef USE_CUDA
    if (d_state_vector_) {
        cudaFree(d_state_vector_);
    }
#endif
}

void QuantumState::set_initial_state(const std::string& state_type) {
    // Clear existing state
    std::fill(state_vector_.begin(), state_vector_.end(), Complex(0.0, 0.0));

    if (state_type == "ground") {
        // |0⟩ ⊗ |0,0,...,0⟩ (system ground state, all pseudomodes in vacuum)
        state_vector_[0] = Complex(1.0, 0.0);

    } else if (state_type == "excited") {
        // |1⟩ ⊗ |0,0,...,0⟩ (system excited state, all pseudomodes in vacuum)
        int excited_index = 1 * std::pow(n_max_, n_modes_); // System index 1
        state_vector_[excited_index] = Complex(1.0, 0.0);

    } else if (state_type == "plus") {
        // |+⟩ = (|0⟩ + |1⟩)/√2 ⊗ |0,0,...,0⟩
        double norm = 1.0 / std::sqrt(2.0);
        state_vector_[0] = Complex(norm, 0.0); // |0⟩ component

        int excited_index = 1 * std::pow(n_max_, n_modes_);
        state_vector_[excited_index] = Complex(norm, 0.0); // |1⟩ component

    } else if (state_type == "minus") {
        // |−⟩ = (|0⟩ - |1⟩)/√2 ⊗ |0,0,...,0⟩
        double norm = 1.0 / std::sqrt(2.0);
        state_vector_[0] = Complex(norm, 0.0); // |0⟩ component

        int excited_index = 1 * std::pow(n_max_, n_modes_);
        state_vector_[excited_index] = Complex(-norm, 0.0); // |1⟩ component

    } else {
        throw std::invalid_argument("Unknown state type: " + state_type);
    }

    normalize();
}

void QuantumState::normalize() {
    double norm_squared = 0.0;

    #pragma omp parallel for reduction(+:norm_squared)
    for (int i = 0; i < total_dim_; ++i) {
        norm_squared += std::norm(state_vector_[i]);
    }

    double norm = std::sqrt(norm_squared);

    if (norm > 1e-12) {
        #pragma omp parallel for
        for (int i = 0; i < total_dim_; ++i) {
            state_vector_[i] /= norm;
        }
    }
}

Complex QuantumState::trace() const {
    // For pure states, trace is always 1
    // For mixed states (density matrices), would need different calculation
    return Complex(1.0, 0.0);
}

double QuantumState::purity() const {
    // For pure states: Tr[ρ²] = 1
    // For mixed states: Tr[ρ²] < 1

    double purity = 0.0;

    #pragma omp parallel for reduction(+:purity)
    for (int i = 0; i < total_dim_; ++i) {
        purity += std::pow(std::abs(state_vector_[i]), 4);
    }

    return purity;
}

Complex QuantumState::expectation_value(const SparseMatrix& observable) const {
    if (observable.rows != total_dim_ || observable.cols != total_dim_) {
        throw std::invalid_argument("Observable dimension mismatch");
    }

    Complex expectation(0.0, 0.0);

    // ⟨ψ|O|ψ⟩ for pure states
    #pragma omp parallel for reduction(+:expectation)
    for (int i = 0; i < observable.rows; ++i) {
        Complex row_sum(0.0, 0.0);

        for (int j = observable.row_ptrs[i]; j < observable.row_ptrs[i + 1]; ++j) {
            int col = observable.col_indices[j];
            row_sum += observable.values[j] * state_vector_[col];
        }

        expectation += std::conj(state_vector_[i]) * row_sum;
    }

    return expectation;
}

std::unique_ptr<QuantumState> QuantumState::partial_trace_system() const {
    // Extract system density matrix by tracing over all pseudomodes

    auto system_state = std::make_unique<QuantumState>(sys_dim_, 0, 1);
    system_state->state_vector_.resize(sys_dim_ * sys_dim_); // Density matrix

    std::fill(system_state->state_vector_.begin(), 
              system_state->state_vector_.end(), Complex(0.0, 0.0));

    // Partial trace: ρ_S = Tr_B[|ψ⟩⟨ψ|]
    const int n_bath_states = std::pow(n_max_, n_modes_);

    for (int i = 0; i < sys_dim_; ++i) {
        for (int j = 0; j < sys_dim_; ++j) {
            Complex matrix_element(0.0, 0.0);

            // Sum over all bath configurations
            for (int bath_config = 0; bath_config < n_bath_states; ++bath_config) {
                int full_index_i = i * n_bath_states + bath_config;
                int full_index_j = j * n_bath_states + bath_config;

                matrix_element += std::conj(state_vector_[full_index_i]) * 
                                state_vector_[full_index_j];
            }

            int dm_index = i * sys_dim_ + j;
            system_state->state_vector_[dm_index] = matrix_element;
        }
    }

    return system_state;
}

#ifdef USE_CUDA
void QuantumState::copy_to_gpu() {
    if (on_gpu_) return;

    size_t size_bytes = total_dim_ * sizeof(cuDoubleComplex);

    cudaError_t err = cudaMalloc(&d_state_vector_, size_bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA malloc failed: " + std::string(cudaGetErrorString(err)));
    }

    err = cudaMemcpy(d_state_vector_, state_vector_.data(), size_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_state_vector_);
        throw std::runtime_error("CUDA memcpy failed: " + std::string(cudaGetErrorString(err)));
    }

    on_gpu_ = true;
}

void QuantumState::copy_from_gpu() {
    if (!on_gpu_) return;

    size_t size_bytes = total_dim_ * sizeof(cuDoubleComplex);

    cudaError_t err = cudaMemcpy(state_vector_.data(), d_state_vector_, 
                               size_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA memcpy failed: " + std::string(cudaGetErrorString(err)));
    }
}
#endif

} // namespace PseudomodeSolver
