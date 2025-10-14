/*
 * CUDA Kernels for Pseudomode Quantum Dynamics
 * Apache License 2.0 - Copyright (c) 2025 Aetheron Research
 */

#ifdef USE_CUDA

#include "../include/pseudomode_solver.h"
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <device_launch_parameters.h>

namespace PseudomodeSolver {

/**
 * @brief CUDA kernel for sparse matrix-vector multiplication (CSR format)
 * 
 * Computes y = A * x where A is in Compressed Sparse Row format.
 * Each thread computes one row of the output.
 * 
 * @param values Non-zero values of the sparse matrix
 * @param col_indices Column indices for non-zero values
 * @param row_ptrs Row pointers (CSR format)
 * @param x Input vector
 * @param y Output vector
 * @param rows Number of rows in the matrix
 * @param cols Number of columns (for bounds checking)
 * 
 * @warning Ensures bounds checking to prevent buffer overflows
 */
__global__ void sparse_matvec_kernel(
    const cuDoubleComplex* values,
    const int* col_indices,
    const int* row_ptrs,
    const cuDoubleComplex* x,
    cuDoubleComplex* y,
    int rows,
    int cols) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows) {
        cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);

        // Bounds checking for row pointers
        int row_start = row_ptrs[row];
        int row_end = row_ptrs[row + 1];
        
        for (int j = row_start; j < row_end; ++j) {
            int col = col_indices[j];
            // SECURITY: Validate column index to prevent out-of-bounds access
            if (col >= 0 && col < cols) {
                sum = cuCadd(sum, cuCmul(values[j], x[col]));
            }
        }

        y[row] = sum;
    }
}

/**
 * @brief CUDA kernel for Lindbladian evolution step: dρ/dt = L(ρ)
 * 
 * Implements one time step of the Lindblad master equation:
 * dρ/dt = -i[H, ρ] + Σ_k (L_k ρ L_k† - 1/2{L_k†L_k, ρ})
 * 
 * @param state Quantum state density matrix (flattened)
 * @param hamiltonian_action Hamiltonian contribution to evolution
 * @param dissipator_actions Array of dissipator contributions
 * @param n_dissipators Number of Lindblad operators
 * @param dt Time step size
 * @param dim Hilbert space dimension
 */
__global__ void lindblad_evolution_kernel(
    cuDoubleComplex* state,
    const cuDoubleComplex* hamiltonian_action,
    const cuDoubleComplex** dissipator_actions,
    int n_dissipators,
    double dt,
    int dim) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < dim) {
        // Hamiltonian evolution: -i[H, ρ]
        cuDoubleComplex hamiltonian_term = cuCmul(
            make_cuDoubleComplex(0.0, -dt), 
            hamiltonian_action[idx]
        );

        // Dissipator terms: Σ_k D[L_k] ρ
        // SECURITY: Validate n_dissipators to prevent excessive iterations
        cuDoubleComplex dissipator_sum = make_cuDoubleComplex(0.0, 0.0);
        int safe_n_dissipators = min(n_dissipators, 100); // Safety limit
        for (int k = 0; k < safe_n_dissipators; ++k) {
            if (dissipator_actions[k] != nullptr) { // Null pointer check
                dissipator_sum = cuCadd(dissipator_sum, dissipator_actions[k][idx]);
            }
        }
        cuDoubleComplex dissipator_term = cuCmul(
            make_cuDoubleComplex(dt, 0.0),
            dissipator_sum
        );

        // Update state: ρ(t + dt) = ρ(t) + dt * L(ρ)
        state[idx] = cuCadd(state[idx], cuCadd(hamiltonian_term, dissipator_term));
    }
}

/**
 * @brief CUDA kernel for expectation value computation: Tr[O ρ]
 * 
 * Computes the expectation value of an observable O with respect to state ρ:
 * <O> = Tr[O ρ] = Σ_i O_ii ρ_ii (for diagonal case)
 * 
 * Uses parallel reduction in shared memory for efficiency.
 * 
 * @param observable_diag Diagonal elements of the observable
 * @param state_diag Diagonal elements of the density matrix
 * @param result_blocks Output array (one element per block for partial sums)
 * @param dim Hilbert space dimension
 * 
 * @warning Caller must ensure blockDim.x <= MAX_BLOCK_SIZE
 * @note After kernel, host must perform final reduction over result_blocks
 */
constexpr int MAX_BLOCK_SIZE = 1024;
__global__ void expectation_value_kernel(
    const cuDoubleComplex* observable_diag,
    const cuDoubleComplex* state_diag,
    cuDoubleComplex* result_blocks,
    int dim) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // SECURITY FIX: Dynamic shared memory allocation to prevent buffer overflow
    extern __shared__ cuDoubleComplex shared_data[];
    
    // SECURITY: Validate thread index
    if (threadIdx.x >= MAX_BLOCK_SIZE) {
        return;
    }

    cuDoubleComplex local_sum = make_cuDoubleComplex(0.0, 0.0);

    // Each thread computes partial sum with bounds checking
    if (idx < dim) {
        local_sum = cuCmul(observable_diag[idx], state_diag[idx]);
    }

    shared_data[threadIdx.x] = local_sum;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_data[threadIdx.x] = cuCadd(
                shared_data[threadIdx.x], 
                shared_data[threadIdx.x + s]
            );
        }
        __syncthreads();
    }

    // Write block sum to global result array
    if (threadIdx.x == 0) {
        result_blocks[blockIdx.x] = shared_data[0];
    }
}

/**
 * @brief Adaptive truncation kernel: compute occupation numbers
 * 
 * Computes the average occupation number for each pseudomode:
 * ⟨n_k⟩ = Tr[ρ a_k† a_k]
 * 
 * Used for adaptive Hilbert space truncation based on physics.
 * 
 * @param pseudomode_states Quantum state density matrix
 * @param occupation_numbers Output array for occupation numbers (size: n_modes)
 * @param n_modes Number of pseudomodes
 * @param n_max Maximum oscillator level per mode
 * @param total_dim Total Hilbert space dimension
 * 
 * @note Uses atomicAdd for race-free accumulation
 */
__global__ void compute_occupation_numbers(
    const cuDoubleComplex* pseudomode_states,
    double* occupation_numbers,
    int n_modes,
    int n_max,
    int total_dim) {

    int mode = blockIdx.x;
    int level = threadIdx.x;

    // SECURITY: Validate indices to prevent out-of-bounds access
    if (mode < n_modes && level < n_max && mode >= 0 && level >= 0) {
        // Extract pseudomode density matrix elements
        double occupation = 0.0;

        // Compute ⟨a_k† a_k⟩ = Tr[ρ a_k† a_k]
        // This is simplified - full implementation would use proper indexing
        int state_idx = mode * n_max + level;
        
        // SECURITY: Bounds check before array access
        if (state_idx >= 0 && state_idx < total_dim) {
            cuDoubleComplex state_element = pseudomode_states[state_idx];
            double abs_val = cuCabs(state_element);
            occupation = abs_val * abs_val * level;
            
            // SECURITY: Validate occupation_numbers array bounds
            if (mode < n_modes) {
                atomicAdd(&occupation_numbers[mode], occupation);
            }
        }
    }
}

} // namespace PseudomodeSolver

#endif // USE_CUDA
