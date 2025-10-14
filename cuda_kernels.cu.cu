/*
 * CUDA Kernels for Pseudomode Quantum Dynamics
 * Apache License 2.0 - Copyright (c) 2025 Aetheron Research
 */

#ifdef USE_CUDA

#include "pseudomode_solver.h"
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <device_launch_parameters.h>

namespace PseudomodeSolver {

// CUDA kernel for sparse matrix-vector multiplication
__global__ void sparse_matvec_kernel(
    const cuDoubleComplex* values,
    const int* col_indices,
    const int* row_ptrs,
    const cuDoubleComplex* x,
    cuDoubleComplex* y,
    int rows) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows) {
        cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);

        for (int j = row_ptrs[row]; j < row_ptrs[row + 1]; ++j) {
            int col = col_indices[j];
            sum = cuCadd(sum, cuCmul(values[j], x[col]));
        }

        y[row] = sum;
    }
}

// CUDA kernel for Lindbladian evolution step: dρ/dt = L(ρ)
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
        cuDoubleComplex dissipator_sum = make_cuDoubleComplex(0.0, 0.0);
        for (int k = 0; k < n_dissipators; ++k) {
            dissipator_sum = cuCadd(dissipator_sum, dissipator_actions[k][idx]);
        }
        cuDoubleComplex dissipator_term = cuCmul(
            make_cuDoubleComplex(dt, 0.0),
            dissipator_sum
        );

        // Update state: ρ(t + dt) = ρ(t) + dt * L(ρ)
        state[idx] = cuCadd(state[idx], cuCadd(hamiltonian_term, dissipator_term));
    }
}

// CUDA kernel for expectation value computation: Tr[O ρ]
__global__ void expectation_value_kernel(
    const cuDoubleComplex* observable_diag,
    const cuDoubleComplex* state_diag,
    cuDoubleComplex* result,
    int dim) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared memory for reduction
    __shared__ cuDoubleComplex shared_data[256];

    cuDoubleComplex local_sum = make_cuDoubleComplex(0.0, 0.0);

    // Each thread computes partial sum
    if (idx < dim) {
        local_sum = cuCmul(observable_diag[idx], state_diag[idx]);
    }

    shared_data[threadIdx.x] = local_sum;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_data[threadIdx.x] = cuCadd(
                shared_data[threadIdx.x], 
                shared_data[threadIdx.x + s]
            );
        }
        __syncthreads();
    }

    // Write result of this block to global memory
    // Write result of this block to global memory using separate double buffers
    if (threadIdx.x == 0) {
        // result_real and result_imag are separate double pointers passed in
        atomicAdd(result_real, shared_data[0].x);
        atomicAdd(result_imag, shared_data[0].y);
    }
}

// Adaptive truncation kernel: compute occupation numbers
__global__ void compute_occupation_numbers(
    const cuDoubleComplex* pseudomode_states,
    double* occupation_numbers,
    int n_modes,
    int n_max,
    int total_dim) {

    int mode = blockIdx.x;
    int level = threadIdx.x;

    if (mode < n_modes && level < n_max) {
        // Extract pseudomode density matrix elements
        double occupation = 0.0;

        // Compute ⟨a_k† a_k⟩ = Tr[ρ a_k† a_k]
        // This is simplified - full implementation would use proper indexing
        int state_idx = mode * n_max + level;
        if (state_idx < total_dim) {
            cuDoubleComplex state_element = pseudomode_states[state_idx];
            occupation = cuCabs(state_element) * cuCabs(state_element) * level;
        }

        atomicAdd(&occupation_numbers[mode], occupation);
    }
}

} // namespace PseudomodeSolver

#endif // USE_CUDA
