/*
 * L-BFGS-B Optimizer Implementation
 * Apache License 2.0 - Copyright (c) 2025 Aetheron Research
 */

#include "lbfgs_optimizer.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <iostream>
#include <iomanip>

namespace PseudomodeFramework {
namespace Optimization {

LBFGSBOptimizer::LBFGSBOptimizer(const LBFGSOptions& options) 
    : options_(options) {}

LBFGSResult LBFGSBOptimizer::minimize(
    const ObjectiveFunction& objective,
    const std::vector<double>& initial_x,
    const std::vector<double>& lower_bounds,
    const std::vector<double>& upper_bounds) {

    const int n = initial_x.size();

    // Initialize result structure
    LBFGSResult result;
    result.x = initial_x;
    result.gradient.resize(n);
    result.iterations = 0;
    result.function_evaluations = 0;
    result.converged = false;

    // Project initial point to bounds
    project_bounds(result.x, lower_bounds, upper_bounds);

    // Initialize optimization state
    InternalState state;
    std::vector<double> x_old = result.x;
    std::vector<double> gradient_old(n);
    std::vector<double> search_direction(n);

    // Evaluate initial point
    result.f = objective(result.x, result.gradient);
    result.function_evaluations++;

    if (options_.verbose) {
        std::cout << std::scientific << std::setprecision(6);
        std::cout << "L-BFGS-B Optimization Started" << std::endl;
        std::cout << "Initial f = " << result.f << ", ||g|| = " 
                  << compute_gradient_norm(result.gradient) << std::endl;
    }

    // Main optimization loop
    for (int iteration = 0; iteration < options_.max_evaluations; ++iteration) {
        result.iterations = iteration + 1;

        // Check gradient-based convergence
        double grad_norm = compute_gradient_norm(result.gradient);
        if (grad_norm < options_.gradient_tolerance) {
            result.converged = true;
            result.message = "Converged: gradient tolerance satisfied";
            break;
        }

        // Compute search direction using L-BFGS two-loop recursion
        compute_search_direction(result.gradient, state, search_direction);

        // Ensure descent direction
        double directional_derivative = std::inner_product(
            result.gradient.begin(), result.gradient.end(),
            search_direction.begin(), 0.0
        );

        if (directional_derivative >= 0) {
            // Reset to steepest descent
            for (int i = 0; i < n; ++i) {
                search_direction[i] = -result.gradient[i];
            }
        }

        // Line search with box constraints
        double step_size = line_search(
            objective, result.x, result.gradient, search_direction,
            lower_bounds, upper_bounds, options_.initial_step
        );

        if (step_size < 1e-12) {
            result.message = "Line search failed: step size too small";
            break;
        }

        // Store old values
        x_old = result.x;
        gradient_old = result.gradient;

        // Take step and project to bounds
        for (int i = 0; i < n; ++i) {
            result.x[i] += step_size * search_direction[i];
        }
        project_bounds(result.x, lower_bounds, upper_bounds);

        // Evaluate new point
        result.f = objective(result.x, result.gradient);
        result.function_evaluations++;

        // Check parameter-based convergence
        if (check_convergence(x_old, result.x, result.gradient)) {
            result.converged = true;
            result.message = "Converged: parameter tolerance satisfied";
            break;
        }

        // Update L-BFGS memory
        update_memory(x_old, result.x, gradient_old, result.gradient, state);

        // Progress output
        if (options_.verbose && (iteration % 10 == 0 || iteration < 10)) {
            std::cout << "Iter " << std::setw(4) << iteration 
                      << ": f = " << std::setw(12) << result.f
                      << ", ||g|| = " << std::setw(10) << grad_norm
                      << ", step = " << std::setw(10) << step_size << std::endl;
        }
    }

    if (!result.converged) {
        result.message = "Maximum iterations reached without convergence";
    }

    if (options_.verbose) {
        std::cout << "L-BFGS-B Optimization Completed" << std::endl;
        std::cout << "Status: " << result.message << std::endl;
        std::cout << "Final f = " << result.f << ", iterations = " << result.iterations
                  << ", evaluations = " << result.function_evaluations << std::endl;
    }

    return result;
}

void LBFGSBOptimizer::project_bounds(
    std::vector<double>& x,
    const std::vector<double>& lower,
    const std::vector<double>& upper) const {

    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = std::max(lower[i], std::min(upper[i], x[i]));
    }
}

double LBFGSBOptimizer::line_search(
    const ObjectiveFunction& objective,
    const std::vector<double>& x,
    const std::vector<double>& gradient,
    const std::vector<double>& direction,
    const std::vector<double>& lower_bounds,
    const std::vector<double>& upper_bounds,
    double initial_step) const {

    const double c1 = options_.c1;
    const double c2 = options_.c2;
    const double rho = 0.5;  // Step reduction factor
    const int max_line_search = 20;

    double step = initial_step;
    double f0 = 0.0;  // Will be set by first objective call
    std::vector<double> grad0(x.size());

    // Compute initial directional derivative
    double derphi0 = std::inner_product(
        gradient.begin(), gradient.end(), direction.begin(), 0.0
    );

    if (derphi0 >= 0) {
        return 0.0; // Not a descent direction
    }

    // Try steps until Armijo condition is satisfied
    for (int i = 0; i < max_line_search; ++i) {
        std::vector<double> x_new = x;

        // Take step and project
        for (size_t j = 0; j < x.size(); ++j) {
            x_new[j] += step * direction[j];
        }
        project_bounds(x_new, lower_bounds, upper_bounds);

        // Evaluate at new point
        std::vector<double> grad_new(x.size());
        double f_new = objective(x_new, grad_new);

        if (i == 0) {
            f0 = objective(x, grad0); // Get baseline value
        }

        // Check Armijo condition
        if (f_new <= f0 + c1 * step * derphi0) {
            return step;
        }

        // Reduce step size
        step *= rho;

        if (step < 1e-10) {
            break;
        }
    }

    return step; // Return last attempted step
}

void LBFGSBOptimizer::compute_search_direction(
    const std::vector<double>& gradient,
    const InternalState& state,
    std::vector<double>& direction) const {

    const int n = gradient.size();

    if (state.memory_count == 0) {
        // No history, use steepest descent
        for (int i = 0; i < n; ++i) {
            direction[i] = -gradient[i];
        }
        return;
    }

    // L-BFGS two-loop recursion
    std::vector<double> q = gradient;
    std::vector<double> alpha(state.memory_count);

    // First loop (backward)
    for (int i = state.memory_count - 1; i >= 0; --i) {
        int idx = (state.memory_start + i) % options_.memory_size;

        alpha[i] = state.rho_values[idx] * 
                   std::inner_product(state.s_vectors[idx].begin(), 
                                    state.s_vectors[idx].end(), q.begin(), 0.0);

        for (int j = 0; j < n; ++j) {
            q[j] -= alpha[i] * state.y_vectors[idx][j];
        }
    }

    // Apply initial Hessian approximation (identity)
    direction = q;
    for (double& d : direction) {
        d = -d;
    }

    // Second loop (forward)
    for (int i = 0; i < state.memory_count; ++i) {
        int idx = (state.memory_start + i) % options_.memory_size;

        double beta = state.rho_values[idx] * 
                      std::inner_product(state.y_vectors[idx].begin(),
                                       state.y_vectors[idx].end(), direction.begin(), 0.0);

        for (int j = 0; j < n; ++j) {
            direction[j] += state.s_vectors[idx][j] * (alpha[i] - beta);
        }
    }
}

void LBFGSBOptimizer::update_memory(
    const std::vector<double>& x_old,
    const std::vector<double>& x_new,
    const std::vector<double>& grad_old,
    const std::vector<double>& grad_new,
    InternalState& state) const {

    const int n = x_old.size();

    // Compute s = x_new - x_old and y = grad_new - grad_old
    std::vector<double> s(n), y(n);
    for (int i = 0; i < n; ++i) {
        s[i] = x_new[i] - x_old[i];
        y[i] = grad_new[i] - grad_old[i];
    }

    // Compute s^T y
    double sy = std::inner_product(s.begin(), s.end(), y.begin(), 0.0);

    // Skip update if curvature condition is not satisfied
    if (sy <= 1e-8) {
        return;
    }

    // Determine storage location
    int store_idx;
    if (state.memory_count < options_.memory_size) {
        // Still have room
        store_idx = state.memory_count;
        state.memory_count++;
    } else {
        // Circular buffer full, overwrite oldest
        store_idx = state.memory_start;
        state.memory_start = (state.memory_start + 1) % options_.memory_size;
    }

    // Ensure vectors are allocated
    if (state.s_vectors.size() <= store_idx) {
        state.s_vectors.resize(options_.memory_size);
        state.y_vectors.resize(options_.memory_size);
        state.rho_values.resize(options_.memory_size);
    }

    // Store vectors
    state.s_vectors[store_idx] = std::move(s);
    state.y_vectors[store_idx] = std::move(y);
    state.rho_values[store_idx] = 1.0 / sy;
}

double LBFGSBOptimizer::compute_gradient_norm(const std::vector<double>& gradient) const {
    double norm_sq = 0.0;
    for (double g : gradient) {
        norm_sq += g * g;
    }
    return std::sqrt(norm_sq);
}

bool LBFGSBOptimizer::check_convergence(
    const std::vector<double>& x_old,
    const std::vector<double>& x_new,
    const std::vector<double>& gradient) const {

    double max_change = 0.0;
    for (size_t i = 0; i < x_old.size(); ++i) {
        max_change = std::max(max_change, std::abs(x_new[i] - x_old[i]));
    }

    return max_change < options_.parameter_tolerance;
}

ObjectiveFunction LBFGSBOptimizer::finite_difference_wrapper(
    const std::function<double(const std::vector<double>&)>& f,
    double epsilon) {

    return [f, epsilon](const std::vector<double>& x, std::vector<double>& grad) -> double {
        const int n = x.size();
        grad.resize(n);

        // Evaluate at current point
        double f0 = f(x);

        // Compute gradient by finite differences
        for (int i = 0; i < n; ++i) {
            std::vector<double> x_plus = x;
            x_plus[i] += epsilon;

            double f_plus = f(x_plus);
            grad[i] = (f_plus - f0) / epsilon;
        }

        return f0;
    };
}

} // namespace Optimization
} // namespace PseudomodeFramework
