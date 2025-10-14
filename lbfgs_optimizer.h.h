/*
 * L-BFGS-B Optimizer - Fast Gradient-Based Parameter Fitting
 * Apache License 2.0 - Copyright (c) 2025 Aetheron Research
 */

#pragma once
#include <vector>
#include <functional>

namespace PseudomodeFramework {
namespace Optimization {

struct LBFGSOptions {
    int max_evaluations = 1000;    // Maximum function evaluations
    int memory_size = 10;          // L-BFGS memory parameter (m)
    double gradient_tolerance = 1e-4;  // Gradient norm stopping criterion
    double parameter_tolerance = 1e-3; // Parameter change stopping criterion
    double initial_step = 0.1;     // Initial step size
    double c1 = 1e-4;              // Armijo condition parameter
    double c2 = 0.9;               // Wolfe condition parameter
    bool verbose = false;          // Print optimization progress
};

struct LBFGSResult {
    std::vector<double> x;         // Optimal parameters
    double f;                      // Optimal function value
    std::vector<double> gradient;  // Final gradient
    int iterations;                // Number of iterations
    int function_evaluations;      // Number of function evaluations  
    bool converged;                // Whether optimization converged
    std::string message;           // Status message
};

// Objective function signature: f(x, grad) -> value
// Function should compute both value and gradient
using ObjectiveFunction = std::function<double(const std::vector<double>&, std::vector<double>&)>;

// Box-constrained L-BFGS-B optimizer
class LBFGSBOptimizer {
public:
    LBFGSBOptimizer(const LBFGSOptions& options = LBFGSOptions{});

    // Main optimization interface
    LBFGSResult minimize(
        const ObjectiveFunction& objective,
        const std::vector<double>& initial_x,
        const std::vector<double>& lower_bounds,
        const std::vector<double>& upper_bounds
    );

    // Finite difference gradient approximation (fallback)
    static ObjectiveFunction finite_difference_wrapper(
        const std::function<double(const std::vector<double>&)>& f,
        double epsilon = 1e-5
    );

private:
    LBFGSOptions options_;

    // Internal optimization state
    struct InternalState {
        std::vector<std::vector<double>> s_vectors; // Parameter differences
        std::vector<std::vector<double>> y_vectors; // Gradient differences  
        std::vector<double> rho_values;             // 1 / (s^T y)
        int memory_start = 0;                       // Circular buffer start
        int memory_count = 0;                       // Number of stored pairs
    };

    // Helper methods
    void project_bounds(
        std::vector<double>& x,
        const std::vector<double>& lower,
        const std::vector<double>& upper
    ) const;

    double line_search(
        const ObjectiveFunction& objective,
        const std::vector<double>& x,
        const std::vector<double>& gradient,
        const std::vector<double>& direction,
        const std::vector<double>& lower_bounds,
        const std::vector<double>& upper_bounds,
        double initial_step
    ) const;

    void compute_search_direction(
        const std::vector<double>& gradient,
        const InternalState& state,
        std::vector<double>& direction
    ) const;

    void update_memory(
        const std::vector<double>& x_old,
        const std::vector<double>& x_new,
        const std::vector<double>& grad_old,
        const std::vector<double>& grad_new,
        InternalState& state
    ) const;

    double compute_gradient_norm(const std::vector<double>& gradient) const;
    bool check_convergence(
        const std::vector<double>& x_old,
        const std::vector<double>& x_new,
        const std::vector<double>& gradient
    ) const;
};

} // namespace Optimization
} // namespace PseudomodeFramework
