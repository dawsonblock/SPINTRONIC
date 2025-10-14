/*
 * Prony Fitting Implementation - C++/CUDA Version
 * Apache License 2.0 - Copyright (c) 2025 Aetheron Research
 */

#include "pseudomode_solver.h"
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <complex>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace PseudomodeSolver {

// Simple linear system solver (Gauss-Jordan elimination)
// Avoids expensive Eigen template instantiation
static std::vector<double> solve_linear_system(
    std::vector<std::vector<double>> A, 
    std::vector<double> b) {
    
    const int n = b.size();
    if (n == 0 || A.size() != static_cast<size_t>(n)) {
        return b; // Return input on error
    }
    
    // Augment matrix [A|b]
    for (int i = 0; i < n; ++i) {
        A[i].push_back(b[i]);
    }
    
    // Forward elimination with partial pivoting
    for (int k = 0; k < n; ++k) {
        // Find pivot
        int pivot = k;
        double max_val = std::abs(A[k][k]);
        for (int i = k + 1; i < n; ++i) {
            if (std::abs(A[i][k]) > max_val) {
                max_val = std::abs(A[i][k]);
                pivot = i;
            }
        }
        
        // Swap rows
        if (pivot != k) {
            std::swap(A[k], A[pivot]);
        }
        
        // Check for singularity
        if (std::abs(A[k][k]) < 1e-12) {
            return b; // Return original vector on singular matrix
        }
        
        // Eliminate column
        for (int i = k + 1; i < n; ++i) {
            double factor = A[i][k] / A[k][k];
            for (int j = k; j <= n; ++j) {
                A[i][j] -= factor * A[k][j];
            }
        }
    }
    
    // Back substitution
    std::vector<double> x(n);
    for (int i = n - 1; i >= 0; --i) {
        x[i] = A[i][n];
        for (int j = i + 1; j < n; ++j) {
            x[i] -= A[i][j] * x[j];
        }
        x[i] /= A[i][i];
    }
    
    return x;
}

PronyFitter::FitResult PronyFitter::fit_correlation(
    const std::vector<Complex>& C_data,
    const std::vector<double>& t_grid,
    int max_modes,
    double temperature_K) {

    Utils::Timer timer("PronyFitter::fit_correlation");

    FitResult best_result;
    best_result.bic = std::numeric_limits<double>::infinity();
    best_result.converged = false;

    std::cout << "Fitting correlation function with BIC model selection..." << std::endl;
    std::cout << "Data points: " << C_data.size() << ", Max K: " << max_modes << std::endl;

    // BIC scan over number of pseudomodes
    for (int K = 1; K <= max_modes; ++K) {
        std::cout << "Trying K = " << K << " pseudomodes..." << std::endl;

        // Initial Prony fit
        auto prony_result = fit_prony_initial(C_data, t_grid, K);

        if (!prony_result.converged) {
            std::cout << "  Prony fit failed: " << prony_result.message << std::endl;
            continue;
        }

        // Constrained refinement
        auto refined_result = refine_parameters(
            prony_result.modes, C_data, t_grid, temperature_K
        );

        if (!refined_result.converged) {
            std::cout << "  Refinement failed: " << refined_result.message << std::endl;
            continue;
        }

        // Compute BIC
        double bic = compute_bic(refined_result, C_data.size());
        refined_result.bic = bic;

        std::cout << "  RMSE = " << refined_result.rmse 
                  << ", BIC = " << bic << std::endl;

        // Update best result if BIC improved
        if (bic < best_result.bic) {
            best_result = std::move(refined_result);
            best_result.converged = true;
        }
    }

    if (best_result.converged) {
        std::cout << "Optimal K = " << best_result.modes.size() 
                  << ", BIC = " << best_result.bic << std::endl;
    } else {
        best_result.message = "All fits failed";
    }

    return best_result;
}

PronyFitter::FitResult PronyFitter::fit_prony_initial(
    const std::vector<Complex>& C_data,
    const std::vector<double>& t_grid,
    int K) {

    FitResult result;
    result.converged = false;

    const int M = C_data.size();

    if (K >= M / 2) {
        result.message = "K too large for data length";
        return result;
    }

    try {
        // Build Hankel matrix using Eigen
        auto hankel_matrix = create_hankel_matrix(C_data, K);

        Eigen::MatrixXcd H(M - K, K);
        for (int i = 0; i < M - K; ++i) {
            for (int j = 0; j < K; ++j) {
                H(i, j) = hankel_matrix[i][j];
            }
        }

        // Right-hand side vector
        Eigen::VectorXcd c(M - K);
        for (int i = 0; i < M - K; ++i) {
            c(i) = -C_data[i + K];
        }

        // Solve linear system using SVD (robust to ill-conditioning)
        Eigen::JacobiSVD<Eigen::MatrixXcd> svd(
            H, Eigen::ComputeThinU | Eigen::ComputeThinV
        );

        // Apply Tikhonov regularization if needed
        double rcond = 1e-12;
        Eigen::VectorXcd a = svd.solve(c);

        // Form characteristic polynomial: z^K + a₁z^{K-1} + ... + aₖ = 0
        Eigen::VectorXcd poly_coeffs(K + 1);
        poly_coeffs(0) = 1.0;
        for (int i = 0; i < K; ++i) {
            poly_coeffs(i + 1) = a(i);
        }

        // Find roots using companion matrix method
        auto roots = find_polynomial_roots(poly_coeffs);

        // Extract physical parameters
        const double dt = t_grid[1] - t_grid[0];
        std::vector<PseudomodeParams> modes;

        for (const auto& root : roots) {
            if (std::abs(root) > 1e-12) {
                Complex log_root = std::log(root);
                double gamma_k = -std::real(log_root) / dt;
                double omega_k = -std::imag(log_root) / dt;

                if (gamma_k > 0.0) { // Stable mode
                    PseudomodeParams mode;
                    mode.gamma_eV = gamma_k;
                    mode.omega_eV = omega_k;
                    mode.mode_type = "prony_fitted";
                    modes.push_back(mode);
                }
            }
        }

        if (modes.empty()) {
            result.message = "No stable modes found";
            return result;
        }

        // Solve for coupling strengths |g_k|² = η_k
        Eigen::MatrixXcd A(M, modes.size());
        for (int i = 0; i < M; ++i) {
            for (size_t k = 0; k < modes.size(); ++k) {
                Complex exponent = -(modes[k].gamma_eV + 
                                   Complex(0.0, modes[k].omega_eV)) * t_grid[i];
                A(i, k) = std::exp(exponent);
            }
        }

        Eigen::VectorXcd C_vec(M);
        for (int i = 0; i < M; ++i) {
            C_vec(i) = C_data[i];
        }

        // Least squares solve for amplitudes
        Eigen::VectorXcd eta = A.colPivHouseholderQr().solve(C_vec);

        // Set coupling strengths (ensure non-negative)
        for (size_t k = 0; k < modes.size(); ++k) {
            double eta_k = std::max(0.0, std::real(eta(k)));
            modes[k].g_eV = std::sqrt(eta_k);
        }

        // Compute RMSE
        Eigen::VectorXcd C_fit = A * eta;
        double rmse = (C_vec - C_fit).norm() / std::sqrt(M);

        result.modes = std::move(modes);
        result.rmse = rmse;
        result.converged = true;
        result.message = "Prony fit successful";

    } catch (const std::exception& e) {
        result.message = "Prony fit exception: " + std::string(e.what());
    }

    return result;
}

std::vector<std::vector<Complex>> PronyFitter::create_hankel_matrix(
    const std::vector<Complex>& data,
    int K) {

    const int M = data.size();
    const int rows = M - K;

    std::vector<std::vector<Complex>> H(rows, std::vector<Complex>(K));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < K; ++j) {
            H[i][j] = data[i + j];
        }
    }

    return H;
}

std::vector<Complex> PronyFitter::find_polynomial_roots(
    const Eigen::VectorXcd& coeffs) {

    const int degree = coeffs.size() - 1;

    if (degree <= 0) {
        return {};
    }

    // Companion matrix method
    Eigen::MatrixXcd companion = Eigen::MatrixXcd::Zero(degree, degree);

    // Fill companion matrix
    for (int i = 0; i < degree - 1; ++i) {
        companion(i + 1, i) = 1.0;
    }

    for (int i = 0; i < degree; ++i) {
        companion(i, degree - 1) = -coeffs(degree - i) / coeffs(0);
    }

    // Compute eigenvalues (= polynomial roots)
    Eigen::ComplexEigenSolver<Eigen::MatrixXcd> solver(companion);

    std::vector<Complex> roots;
    auto eigenvalues = solver.eigenvalues();

    for (int i = 0; i < eigenvalues.size(); ++i) {
        roots.push_back(Complex(eigenvalues(i).real(), eigenvalues(i).imag()));
    }

    return roots;
}

PronyFitter::FitResult PronyFitter::refine_parameters(
    const std::vector<PseudomodeParams>& initial_params,
    const std::vector<Complex>& C_data,
    const std::vector<double>& t_grid,
    double temperature_K) {

    FitResult result;
    result.converged = false;

    const int K = initial_params.size();
    const int M = C_data.size();

    // Parameter vector: [ω₁...ωₖ, γ₁...γₖ, η₁...ηₖ]
    Eigen::VectorXd theta(3 * K);

    for (int k = 0; k < K; ++k) {
        theta(k) = initial_params[k].omega_eV;
        theta(k + K) = std::max(1e-6, initial_params[k].gamma_eV); // Ensure positive
        theta(k + 2*K) = std::max(0.0, initial_params[k].g_eV * initial_params[k].g_eV); // η = |g|²
    }

    // Levenberg-Marquardt parameters
    double lambda = 0.01;
    const int max_iterations = 100;
    const double tolerance = 1e-8;

    for (int iter = 0; iter < max_iterations; ++iter) {
        // Compute residuals and Jacobian
        auto [residuals, jacobian] = compute_residuals_and_jacobian(
            theta, C_data, t_grid, temperature_K
        );

        // Compute loss using manual dot product to avoid .squaredNorm() template
        double current_loss = 0.0;
        for (int i = 0; i < residuals.size(); ++i) {
            current_loss += residuals(i) * residuals(i);
        }
        current_loss *= 0.5;

        // Compute J^T * J manually (avoid expensive template instantiation)
        const int n_params = 3 * K;
        std::vector<std::vector<double>> JTJ_data(n_params, std::vector<double>(n_params, 0.0));
        std::vector<double> JTr_data(n_params, 0.0);
        
        // Compute J^T * J and J^T * r
        for (int i = 0; i < n_params; ++i) {
            for (int j = 0; j < n_params; ++j) {
                double sum = 0.0;
                for (int k = 0; k < residuals.size(); ++k) {
                    sum += jacobian(k, i) * jacobian(k, j);
                }
                JTJ_data[i][j] = sum;
            }
            
            double sum = 0.0;
            for (int k = 0; k < residuals.size(); ++k) {
                sum += jacobian(k, i) * residuals(k);
            }
            JTr_data[i] = sum;
        }
        
        // Add regularization: H = J^T*J + λ*I
        for (int i = 0; i < n_params; ++i) {
            JTJ_data[i][i] += lambda;
        }
        
        // Solve using simple Gauss-Jordan (replace LDLT)
        std::vector<double> delta_theta_data = solve_linear_system(JTJ_data, JTr_data);
        
        // Build new parameter vector
        Eigen::VectorXd delta_theta(n_params);
        Eigen::VectorXd theta_new(n_params);
        double delta_norm_sq = 0.0;
        
        for (int i = 0; i < n_params; ++i) {
            delta_theta(i) = -delta_theta_data[i];
            theta_new(i) = theta(i) + delta_theta(i);
            delta_norm_sq += delta_theta(i) * delta_theta(i);
        }

        // Project onto feasible region
        project_onto_constraints(theta_new, K);

        // Check if update improves loss
        auto [new_residuals, jacobian_unused] = compute_residuals_and_jacobian(
            theta_new, C_data, t_grid, temperature_K
        );
        [[maybe_unused]] auto& unused_jacobian = jacobian_unused;
        
        double new_loss = 0.0;
        for (int i = 0; i < new_residuals.size(); ++i) {
            new_loss += new_residuals(i) * new_residuals(i);
        }
        new_loss *= 0.5;

        if (new_loss < current_loss) {
            // Accept update
            theta = theta_new;
            lambda *= 0.1; // Reduce damping

            // Check convergence
            if (std::sqrt(delta_norm_sq) < tolerance) {
                result.converged = true;
                break;
            }
        } else {
            // Reject update, increase damping
            lambda *= 10.0;
            if (lambda > 1e6) {
                result.message = "Levenberg-Marquardt diverged";
                break;
            }
        }
    }

    if (result.converged) {
        // Extract final parameters
        std::vector<PseudomodeParams> final_modes(K);
        for (int k = 0; k < K; ++k) {
            final_modes[k].omega_eV = theta(k);
            final_modes[k].gamma_eV = theta(k + K);
            final_modes[k].g_eV = std::sqrt(std::max(0.0, theta(k + 2*K)));
            final_modes[k].mode_type = "refined";
        }

        // Compute final RMSE
        auto [final_residuals, _] = compute_residuals_and_jacobian(
            theta, C_data, t_grid, temperature_K
        );

        // Compute norm manually (avoid .norm() template)
        double norm_sq = 0.0;
        for (int i = 0; i < final_residuals.size(); ++i) {
            norm_sq += final_residuals(i) * final_residuals(i);
        }

        result.modes = std::move(final_modes);
        result.rmse = std::sqrt(norm_sq) / std::sqrt(2 * M); // Factor 2 for complex
        result.message = "Constrained refinement successful";

    } else {
        result.message = "Refinement failed to converge";
    }

    return result;
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> 
PronyFitter::compute_residuals_and_jacobian(
    const Eigen::VectorXd& theta,
    const std::vector<Complex>& C_data,
    const std::vector<double>& t_grid,
    double temperature_K) {

    const int K = theta.size() / 3;
    const int M = C_data.size();

    // Pre-allocate return structures
    Eigen::VectorXd residuals(2 * M);
    Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(2 * M, 3 * K);

    // Manual extraction (avoid segment for faster compilation)
    std::vector<double> omega(K), gamma(K), eta(K);
    for (int k = 0; k < K; ++k) {
        omega[k] = theta(k);
        gamma[k] = theta(k + K);
        eta[k] = theta(k + 2*K);
    }

    // Combined residual and Jacobian computation (single pass)
    for (int i = 0; i < M; ++i) {
        const double t = t_grid[i];
        Complex C_model(0.0, 0.0);
        
        for (int k = 0; k < K; ++k) {
            const double exp_arg_real = -gamma[k] * t;
            const double exp_arg_imag = -omega[k] * t;
            
            // Compute exp(-(γ + iω)t) = exp(-γt) * [cos(ωt) - i*sin(ωt)]
            const double exp_real = std::exp(exp_arg_real);
            const double cos_wt = std::cos(exp_arg_imag);
            const double sin_wt = std::sin(exp_arg_imag);
            
            const Complex exp_val(exp_real * cos_wt, -exp_real * sin_wt);
            C_model += eta[k] * exp_val;
            
            // Jacobian computation inline
            // ∂C/∂ωₖ = -i t ηₖ exp(-(γₖ + iωₖ)t)
            const Complex d_omega = Complex(0.0, -1.0) * t * eta[k] * exp_val;
            jacobian(i, k) = -d_omega.real();
            jacobian(i + M, k) = -d_omega.imag();
            
            // ∂C/∂γₖ = -t ηₖ exp(-(γₖ + iωₖ)t)
            const Complex d_gamma = -t * eta[k] * exp_val;
            jacobian(i, K + k) = -d_gamma.real();
            jacobian(i + M, K + k) = -d_gamma.imag();
            
            // ∂C/∂ηₖ = exp(-(γₖ + iωₖ)t)
            jacobian(i, 2*K + k) = -exp_val.real();
            jacobian(i + M, 2*K + k) = -exp_val.imag();
        }
        
        // Residuals
        const Complex diff = C_data[i] - C_model;
        residuals(i) = diff.real();
        residuals(i + M) = diff.imag();
    }

    // Add constraint penalties (simple version to avoid conservativeResize)
    const double penalty_weight = 100.0;
    for (int k = 0; k < K; ++k) {
        if (gamma[k] < 0) {
            residuals(k % M) += penalty_weight * (-gamma[k]);
        }
        if (eta[k] < 0) {
            residuals(k % M) += penalty_weight * (-eta[k]);
        }
    }

    return {residuals, jacobian};
}

double PronyFitter::compute_bic(const FitResult& fit, int n_data_points) {
    if (!fit.converged) {
        return std::numeric_limits<double>::infinity();
    }

    const int p = 3 * fit.modes.size(); // 3 parameters per mode
    const double log_likelihood = -0.5 * n_data_points * std::log(fit.rmse * fit.rmse);

    return p * std::log(n_data_points) - 2.0 * log_likelihood;
}

// Helper function stubs (implementations may be refined based on physics requirements)
void PronyFitter::add_constraint_penalties(
    const Eigen::VectorXd& theta,
    Eigen::VectorXd& residuals,
    double temperature_K) {
    // Constraints now handled inline in compute_residuals_and_jacobian
    // to avoid expensive conservativeResize operations
}

Eigen::MatrixXd PronyFitter::compute_jacobian(
    const Eigen::VectorXd& theta,
    const std::vector<double>& t_grid,
    double temperature_K) {
    
    // Simplified stub - actual computation done in compute_residuals_and_jacobian
    const int K = theta.size() / 3;
    const int M = t_grid.size();
    return Eigen::MatrixXd::Zero(2 * M, 3 * K);
}

void PronyFitter::project_onto_constraints(
    Eigen::VectorXd& theta,
    int K) {
    // Project parameters to ensure physical validity
    
    // ω (frequencies) can be any value
    // γ (damping rates) must be positive
    for (int k = 0; k < K; ++k) {
        if (theta(K + k) < 1e-10) {
            theta(K + k) = 1e-10;  // Small positive value
        }
    }
    
    // η (coupling strengths) must be positive
    for (int k = 0; k < K; ++k) {
        if (theta(2*K + k) < 1e-10) {
            theta(2*K + k) = 1e-10;
        }
    }
}

} // namespace PseudomodeSolver
