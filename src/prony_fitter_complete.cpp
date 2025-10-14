/*
 * Complete Prony Fitter Implementation
 * Apache License 2.0 - Copyright (c) 2025 Aetheron Research
 */

#include "pseudomode_solver_complete.h"

#ifdef USE_EIGEN
#include <Eigen/Dense>
#include <Eigen/SVD>
#endif

namespace PseudomodeFramework {

PronyFitter::PronyFitter(int max_modes, double regularization)
    : max_modes_(max_modes), regularization_(regularization) {

    if (max_modes <= 0) {
        throw std::invalid_argument("max_modes must be positive");
    }
    if (regularization < 0) {
        throw std::invalid_argument("regularization must be non-negative");
    }
}

ComplexVector PronyFitter::spectrum_to_correlation(
    const RealVector& J_omega,
    const RealVector& omega_grid,
    const RealVector& t_grid,
    double temperature_K) {

    ComplexVector C_t(t_grid.size());
    const double kB = PhysConstants::KB_EV;

    #pragma omp parallel for if(t_grid.size() > 100)
    for (size_t i = 0; i < t_grid.size(); ++i) {
        double t = t_grid[i];
        Complex integral = 0.0;

        for (size_t j = 0; j < omega_grid.size(); ++j) {
            double omega = omega_grid[j];

            // Thermal occupation number
            double n_omega = 0.0;
            if (omega > 0.0 && temperature_K > 0.0) {
                double beta_omega = omega / (kB * temperature_K);
                if (beta_omega < 50.0) { // Avoid overflow
                    n_omega = 1.0 / (std::exp(beta_omega) - 1.0);
                }
            }

            // Correlation function integrand
            Complex exp_neg = std::exp(Complex(0.0, -omega * t));
            Complex exp_pos = std::exp(Complex(0.0, omega * t));

            integral += J_omega[j] * (n_omega * exp_neg + (n_omega + 1.0) * exp_pos);
        }

        // Trapezoidal integration
        if (omega_grid.size() > 1) {
            double domega = omega_grid[1] - omega_grid[0];
            C_t[i] = integral * domega;
        } else {
            C_t[i] = integral;
        }
    }

    return C_t;
}

PronyFitter::FitResult PronyFitter::fit_correlation(
    const ComplexVector& C_data,
    const RealVector& t_grid,
    double temperature_K) {

    if (C_data.size() != t_grid.size()) {
        throw std::invalid_argument("C_data and t_grid size mismatch");
    }

    if (C_data.size() < 2 * max_modes_) {
        throw std::invalid_argument("Insufficient data points for fitting");
    }

    FitResult best_result;
    best_result.bic = std::numeric_limits<double>::infinity();

    // Try different numbers of modes
    for (int K = 1; K <= max_modes_; ++K) {
        try {
            auto result = fit_single_K(C_data, t_grid, K, temperature_K);

            if (result.converged && result.bic < best_result.bic) {
                best_result = std::move(result);
            }

        } catch (const std::exception& e) {
            // Continue with next K value
            continue;
        }
    }

    if (!best_result.converged) {
        best_result.message = "All fits failed";
    }

    return best_result;
}

PronyFitter::FitResult PronyFitter::fit_single_K(
    const ComplexVector& C_data,
    const RealVector& t_grid,
    int K,
    double temperature_K) {

    const int N = C_data.size();
    const int M = N - K;

    if (M <= 0) {
        throw std::invalid_argument("K too large for data size");
    }

#ifdef USE_EIGEN
    // Use Eigen for numerical stability
    Eigen::MatrixXcd H(M, K);
    Eigen::VectorXcd c(M);

    // Build Hankel matrix
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            H(i, j) = C_data[i + j];
        }
        c(i) = -C_data[i + K];
    }

    // Solve with regularization
    Eigen::MatrixXcd HtH = H.adjoint() * H;
    Eigen::VectorXcd Htc = H.adjoint() * c;

    // Add regularization
    HtH += regularization_ * Eigen::MatrixXcd::Identity(K, K);

    // Solve linear system
    Eigen::VectorXcd a_coeffs = HtH.ldlt().solve(Htc);

    // Convert to std::vector for root finding
    ComplexVector coeffs(K + 1);
    coeffs[0] = 1.0;
    for (int i = 0; i < K; ++i) {
        coeffs[i + 1] = a_coeffs(i);
    }

#else
    // Fallback implementation without Eigen
    throw std::runtime_error("Eigen required for stable Prony fitting");
#endif

    // Find polynomial roots using companion matrix
    auto roots = companion_matrix_roots(coeffs);

    // Extract physical parameters
    double dt = t_grid.size() > 1 ? (t_grid[1] - t_grid[0]) : 1e-12;
    std::vector<PseudomodeParams> modes;

    for (const auto& root : roots) {
        if (std::abs(root) > 1e-12) {
            Complex log_root = std::log(root + Complex(1e-16, 0));
            double gamma_k = -std::real(log_root) / dt;
            double omega_k = std::imag(log_root) / dt;

            if (gamma_k > 0) { // Stable mode
                PseudomodeParams mode;
                mode.omega_eV = omega_k;
                mode.gamma_eV = gamma_k;
                mode.g_eV = 0.0; // Will be fitted next
                mode.type = "fitted";
                modes.push_back(mode);
            }
        }
    }

    if (modes.empty()) {
        FitResult result;
        result.converged = false;
        result.message = "No stable modes found";
        return result;
    }

    // Fit coupling strengths
#ifdef USE_EIGEN
    Eigen::MatrixXcd A(N, modes.size());
    for (int i = 0; i < N; ++i) {
        for (size_t k = 0; k < modes.size(); ++k) {
            Complex exponent = -(modes[k].gamma_eV + Complex(0, modes[k].omega_eV)) * t_grid[i];
            A(i, k) = std::exp(exponent);
        }
    }

    Eigen::VectorXcd C_vec(N);
    for (int i = 0; i < N; ++i) {
        C_vec(i) = C_data[i];
    }

    Eigen::VectorXcd eta = A.colPivHouseholderQr().solve(C_vec);

    for (size_t k = 0; k < modes.size(); ++k) {
        modes[k].g_eV = std::sqrt(std::max(0.0, std::real(eta(k))));
    }
#endif

    // Apply physical constraints
    apply_physical_constraints(modes, temperature_K);

    // Compute fit quality
    ComplexVector C_fit(N, 0.0);
    for (int i = 0; i < N; ++i) {
        for (const auto& mode : modes) {
            double eta_k = mode.g_eV * mode.g_eV;
            Complex exponent = -(mode.gamma_eV + Complex(0, mode.omega_eV)) * t_grid[i];
            C_fit[i] += eta_k * std::exp(exponent);
        }
    }

    // RMSE
    double rmse = 0.0;
    for (int i = 0; i < N; ++i) {
        rmse += std::norm(C_data[i] - C_fit[i]);
    }
    rmse = std::sqrt(rmse / N);

    FitResult result;
    result.modes = modes;
    result.rmse = rmse;
    result.bic = compute_bic(result, N);
    result.converged = true;
    result.message = "Fit successful";

    return result;
}

std::vector<Complex> PronyFitter::companion_matrix_roots(
    const ComplexVector& coefficients) {

    int n = coefficients.size() - 1;
    if (n <= 0) return {};

#ifdef USE_EIGEN
    // Build companion matrix
    Eigen::MatrixXcd C = Eigen::MatrixXcd::Zero(n, n);

    if (n > 1) {
        C.block(1, 0, n-1, n-1) = Eigen::MatrixXcd::Identity(n-1, n-1);
    }

    for (int i = 0; i < n; ++i) {
        C(i, n-1) = -coefficients[n-i] / coefficients[0];
    }

    // Compute eigenvalues
    Eigen::ComplexEigenSolver<Eigen::MatrixXcd> solver(C);
    auto eigenvals = solver.eigenvalues();

    std::vector<Complex> roots;
    for (int i = 0; i < eigenvals.size(); ++i) {
        roots.push_back(Complex(eigenvals(i).real(), eigenvals(i).imag()));
    }

    return roots;
#else
    throw std::runtime_error("Eigen required for companion matrix method");
#endif
}

void PronyFitter::apply_physical_constraints(
    std::vector<PseudomodeParams>& modes,
    double temperature_K) {

    const double kB = PhysConstants::KB_EV;

    for (auto& mode : modes) {
        // Enforce positive decay rate
        mode.gamma_eV = std::max(mode.gamma_eV, 1e-6);

        // Limit coupling strength
        mode.g_eV = std::min(mode.g_eV, 1.0);

        // Compute thermal occupation
        if (mode.omega_eV > 0 && temperature_K > 0) {
            double beta_omega = mode.omega_eV / (kB * temperature_K);
            if (beta_omega < 50) {
                mode.n_thermal = 1.0 / (std::exp(beta_omega) - 1.0);
            } else {
                mode.n_thermal = 0.0;
            }
        } else {
            mode.n_thermal = 0.0;
        }

        // Classify mode type based on frequency
        double omega_meV = mode.omega_eV * 1000;
        if (omega_meV < 25) {
            if (omega_meV < 15) {
                mode.type = "flexural";
            } else {
                mode.type = "acoustic";
            }
        } else {
            mode.type = "optical";
        }
    }
}

double PronyFitter::compute_bic(const FitResult& result, int n_data) {
    if (!result.converged) {
        return std::numeric_limits<double>::infinity();
    }

    int n_params = 3 * result.modes.size(); // ω, γ, g per mode
    double log_likelihood = -0.5 * n_data * std::log(result.rmse * result.rmse);

    return n_params * std::log(n_data) - 2.0 * log_likelihood;
}

} // namespace PseudomodeFramework
