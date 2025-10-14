/*
 * SpectralDensity2D Implementation
 * Apache License 2.0 - Copyright (c) 2025 Aetheron Research
 */

#include "pseudomode_solver.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace PseudomodeSolver {

std::vector<double> SpectralDensity2D::acoustic(
    const std::vector<double>& omega,
    double alpha,
    double omega_c,
    double q) {

    std::vector<double> J(omega.size());

    #pragma omp parallel for
    for (size_t i = 0; i < omega.size(); ++i) {
        if (omega[i] <= 0.0) {
            J[i] = 0.0;
        } else {
            J[i] = alpha * omega[i] * std::exp(-std::pow(omega[i] / omega_c, q));
        }
    }

    return J;
}

std::vector<double> SpectralDensity2D::flexural(
    const std::vector<double>& omega,
    double alpha_f,
    double omega_f,
    double s_f,
    double q) {

    std::vector<double> J(omega.size());

    #pragma omp parallel for
    for (size_t i = 0; i < omega.size(); ++i) {
        if (omega[i] <= 0.0) {
            J[i] = 0.0;
        } else {
            J[i] = alpha_f * std::pow(omega[i], s_f) * 
                   std::exp(-std::pow(omega[i] / omega_f, q));
        }
    }

    return J;
}

std::vector<double> SpectralDensity2D::lorentzian_peak(
    const std::vector<double>& omega,
    double Omega_j,
    double lambda_j,
    double Gamma_j) {

    std::vector<double> J(omega.size());

    #pragma omp parallel for
    for (size_t i = 0; i < omega.size(); ++i) {
        double denominator = std::pow(omega[i] - Omega_j, 2) + std::pow(Gamma_j, 2);
        J[i] = (2.0 * lambda_j * lambda_j * Gamma_j) / denominator;
    }

    return J;
}

std::vector<double> SpectralDensity2D::build_material_spectrum(
    const std::vector<double>& omega,
    const std::string& material,
    const std::unordered_map<std::string, double>& params) {

    std::vector<double> J_total(omega.size(), 0.0);

    if (material == "MoS2") {
        // MoS2 monolayer parameters
        auto J_ac = acoustic(omega, 0.01, 0.04, 1.5);
        auto J_flex = flexural(omega, 0.005, 0.02, 0.3, 2.0);
        auto J_optical = lorentzian_peak(omega, 0.048, 0.002, 0.001);

        #pragma omp parallel for
        for (size_t i = 0; i < omega.size(); ++i) {
            J_total[i] = J_ac[i] + J_flex[i] + J_optical[i];
        }

    } else if (material == "WSe2") {
        // WSe2 monolayer (stronger SOC)
        auto J_ac = acoustic(omega, 0.012, 0.035, 1.5);
        auto J_flex = flexural(omega, 0.008, 0.018, 0.4, 2.0);
        auto J_optical = lorentzian_peak(omega, 0.032, 0.003, 0.0008);

        #pragma omp parallel for
        for (size_t i = 0; i < omega.size(); ++i) {
            J_total[i] = J_ac[i] + J_flex[i] + J_optical[i];
        }

    } else if (material == "graphene") {
        // Graphene on hBN
        auto J_ac = acoustic(omega, 0.008, 0.15, 1.2);
        auto J_flex = flexural(omega, 0.015, 0.025, 0.2, 2.0);

        #pragma omp parallel for
        for (size_t i = 0; i < omega.size(); ++i) {
            J_total[i] = J_ac[i] + J_flex[i]; // minimal optical coupling
        }

    } else if (material == "GaN_2D") {
        // Hypothetical 2D GaN
        auto J_ac = acoustic(omega, 0.02, 0.08, 1.8);
        auto J_flex = flexural(omega, 0.003, 0.06, 0.6, 2.0);

        #pragma omp parallel for 
        for (size_t i = 0; i < omega.size(); ++i) {
            J_total[i] = J_ac[i] + J_flex[i];
        }

    } else {
        throw std::invalid_argument("Unknown material: " + material);
    }

    return J_total;
}

} // namespace PseudomodeSolver
