/*
 * Material Database Implementation - Complete 2D Materials
 * Apache License 2.0 - Copyright (c) 2025 Aetheron Research
 */

#include "pseudomode_solver_complete.h"
#include <cmath>
#include <algorithm>

namespace PseudomodeFramework {

// Static member initialization
std::unordered_map<std::string, std::unordered_map<std::string, double>> 
MaterialDatabase::material_params_ = {
    {"MoS2", {
        {"alpha_ac", 0.01}, {"omega_c_ac", 0.04}, {"q_ac", 1.5},
        {"alpha_flex", 0.005}, {"omega_f_flex", 0.02}, {"s_f_flex", 0.3}, {"q_flex", 2.0},
        {"Omega_opt", 0.048}, {"lambda_opt", 0.002}, {"Gamma_opt", 0.001}
    }},
    {"WSe2", {
        {"alpha_ac", 0.012}, {"omega_c_ac", 0.035}, {"q_ac", 1.5},
        {"alpha_flex", 0.008}, {"omega_f_flex", 0.018}, {"s_f_flex", 0.4}, {"q_flex", 2.0},
        {"Omega_opt", 0.032}, {"lambda_opt", 0.003}, {"Gamma_opt", 0.0008}
    }},
    {"graphene", {
        {"alpha_ac", 0.008}, {"omega_c_ac", 0.15}, {"q_ac", 1.2},
        {"alpha_flex", 0.015}, {"omega_f_flex", 0.025}, {"s_f_flex", 0.2}, {"q_flex", 2.0},
        {"Omega_opt", 0.0}, {"lambda_opt", 0.0}, {"Gamma_opt", 0.001}  // No optical phonons
    }},
    {"GaN_2D", {
        {"alpha_ac", 0.02}, {"omega_c_ac", 0.08}, {"q_ac", 1.8},
        {"alpha_flex", 0.003}, {"omega_f_flex", 0.06}, {"s_f_flex", 0.6}, {"q_flex", 2.0},
        {"Omega_opt", 0.0}, {"lambda_opt", 0.0}, {"Gamma_opt", 0.001}
    }}
};

RealVector MaterialDatabase::build_spectral_density(
    const RealVector& omega_grid,
    const std::string& material) {

    auto params_it = material_params_.find(material);
    if (params_it == material_params_.end()) {
        throw std::invalid_argument("Unknown material: " + material);
    }

    const auto& params = params_it->second;

    // Acoustic phonons
    RealVector J_ac = acoustic_2d(
        omega_grid,
        params.at("alpha_ac"),
        params.at("omega_c_ac"),
        params.at("q_ac")
    );

    // Flexural phonons
    RealVector J_flex = flexural_2d(
        omega_grid,
        params.at("alpha_flex"),
        params.at("omega_f_flex"),
        params.at("s_f_flex"),
        params.at("q_flex")
    );

    // Optical phonons (if present)
    RealVector J_opt(omega_grid.size(), 0.0);
    if (params.at("lambda_opt") > 0.0) {
        J_opt = optical_peak(
            omega_grid,
            params.at("Omega_opt"),
            params.at("lambda_opt"),
            params.at("Gamma_opt")
        );
    }

    // Combine all contributions
    RealVector J_total(omega_grid.size());
    for (size_t i = 0; i < omega_grid.size(); ++i) {
        J_total[i] = J_ac[i] + J_flex[i] + J_opt[i];
    }

    return J_total;
}

RealVector MaterialDatabase::acoustic_2d(
    const RealVector& omega,
    double alpha,
    double omega_c,
    double q) {

    RealVector J(omega.size());

    #pragma omp parallel for if(omega.size() > 1000)
    for (size_t i = 0; i < omega.size(); ++i) {
        if (omega[i] <= 0.0) {
            J[i] = 0.0;
        } else {
            J[i] = alpha * omega[i] * std::exp(-std::pow(omega[i] / omega_c, q));
        }
    }

    return J;
}

RealVector MaterialDatabase::flexural_2d(
    const RealVector& omega,
    double alpha_f,
    double omega_f,
    double s_f,
    double q) {

    RealVector J(omega.size());

    #pragma omp parallel for if(omega.size() > 1000)
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

RealVector MaterialDatabase::optical_peak(
    const RealVector& omega,
    double Omega_j,
    double lambda_j,
    double Gamma_j) {

    RealVector J(omega.size());

    #pragma omp parallel for if(omega.size() > 1000)
    for (size_t i = 0; i < omega.size(); ++i) {
        double denominator = std::pow(omega[i] - Omega_j, 2) + std::pow(Gamma_j, 2);
        J[i] = (2.0 * lambda_j * lambda_j * Gamma_j) / denominator;
    }

    return J;
}

} // namespace PseudomodeFramework
