/*
 * Complete High-Level Framework Implementation
 * Apache License 2.0 - Copyright (c) 2025 Aetheron Research
 */

#include "pseudomode_solver_complete.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <thread>
#include <future>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace PseudomodeFramework {

PseudomodeFramework::PseudomodeFramework(const SimulationConfig& config)
    : config_(config) {

    materials_ = std::make_unique<MaterialDatabase>();
    fitter_ = std::make_unique<PronyFitter>(config_.max_modes);

    std::cout << "Initialized 2D Pseudomode Framework" << std::endl;
    std::cout << "  Max modes: " << config_.max_modes << std::endl;
    std::cout << "  n_max: " << config_.total_time_ps << " ps" << std::endl;
    std::cout << "  GPU: " << (config_.use_gpu ? "enabled" : "disabled") << std::endl;

#ifdef USE_OPENMP
    int n_threads = config_.n_threads > 0 ? config_.n_threads : omp_get_max_threads();
    omp_set_num_threads(n_threads);
    std::cout << "  OpenMP threads: " << n_threads << std::endl;
#endif
}

PseudomodeFramework::~PseudomodeFramework() = default;

SimulationResult PseudomodeFramework::simulate_material(
    const std::string& material,
    const SystemParams& system_params,
    const RealVector& omega_grid,
    const RealVector& time_grid) {

    Timer timer("simulate_material");

    SimulationResult result;
    result.status = "started";

    try {
        std::cout << "\n=== Simulating " << material << " at " 
                  << system_params.temperature_K << "K ===" << std::endl;

        // Step 1: Generate spectral density
        RealVector omega = omega_grid.empty() ? create_default_omega_grid() : omega_grid;
        RealVector J_omega;

        try {
            J_omega = MaterialDatabase::build_spectral_density(omega, material);
            std::cout << "✓ Generated spectral density (max J = " 
                      << *std::max_element(J_omega.begin(), J_omega.end()) << ")" << std::endl;
        } catch (const std::exception& e) {
            result.status = "spectral_density_failed: " + std::string(e.what());
            return result;
        }

        // Step 2: Convert to correlation function
        RealVector times = time_grid.empty() ? create_default_time_grid() : time_grid;
        ComplexVector C_data;

        try {
            C_data = fitter_->spectrum_to_correlation(J_omega, omega, times, system_params.temperature_K);
            std::cout << "✓ Generated correlation function (C(0) = " 
                      << std::abs(C_data[0]) << ")" << std::endl;
        } catch (const std::exception& e) {
            result.status = "correlation_failed: " + std::string(e.what());
            return result;
        }

        // Step 3: Fit pseudomode parameters
        PronyFitter::FitResult fit_result;
        try {
            fit_result = fitter_->fit_correlation(C_data, times, system_params.temperature_K);

            if (!fit_result.converged) {
                result.status = "fitting_failed: " + fit_result.message;
                return result;
            }

            result.modes = fit_result.modes;
            result.fit_rmse = fit_result.rmse;
            result.fit_bic = fit_result.bic;

            std::cout << "✓ Fitted " << fit_result.modes.size() << " pseudomodes:" << std::endl;
            for (size_t i = 0; i < fit_result.modes.size(); ++i) {
                const auto& mode = fit_result.modes[i];
                std::cout << "   Mode " << (i+1) << ": ω=" << (mode.omega_eV*1000) 
                          << " meV, γ=" << (mode.gamma_eV*1000) 
                          << " meV, g=" << (mode.g_eV*1000) << " meV" << std::endl;
            }

        } catch (const std::exception& e) {
            result.status = "fitting_failed: " + std::string(e.what());
            return result;
        }

        // Step 4: Quantum dynamics simulation
        try {
            LindbladSolver solver(system_params, fit_result.modes, config_);

            // Create initial state
            QuantumState initial_state(2, fit_result.modes.size(), system_params.n_max);
            initial_state.set_initial_state("plus");

            std::cout << "✓ Created initial state |+⟩ ⊗ |vacuum⟩" << std::endl;

            // Time evolution
            auto evolution = solver.evolve(initial_state, times);

            std::cout << "✓ Completed quantum evolution over " 
                      << evolution.size() << " time points" << std::endl;

            // Extract coherence times
            result.coherence_times = solver.extract_coherence_times(evolution, times);

            std::cout << "✓ Extracted coherence times:" << std::endl;
            if (result.coherence_times.valid) {
                std::cout << "   T₂* = " << result.coherence_times.T2_star_ps << " ps" << std::endl;
            } else {
                std::cout << "   T₂* = could not be determined" << std::endl;
            }

        } catch (const std::exception& e) {
            result.status = "dynamics_failed: " + std::string(e.what());
            return result;
        }

        result.status = "completed_successfully";
        result.success = true;

        std::cout << "✓ Simulation completed successfully!" << std::endl;

    } catch (const std::exception& e) {
        result.status = "unexpected_error: " + std::string(e.what());
        std::cerr << "Simulation error: " << e.what() << std::endl;
    }

    return result;
}

std::vector<SimulationResult> PseudomodeFramework::batch_simulate(
    const std::vector<std::string>& materials,
    const std::vector<SystemParams>& systems) {

    if (materials.size() != systems.size()) {
        throw std::invalid_argument("Materials and systems size mismatch");
    }

    std::vector<SimulationResult> results(materials.size());

    std::cout << "Starting batch simulation of " << materials.size() 
              << " materials..." << std::endl;

    // Sequential execution (parallel can be added later)
    for (size_t i = 0; i < materials.size(); ++i) {
        std::cout << "\nSimulation " << (i+1) << "/" << materials.size() 
                  << ": " << materials[i] << std::endl;

        results[i] = simulate_material(materials[i], systems[i]);

        if (results[i].success) {
            std::cout << "✓ Completed: T₂* = " 
                      << results[i].coherence_times.T2_star_ps << " ps" << std::endl;
        } else {
            std::cout << "✗ Failed: " << results[i].status << std::endl;
        }
    }

    return results;
}

void PseudomodeFramework::export_results(
    const SimulationResult& result,
    const std::string& filename,
    const std::string& format) const {

    if (format == "json") {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        file << "{\n";
        file << "  \"status\": \"" << result.status << "\",\n";
        file << "  \"success\": " << (result.success ? "true" : "false") << ",\n";
        file << "  \"computation_time_s\": " << result.computation_time_s << ",\n";

        // Coherence times
        file << "  \"coherence_times\": {\n";
        file << "    \"T2_star_ps\": " << result.coherence_times.T2_star_ps << ",\n";
        file << "    \"T1_ps\": " << result.coherence_times.T1_ps << ",\n";
        file << "    \"valid\": " << (result.coherence_times.valid ? "true" : "false") << "\n";
        file << "  },\n";

        // Fitted modes
        file << "  \"fitted_modes\": [\n";
        for (size_t i = 0; i < result.modes.size(); ++i) {
            const auto& mode = result.modes[i];
            file << "    {\n";
            file << "      \"mode_id\": " << (i+1) << ",\n";
            file << "      \"omega_eV\": " << mode.omega_eV << ",\n";
            file << "      \"gamma_eV\": " << mode.gamma_eV << ",\n";
            file << "      \"g_eV\": " << mode.g_eV << ",\n";
            file << "      \"type\": \"" << mode.type << "\"\n";
            file << "    }";
            if (i < result.modes.size() - 1) file << ",";
            file << "\n";
        }
        file << "  ],\n";

        // Fit quality
        file << "  \"fit_quality\": {\n";
        file << "    \"rmse\": " << result.fit_rmse << ",\n";
        file << "    \"bic\": " << result.fit_bic << "\n";
        file << "  }\n";

        file << "}\n";

    } else if (format == "csv") {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        // Header
        file << "mode_id,omega_eV,gamma_eV,g_eV,type\n";

        // Mode data
        for (size_t i = 0; i < result.modes.size(); ++i) {
            const auto& mode = result.modes[i];
            file << (i+1) << "," << mode.omega_eV << "," << mode.gamma_eV 
                 << "," << mode.g_eV << "," << mode.type << "\n";
        }

        // Metadata as comments
        file << "\n# Coherence times:\n";
        file << "# T2_star_ps," << result.coherence_times.T2_star_ps << "\n";
        file << "# T1_ps," << result.coherence_times.T1_ps << "\n";
        file << "# RMSE," << result.fit_rmse << "\n";
        file << "# BIC," << result.fit_bic << "\n";

    } else {
        throw std::invalid_argument("Unknown format: " + format);
    }

    std::cout << "Results exported to " << filename << " (format: " << format << ")" << std::endl;
}

RealVector PseudomodeFramework::create_default_omega_grid() const {
    RealVector omega_grid;
    double omega_max = 0.15; // eV
    int n_points = 1000;

    for (int i = 0; i < n_points; ++i) {
        double omega = 0.001 + (omega_max - 0.001) * i / (n_points - 1);
        omega_grid.push_back(omega);
    }

    return omega_grid;
}

RealVector PseudomodeFramework::create_default_time_grid() const {
    RealVector time_grid;
    double t_max = config_.total_time_ps * 1e-12; // Convert to seconds
    int n_points = static_cast<int>(t_max / (config_.time_step_ps * 1e-12));

    for (int i = 0; i < n_points; ++i) {
        double t = i * config_.time_step_ps * 1e-12;
        time_grid.push_back(t);
    }

    return time_grid;
}

// Timer implementation
PseudomodeFramework::Timer::Timer(const std::string& name) 
    : name_(name), start_(std::chrono::high_resolution_clock::now()) {
}

PseudomodeFramework::Timer::~Timer() {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_);
    std::cout << "[Timer] " << name_ << ": " << duration.count() << " ms" << std::endl;
}

} // namespace PseudomodeFramework
