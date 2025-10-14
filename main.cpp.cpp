/*
 * Main CLI Application for 2D Pseudomode Framework
 * Apache License 2.0 - Copyright (c) 2025 Aetheron Research
 */

#include "pseudomode_solver.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

using namespace PseudomodeSolver;

void print_usage() {
    std::cout << "2D Pseudomode Framework - Command Line Interface\n"
              << "Apache License 2.0 - Copyright (c) 2025 Aetheron Research\n\n"
              << "Usage: pseudomode_cli [options]\n\n"
              << "Options:\n"
              << "  --material <name>        Material name (MoS2, WSe2, graphene, GaN_2D)\n"
              << "  --omega0 <eV>           System frequency in eV\n" 
              << "  --temperature <K>        Temperature in Kelvin\n"
              << "  --max-modes <K>         Maximum number of pseudomodes\n"
              << "  --time-max <ps>         Maximum simulation time in ps\n"
              << "  --time-step <ps>        Time step in ps\n"
              << "  --coupling <op>         Coupling operator (sigma_x, sigma_y, sigma_z)\n"
              << "  --use-gpu               Enable CUDA GPU acceleration\n"
              << "  --output <file>         Output file (JSON format)\n"
              << "  --help                  Show this help message\n\n"
              << "Example:\n"
              << "  pseudomode_cli --material MoS2 --temperature 300 --max-modes 5 --use-gpu\n";
}

int main(int argc, char* argv[]) {
    // Default parameters
    std::string material = "MoS2";
    double omega0_eV = 1.8;
    double temperature_K = 300.0;
    int max_modes = 6;
    double time_max_ps = 100.0;
    double time_step_ps = 0.01;
    std::string coupling_operator = "sigma_z";
    bool use_gpu = false;
    std::string output_file = "pseudomode_results.json";

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help") {
            print_usage();
            return 0;
        } else if (arg == "--material" && i + 1 < argc) {
            material = argv[++i];
        } else if (arg == "--omega0" && i + 1 < argc) {
            omega0_eV = std::stod(argv[++i]);
        } else if (arg == "--temperature" && i + 1 < argc) {
            temperature_K = std::stod(argv[++i]);
        } else if (arg == "--max-modes" && i + 1 < argc) {
            max_modes = std::stoi(argv[++i]);
        } else if (arg == "--time-max" && i + 1 < argc) {
            time_max_ps = std::stod(argv[++i]);
        } else if (arg == "--time-step" && i + 1 < argc) {
            time_step_ps = std::stod(argv[++i]);
        } else if (arg == "--coupling" && i + 1 < argc) {
            coupling_operator = argv[++i];
        } else if (arg == "--use-gpu") {
            use_gpu = true;
        } else if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage();
            return 1;
        }
    }

    // Print configuration
    std::cout << "=== 2D Pseudomode Framework Simulation ===" << std::endl;
    std::cout << "Material: " << material << std::endl;
    std::cout << "System frequency: " << omega0_eV << " eV" << std::endl;
    std::cout << "Temperature: " << temperature_K << " K" << std::endl;
    std::cout << "Max pseudomodes: " << max_modes << std::endl;
    std::cout << "Simulation time: " << time_max_ps << " ps" << std::endl;
    std::cout << "Time step: " << time_step_ps << " ps" << std::endl;
    std::cout << "Coupling: " << coupling_operator << std::endl;
    std::cout << "GPU acceleration: " << (use_gpu ? "enabled" : "disabled") << std::endl;
    std::cout << "Output file: " << output_file << std::endl;
    std::cout << std::endl;

    try {
        // Set up simulation parameters
        System2DParams system_params;
        system_params.omega0_eV = omega0_eV;
        system_params.temperature_K = temperature_K;

        SimulationConfig config;
        config.max_pseudomodes = max_modes;
        config.total_time_ps = time_max_ps;
        config.time_step_ps = time_step_ps;
        config.coupling_operator = coupling_operator;
        config.use_gpu = use_gpu;

        // Create frequency and time grids
        std::vector<double> omega_grid;
        for (double w = 0.001; w <= 0.2; w += 0.0001) {
            omega_grid.push_back(w);
        }

        std::vector<double> time_grid;
        for (double t = 0.0; t <= time_max_ps; t += time_step_ps) {
            time_grid.push_back(t);
        }

        // Initialize framework
        PseudomodeFramework2D framework(config);

        // Run simulation
        std::cout << "Starting simulation..." << std::endl;
        auto result = framework.simulate_material(
            material, system_params, omega_grid, time_grid
        );

        // Print results
        std::cout << std::endl << "=== Results ===" << std::endl;
        std::cout << "Status: " << result.status << std::endl;
        std::cout << "Computation time: " << result.computation_time_seconds << " seconds" << std::endl;
        std::cout << "Fitted pseudomodes: " << result.fitted_modes.size() << std::endl;

        for (size_t k = 0; k < result.fitted_modes.size(); ++k) {
            const auto& mode = result.fitted_modes[k];
            std::cout << "  Mode " << (k+1) << ": ω=" << mode.omega_eV*1000 << " meV, "
                      << "γ=" << mode.gamma_eV*1000 << " meV, "
                      << "g=" << mode.g_eV*1000 << " meV" << std::endl;
        }

        std::cout << std::endl << "Coherence times:" << std::endl;
        std::cout << "  T₁ = " << result.coherence_times.T1_ps << " ps" << std::endl;
        std::cout << "  T₂* = " << result.coherence_times.T2_star_ps << " ps" << std::endl;
        std::cout << "  T₂ (echo) = " << result.coherence_times.T2_echo_ps << " ps" << std::endl;

        // Export results
        std::cout << std::endl << "Exporting results to " << output_file << "..." << std::endl;
        framework.export_results(result, output_file, "json");

        std::cout << "Simulation completed successfully!" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
