/*
 * High-Level Interface Implementation
 * Apache License 2.0 - Copyright (c) 2025 Aetheron Research
 */

#include "pseudomode_solver.h"
#include <chrono>
#include <thread>
#include <future>
#include <fstream>
#include <json/json.h>

namespace PseudomodeSolver {

PseudomodeFramework2D::PseudomodeFramework2D(const SimulationConfig& config)
    : config_(config) {

#ifdef USE_CUDA
    detect_cuda_capabilities();
#endif

    std::cout << "Initialized 2D Pseudomode Framework" << std::endl;
    std::cout << "  Max pseudomodes: " << config_.max_pseudomodes << std::endl;
    std::cout << "  Adaptive n_max: " << config_.adaptive_n_max << std::endl;
    std::cout << "  GPU acceleration: " << (config_.use_gpu ? "enabled" : "disabled") << std::endl;
}

PseudomodeFramework2D::SimulationResult PseudomodeFramework2D::simulate_material(
    const std::string& material_name,
    const System2DParams& system_params,
    const std::vector<double>& omega_grid,
    const std::vector<double>& time_grid) {

    auto start_time = std::chrono::high_resolution_clock::now();

    SimulationResult result;
    result.status = "started";

    try {
        std::cout << "Step 1: Generating spectral density for " << material_name << std::endl;

        // Generate material spectral density
        auto J_omega = SpectralDensity2D::build_material_spectrum(omega_grid, material_name);

        // Compute correlation function via FFT
        std::vector<Complex> C_data(time_grid.size());
        Utils::fft_correlation_to_spectrum(C_data, J_omega);

        std::cout << "Step 2: Fitting pseudomode decomposition" << std::endl;

        // Fit pseudomode decomposition
        auto fit_result = PronyFitter::fit_correlation(
            C_data, time_grid, config_.max_pseudomodes, system_params.temperature_K
        );

        if (!fit_result.converged) {
            result.status = "fit_failed: " + fit_result.message;
            return result;
        }

        result.fitted_modes = fit_result.modes;

        std::cout << "Step 3: Quantum dynamics simulation" << std::endl;

        // Adaptive truncation
        int adaptive_n_max = Utils::compute_adaptive_n_max(
            fit_result.modes, system_params.temperature_K
        );

        SimulationConfig adaptive_config = config_;
        adaptive_config.adaptive_n_max = std::min(adaptive_n_max, config_.adaptive_n_max);

        std::cout << "Using adaptive n_max = " << adaptive_config.adaptive_n_max << std::endl;

        // Set up Lindbladian evolution
        LindbladEvolution evolution(system_params, fit_result.modes, adaptive_config);

        // Initial state (coherent superposition)
        QuantumState initial_state(2, fit_result.modes.size(), adaptive_config.adaptive_n_max);
        initial_state.set_initial_state("plus");

        // Time evolution
        auto time_evolution = evolution.evolve(initial_state, time_grid);

        // Extract coherence times
        result.coherence_times = evolution.extract_coherence_times(time_evolution);
        result.time_evolution = std::move(time_evolution);

        result.status = "completed_successfully";

    } catch (const std::exception& e) {
        result.status = "error: " + std::string(e.what());
        std::cerr << "Simulation error: " << e.what() << std::endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    result.computation_time_seconds = duration.count() / 1000.0;

    return result;
}

std::vector<PseudomodeFramework2D::SimulationResult> 
PseudomodeFramework2D::batch_simulate(
    const std::vector<std::string>& materials,
    const std::vector<System2DParams>& systems,
    int n_parallel_jobs) {

    if (materials.size() != systems.size()) {
        throw std::invalid_argument("Materials and systems vectors must have same size");
    }

    // Auto-detect number of parallel jobs
    if (n_parallel_jobs <= 0) {
        n_parallel_jobs = std::thread::hardware_concurrency();
        if (n_parallel_jobs == 0) n_parallel_jobs = 4; // fallback
    }

    std::cout << "Starting batch simulation with " << n_parallel_jobs 
              << " parallel jobs" << std::endl;

    std::vector<SimulationResult> results(materials.size());
    std::vector<std::future<SimulationResult>> futures;

    // Default grids (could be made configurable)
    std::vector<double> omega_grid;
    for (double w = 0.001; w <= 0.2; w += 0.0001) {
        omega_grid.push_back(w);
    }

    std::vector<double> time_grid;
    for (double t = 0.0; t <= config_.total_time_ps; t += config_.time_step_ps) {
        time_grid.push_back(t);
    }

    // Launch async simulations
    for (size_t i = 0; i < materials.size(); ++i) {
        futures.push_back(std::async(
            std::launch::async,
            [this, &materials, &systems, &omega_grid, &time_grid](size_t idx) {
                return this->simulate_material(materials[idx], systems[idx], omega_grid, time_grid);
            },
            i
        ));
    }

    // Collect results
    for (size_t i = 0; i < futures.size(); ++i) {
        try {
            results[i] = futures[i].get();
            std::cout << "Completed simulation " << (i+1) << "/" << materials.size() 
                      << " (" << materials[i] << ")" << std::endl;
        } catch (const std::exception& e) {
            results[i].status = "async_error: " + std::string(e.what());
        }
    }

    return results;
}

void PseudomodeFramework2D::export_results(
    const SimulationResult& result,
    const std::string& filename,
    const std::string& format) {

    if (format == "json") {
        Json::Value root;

        // Basic information
        root["status"] = result.status;
        root["computation_time_seconds"] = result.computation_time_seconds;
        root["framework_version"] = "1.0.0-cpp";
        root["timestamp"] = std::time(nullptr);

        // Fitted modes
        Json::Value modes_json(Json::arrayValue);
        for (size_t k = 0; k < result.fitted_modes.size(); ++k) {
            Json::Value mode;
            mode["mode_id"] = static_cast<int>(k + 1);
            mode["omega_eV"] = result.fitted_modes[k].omega_eV;
            mode["gamma_eV"] = result.fitted_modes[k].gamma_eV;
            mode["g_eV"] = result.fitted_modes[k].g_eV;
            mode["mode_type"] = result.fitted_modes[k].mode_type;
            modes_json.append(mode);
        }
        root["pseudomodes"] = modes_json;

        // Coherence times
        Json::Value coherence;
        coherence["T1_ps"] = result.coherence_times.T1_ps;
        coherence["T2_star_ps"] = result.coherence_times.T2_star_ps;
        coherence["T2_echo_ps"] = result.coherence_times.T2_echo_ps;
        root["coherence_times"] = coherence;

        // Write to file
        std::ofstream file(filename);
        file << root;

    } else if (format == "csv") {
        std::ofstream file(filename);

        // CSV header
        file << "mode_id,omega_eV,gamma_eV,g_eV,mode_type\n";

        // Mode data
        for (size_t k = 0; k < result.fitted_modes.size(); ++k) {
            const auto& mode = result.fitted_modes[k];
            file << (k+1) << "," << mode.omega_eV << "," << mode.gamma_eV 
                 << "," << mode.g_eV << "," << mode.mode_type << "\n";
        }

        file << "\n# Coherence times:\n";
        file << "# T1_ps," << result.coherence_times.T1_ps << "\n";
        file << "# T2_star_ps," << result.coherence_times.T2_star_ps << "\n";
        file << "# T2_echo_ps," << result.coherence_times.T2_echo_ps << "\n";

    } else {
        throw std::invalid_argument("Unknown export format: " + format);
    }

    std::cout << "Results exported to " << filename << " (format: " << format << ")" << std::endl;
}

#ifdef USE_CUDA
void PseudomodeFramework2D::detect_cuda_capabilities() {
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess || device_count == 0) {
        cuda_available_ = false;
        std::cout << "CUDA not available or no devices found" << std::endl;
        return;
    }

    cuda_available_ = true;
    cuda_device_count_ = device_count;

    std::cout << "CUDA detected: " << device_count << " device(s)" << std::endl;

    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        std::cout << "  Device " << i << ": " << prop.name 
                  << " (CC " << prop.major << "." << prop.minor << ")" << std::endl;
        std::cout << "    Global memory: " << (prop.totalGlobalMem / 1024 / 1024) << " MB" << std::endl;
        std::cout << "    Shared memory per block: " << (prop.sharedMemPerBlock / 1024) << " KB" << std::endl;
        std::cout << "    Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    }
}
#endif

} // namespace PseudomodeSolver
