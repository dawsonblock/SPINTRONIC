/*
 * Python Bindings for 2D Pseudomode Framework
 * Apache License 2.0 - Copyright (c) 2025 Aetheron Research
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include "pseudomode_solver.h"

namespace py = pybind11;
using namespace PseudomodeSolver;

PYBIND11_MODULE(pseudomode_py, m) {
    m.doc() = "2D Non-Markovian Pseudomode Framework - C++/CUDA Backend";

    // Physical constants
    py::module_ constants = m.def_submodule("constants", "Physical constants");
    constants.attr("HBAR_EVS") = PhysicalConstants::HBAR_EVS;
    constants.attr("KB_EV") = PhysicalConstants::KB_EV;
    constants.attr("C_LIGHT") = PhysicalConstants::C_LIGHT;

    // PseudomodeParams struct
    py::class_<PseudomodeParams>(m, "PseudomodeParams")
        .def(py::init<>())
        .def_readwrite("omega_eV", &PseudomodeParams::omega_eV)
        .def_readwrite("gamma_eV", &PseudomodeParams::gamma_eV) 
        .def_readwrite("g_eV", &PseudomodeParams::g_eV)
        .def_readwrite("mode_type", &PseudomodeParams::mode_type)
        .def("is_valid", &PseudomodeParams::is_valid)
        .def("__repr__", [](const PseudomodeParams& p) {
            return "PseudomodeParams(ω=" + std::to_string(p.omega_eV) + 
                   " eV, γ=" + std::to_string(p.gamma_eV) +
                   " eV, g=" + std::to_string(p.g_eV) + " eV, type=" + p.mode_type + ")";
        });

    // System2DParams struct  
    py::class_<System2DParams>(m, "System2DParams")
        .def(py::init<>())
        .def_readwrite("omega0_eV", &System2DParams::omega0_eV)
        .def_readwrite("alpha_R_eV", &System2DParams::alpha_R_eV)
        .def_readwrite("beta_D_eV", &System2DParams::beta_D_eV)
        .def_readwrite("Delta_v_eV", &System2DParams::Delta_v_eV)
        .def_readwrite("temperature_K", &System2DParams::temperature_K);

    // SimulationConfig struct
    py::class_<SimulationConfig>(m, "SimulationConfig")
        .def(py::init<>())
        .def_readwrite("max_pseudomodes", &SimulationConfig::max_pseudomodes)
        .def_readwrite("adaptive_n_max", &SimulationConfig::adaptive_n_max)
        .def_readwrite("time_step_ps", &SimulationConfig::time_step_ps)
        .def_readwrite("total_time_ps", &SimulationConfig::total_time_ps)
        .def_readwrite("convergence_tol", &SimulationConfig::convergence_tol)
        .def_readwrite("use_gpu", &SimulationConfig::use_gpu)
        .def_readwrite("gpu_device_id", &SimulationConfig::gpu_device_id)
        .def_readwrite("coupling_operator", &SimulationConfig::coupling_operator);

    // SpectralDensity2D class
    py::class_<SpectralDensity2D>(m, "SpectralDensity2D")
        .def_static("acoustic", &SpectralDensity2D::acoustic,
                   "Acoustic phonon spectral density",
                   py::arg("omega"), py::arg("alpha"), py::arg("omega_c"), py::arg("q") = 1.5)
        .def_static("flexural", &SpectralDensity2D::flexural,
                   "Flexural (ZA) phonon spectral density", 
                   py::arg("omega"), py::arg("alpha_f"), py::arg("omega_f"),
                   py::arg("s_f") = 0.5, py::arg("q") = 2.0)
        .def_static("lorentzian_peak", &SpectralDensity2D::lorentzian_peak,
                   "Discrete vibronic/magnon peak",
                   py::arg("omega"), py::arg("Omega_j"), py::arg("lambda_j"), py::arg("Gamma_j"))
        .def_static("build_material_spectrum", &SpectralDensity2D::build_material_spectrum,
                   "Build material-specific 2D spectral density",
                   py::arg("omega"), py::arg("material"), 
                   py::arg("params") = std::unordered_map<std::string, double>{})
        // Phase 5: Extended materials database
        .def_static("build_material_spectrum_T", &SpectralDensity2D::build_material_spectrum_T,
                   "Build temperature-dependent material spectral density",
                   py::arg("omega"), py::arg("material"), py::arg("temperature_K"))
        .def_static("load_material_from_json", 
                   [](const std::string& json_filename, 
                      std::string& material_name,
                      std::unordered_map<std::string, double>& params) {
                       return SpectralDensity2D::load_material_from_json(
                           json_filename, material_name, params, nullptr);
                   },
                   "Load custom material from JSON file",
                   py::arg("json_filename"), py::arg("material_name"), py::arg("params"))
        .def_static("build_custom_material_spectrum", &SpectralDensity2D::build_custom_material_spectrum,
                   "Build spectral density from custom parameters",
                   py::arg("omega"), py::arg("params"))
        .def_static("list_available_materials", &SpectralDensity2D::list_available_materials,
                   "List all available materials in database")
        .def_static("get_material_properties", &SpectralDensity2D::get_material_properties,
                   "Get physical properties of a material",
                   py::arg("material"));

    // PronyFitter::FitResult
    py::class_<PronyFitter::FitResult>(m, "FitResult")
        .def(py::init<>())
        .def_readwrite("modes", &PronyFitter::FitResult::modes)
        .def_readwrite("rmse", &PronyFitter::FitResult::rmse)
        .def_readwrite("bic", &PronyFitter::FitResult::bic)
        .def_readwrite("converged", &PronyFitter::FitResult::converged)
        .def_readwrite("message", &PronyFitter::FitResult::message);

    // PronyFitter class
    py::class_<PronyFitter>(m, "PronyFitter")
        .def_static("fit_correlation", &PronyFitter::fit_correlation,
                   "Fit correlation function using Prony method + BIC selection",
                   py::arg("C_data"), py::arg("t_grid"), py::arg("max_modes"), py::arg("temperature_K"));

    // QuantumState class
    py::class_<QuantumState>(m, "QuantumState")
        .def(py::init<int, int, int>())
        .def("set_initial_state", &QuantumState::set_initial_state)
        .def("normalize", &QuantumState::normalize)
        .def("trace", &QuantumState::trace)
        .def("purity", &QuantumState::purity)
        .def("partial_trace_system", &QuantumState::partial_trace_system,
             py::return_value_policy::take_ownership);

    // CoherenceTimes struct
    py::class_<LindbladEvolution::CoherenceTimes>(m, "CoherenceTimes")
        .def(py::init<>())
        .def_readwrite("T1_ps", &LindbladEvolution::CoherenceTimes::T1_ps)
        .def_readwrite("T2_star_ps", &LindbladEvolution::CoherenceTimes::T2_star_ps)
        .def_readwrite("T2_echo_ps", &LindbladEvolution::CoherenceTimes::T2_echo_ps);

    // LindbladEvolution class
    py::class_<LindbladEvolution>(m, "LindbladEvolution")
        .def(py::init<const System2DParams&, const std::vector<PseudomodeParams>&, const SimulationConfig&>())
        .def("evolve", &LindbladEvolution::evolve,
             "Time evolution of quantum state",
             py::arg("initial_state"), py::arg("times"),
             py::return_value_policy::take_ownership)
        .def("extract_coherence_times", &LindbladEvolution::extract_coherence_times,
             "Extract T1, T2* from time evolution");

    // High-level interface
    py::class_<PseudomodeFramework2D::SimulationResult>(m, "SimulationResult")
        .def(py::init<>())
        .def_readwrite("fitted_modes", &PseudomodeFramework2D::SimulationResult::fitted_modes)
        .def_readwrite("coherence_times", &PseudomodeFramework2D::SimulationResult::coherence_times)
        .def_readwrite("computation_time_seconds", &PseudomodeFramework2D::SimulationResult::computation_time_seconds)
        .def_readwrite("status", &PseudomodeFramework2D::SimulationResult::status)
        .def("__repr__", [](const PseudomodeFramework2D::SimulationResult& r) {
            return "SimulationResult(modes=" + std::to_string(r.fitted_modes.size()) +
                   ", T1=" + std::to_string(r.coherence_times.T1_ps) + " ps" +
                   ", T2*=" + std::to_string(r.coherence_times.T2_star_ps) + " ps" +
                   ", status='" + r.status + "')";
        });

    py::class_<PseudomodeFramework2D>(m, "PseudomodeFramework2D")
        .def(py::init<const SimulationConfig&>())
        // Note: simulate_material returns vector of unique_ptr which can't be copied to Python
        // We only expose fitted_modes and coherence_times through SimulationResult
        // Users should use the result object, not direct time_evolution access
        .def("export_results", &PseudomodeFramework2D::export_results,
             "Export results to various formats",
             py::arg("result"), py::arg("filename"), py::arg("format") = "json");

    // Utility functions
    py::module_ utils = m.def_submodule("utils", "Utility functions");

    utils.def("compute_adaptive_n_max", &Utils::compute_adaptive_n_max,
              "Compute adaptive truncation based on occupation numbers",
              py::arg("modes"), py::arg("temperature_K"), py::arg("occupation_threshold") = 0.01);

    utils.def("estimate_memory_usage", &Utils::estimate_memory_usage,
              "Estimate memory usage in bytes",
              py::arg("system_dim"), py::arg("n_pseudomodes"), py::arg("n_max"));

    // Convenience functions for Phase 5 materials
    m.def("list_materials", &SpectralDensity2D::list_available_materials,
          "List all available 2D materials in database");
    
    m.def("material_info", &SpectralDensity2D::get_material_properties,
          "Get material properties (mass, lattice constant, band gap, etc.)",
          py::arg("material"));
    
    m.def("spectral_density", 
          [](const std::vector<double>& omega, const std::string& material, double T) {
              if (!std::isfinite(T) || T < 0.0) {
                  throw std::invalid_argument("temperature_K must be a finite, non-negative value");
              }
              return SpectralDensity2D::build_material_spectrum_T(omega, material, T);
          },
          "Compute temperature-dependent spectral density for a material",
          py::arg("omega"), py::arg("material"), py::arg("temperature_K") = 300.0);
    
    // Version information
    m.attr("__version__") = "1.0.0-phase5-7";
    m.attr("__phase__") = "Phase 5-7 Complete: Materials + CUDA + Bindings";
    m.attr("__cuda_available__") = 
#ifdef USE_CUDA
        true;
#else
        false;
#endif
    m.attr("__n_materials__") = 13;
}
