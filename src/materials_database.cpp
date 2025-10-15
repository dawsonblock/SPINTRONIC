/*
 * Materials Database for 2D Quantum Systems
 * Apache License 2.0 - Copyright (c) 2025 Aetheron Research
 * 
 * Comprehensive database of 2D materials with:
 * - Temperature-dependent phonon coupling
 * - Acoustic, flexural, and optical phonon contributions
 * - Custom material JSON import
 * - 13 validated 2D material systems
 */

#include "pseudomode_solver.h"
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>

namespace PseudomodeSolver {

// Internal material parameters structure
struct MaterialParams {
    std::string name;
    
    // Acoustic phonon parameters: J_ac = α_ac * ω * exp(-(ω/ω_c)^q)
    double alpha_ac;    // coupling strength
    double omega_c_ac;  // cutoff frequency (eV)
    double q_ac;        // decay exponent
    
    // Flexural phonon parameters: J_flex = α_flex * ω^s * exp(-(ω/ω_f)^q)
    double alpha_flex;  // coupling strength
    double omega_c_flex; // cutoff frequency (eV)
    double s_flex;      // power law exponent
    double q_flex;      // decay exponent
    
    // Optical phonon peaks (Lorentzian): J_opt = Σ_j (2λ_j^2 Γ_j) / ((ω-Ω_j)^2 + Γ_j^2)
    std::vector<double> omega_opt;   // peak frequencies (eV)
    std::vector<double> lambda_opt;  // coupling strengths (eV)
    std::vector<double> gamma_opt;   // linewidths (eV)
    
    // Temperature dependence
    double T_ref;       // reference temperature (K)
    double alpha_T;     // temperature coefficient for coupling (1/K)
    double gamma_T;     // temperature coefficient for linewidth (1/K)
    
    // Material properties
    double mass;        // effective mass (m_e)
    double lattice_const; // lattice constant (Å)
    double bandgap;     // band gap (eV)
};

// Materials database class
class MaterialsDatabase {
public:
    static MaterialParams get_material(const std::string& name);
    
private:
    // Transition Metal Dichalcogenides (TMDs)
    static MaterialParams get_MoS2();
    static MaterialParams get_WSe2();
    static MaterialParams get_WS2();
    static MaterialParams get_MoSe2();
    static MaterialParams get_MoTe2();
    static MaterialParams get_WSe2_multilayer();
    
    // Other 2D materials
    static MaterialParams get_graphene();
    static MaterialParams get_hBN();
    static MaterialParams get_GaN();
    static MaterialParams get_phosphorene();
    
    // Group-IV 2D materials
    static MaterialParams get_silicene();
    static MaterialParams get_germanene();
    static MaterialParams get_stanene();
};

// ============================================================================
// Material Definitions
// ============================================================================

MaterialParams MaterialsDatabase::get_MoS2() {
    MaterialParams p;
    p.name = "MoS2";
    
    // Acoustic phonons (LA + TA modes)
    p.alpha_ac = 0.01;
    p.omega_c_ac = 0.04;
    p.q_ac = 1.5;
    
    // Flexural phonons (ZA mode - out-of-plane)
    p.alpha_flex = 0.005;
    p.omega_c_flex = 0.02;
    p.s_flex = 0.3;
    p.q_flex = 2.0;
    
    // Optical phonons (A1' and E' modes)
    p.omega_opt = {0.048, 0.050};  // ~48-50 meV
    p.lambda_opt = {0.002, 0.0015};
    p.gamma_opt = {0.001, 0.0008};
    
    // Temperature dependence
    p.T_ref = 300.0;
    p.alpha_T = 0.0005;   // 0.05% per K
    p.gamma_T = 0.00001;  // linewidth broadening
    
    // Material properties
    p.mass = 0.5;         // effective mass in m_e units
    p.lattice_const = 3.16; // Å
    p.bandgap = 1.8;      // eV (direct gap at K point)
    
    return p;
}

MaterialParams MaterialsDatabase::get_WSe2() {
    MaterialParams p;
    p.name = "WSe2";
    
    // Stronger SOC than MoS2
    p.alpha_ac = 0.012;
    p.omega_c_ac = 0.035;
    p.q_ac = 1.5;
    
    p.alpha_flex = 0.008;
    p.omega_c_flex = 0.018;
    p.s_flex = 0.4;
    p.q_flex = 2.0;
    
    // Lower frequency optical modes
    p.omega_opt = {0.032, 0.035};  // ~32-35 meV
    p.lambda_opt = {0.003, 0.0025};
    p.gamma_opt = {0.0008, 0.0006};
    
    p.T_ref = 300.0;
    p.alpha_T = 0.0006;
    p.gamma_T = 0.000012;
    
    p.mass = 0.35;
    p.lattice_const = 3.28;
    p.bandgap = 1.65;
    
    return p;
}

MaterialParams MaterialsDatabase::get_WS2() {
    MaterialParams p;
    p.name = "WS2";
    
    p.alpha_ac = 0.011;
    p.omega_c_ac = 0.042;
    p.q_ac = 1.6;
    
    p.alpha_flex = 0.006;
    p.omega_c_flex = 0.022;
    p.s_flex = 0.35;
    p.q_flex = 2.0;
    
    p.omega_opt = {0.052, 0.054};  // Higher than MoS2
    p.lambda_opt = {0.0018, 0.0014};
    p.gamma_opt = {0.0009, 0.0007};
    
    p.T_ref = 300.0;
    p.alpha_T = 0.00055;
    p.gamma_T = 0.000011;
    
    p.mass = 0.4;
    p.lattice_const = 3.18;
    p.bandgap = 2.05;
    
    return p;
}

MaterialParams MaterialsDatabase::get_MoSe2() {
    MaterialParams p;
    p.name = "MoSe2";
    
    p.alpha_ac = 0.0095;
    p.omega_c_ac = 0.038;
    p.q_ac = 1.5;
    
    p.alpha_flex = 0.0055;
    p.omega_c_flex = 0.019;
    p.s_flex = 0.32;
    p.q_flex = 2.0;
    
    p.omega_opt = {0.030, 0.033};  // Lower than MoS2
    p.lambda_opt = {0.0022, 0.0017};
    p.gamma_opt = {0.00085, 0.00065};
    
    p.T_ref = 300.0;
    p.alpha_T = 0.00052;
    p.gamma_T = 0.0000105;
    
    p.mass = 0.55;
    p.lattice_const = 3.32;
    p.bandgap = 1.55;
    
    return p;
}

MaterialParams MaterialsDatabase::get_MoTe2() {
    MaterialParams p;
    p.name = "MoTe2";
    
    p.alpha_ac = 0.009;
    p.omega_c_ac = 0.030;
    p.q_ac = 1.4;
    
    p.alpha_flex = 0.007;
    p.omega_c_flex = 0.015;
    p.s_flex = 0.38;
    p.q_flex = 1.9;
    
    p.omega_opt = {0.024, 0.028};  // Lowest of TMDs
    p.lambda_opt = {0.0025, 0.002};
    p.gamma_opt = {0.0007, 0.0005};
    
    p.T_ref = 300.0;
    p.alpha_T = 0.0007;
    p.gamma_T = 0.000015;
    
    p.mass = 0.6;
    p.lattice_const = 3.52;
    p.bandgap = 1.1;  // Smallest gap TMD
    
    return p;
}

MaterialParams MaterialsDatabase::get_WSe2_multilayer() {
    MaterialParams p;
    p.name = "WSe2_multilayer";
    
    // Stronger interlayer coupling
    p.alpha_ac = 0.015;
    p.omega_c_ac = 0.040;
    p.q_ac = 1.6;
    
    p.alpha_flex = 0.012;  // Reduced flexural (stiffening)
    p.omega_c_flex = 0.025;
    p.s_flex = 0.5;
    p.q_flex = 2.2;
    
    p.omega_opt = {0.032, 0.035, 0.038};  // Additional interlayer mode
    p.lambda_opt = {0.003, 0.0025, 0.0015};
    p.gamma_opt = {0.0008, 0.0006, 0.001};
    
    p.T_ref = 300.0;
    p.alpha_T = 0.0004;  // Less temperature sensitive
    p.gamma_T = 0.00001;
    
    p.mass = 0.3;
    p.lattice_const = 3.28;
    p.bandgap = 1.4;  // Indirect gap for multilayer
    
    return p;
}

MaterialParams MaterialsDatabase::get_graphene() {
    MaterialParams p;
    p.name = "graphene";
    
    // Weak electron-phonon coupling
    p.alpha_ac = 0.008;
    p.omega_c_ac = 0.15;  // High cutoff
    p.q_ac = 1.2;
    
    // Strong flexural coupling (no gap)
    p.alpha_flex = 0.015;
    p.omega_c_flex = 0.025;
    p.s_flex = 0.2;
    p.q_flex = 2.0;
    
    // Minimal optical coupling
    p.omega_opt = {0.196};  // G-band ~196 meV
    p.lambda_opt = {0.0001};
    p.gamma_opt = {0.002};
    
    p.T_ref = 300.0;
    p.alpha_T = 0.0002;  // Very stable
    p.gamma_T = 0.000005;
    
    p.mass = 0.0;  // Massless Dirac fermions
    p.lattice_const = 2.46;
    p.bandgap = 0.0;
    
    return p;
}

MaterialParams MaterialsDatabase::get_hBN() {
    MaterialParams p;
    p.name = "hBN";
    
    // Insulator with weak coupling
    p.alpha_ac = 0.005;
    p.omega_c_ac = 0.12;
    p.q_ac = 1.8;
    
    p.alpha_flex = 0.003;
    p.omega_c_flex = 0.03;
    p.s_flex = 0.4;
    p.q_flex = 2.1;
    
    // High-frequency optical modes
    p.omega_opt = {0.165, 0.172};  // ~165-172 meV
    p.lambda_opt = {0.0005, 0.0004};
    p.gamma_opt = {0.0015, 0.0012};
    
    p.T_ref = 300.0;
    p.alpha_T = 0.0001;
    p.gamma_T = 0.000003;
    
    p.mass = 0.8;
    p.lattice_const = 2.50;
    p.bandgap = 6.0;  // Wide bandgap
    
    return p;
}

MaterialParams MaterialsDatabase::get_GaN() {
    MaterialParams p;
    p.name = "GaN";
    
    // Hypothetical 2D GaN (wurtzite-derived)
    p.alpha_ac = 0.02;
    p.omega_c_ac = 0.08;
    p.q_ac = 1.8;
    
    p.alpha_flex = 0.003;
    p.omega_c_flex = 0.06;
    p.s_flex = 0.6;
    p.q_flex = 2.0;
    
    // High-frequency optical modes
    p.omega_opt = {0.092};  // E2 mode ~92 meV
    p.lambda_opt = {0.003};
    p.gamma_opt = {0.002};
    
    p.T_ref = 300.0;
    p.alpha_T = 0.0003;
    p.gamma_T = 0.000008;
    
    p.mass = 0.2;
    p.lattice_const = 3.19;
    p.bandgap = 3.4;
    
    return p;
}

MaterialParams MaterialsDatabase::get_phosphorene() {
    MaterialParams p;
    p.name = "phosphorene";
    
    // Black phosphorus monolayer (anisotropic)
    p.alpha_ac = 0.015;
    p.omega_c_ac = 0.045;
    p.q_ac = 1.6;
    
    // Strong flexural coupling (puckered structure)
    p.alpha_flex = 0.010;
    p.omega_c_flex = 0.020;
    p.s_flex = 0.45;
    p.q_flex = 1.8;
    
    p.omega_opt = {0.035, 0.045};  // A_g modes
    p.lambda_opt = {0.004, 0.003};
    p.gamma_opt = {0.0012, 0.001};
    
    p.T_ref = 300.0;
    p.alpha_T = 0.0008;  // Temperature sensitive
    p.gamma_T = 0.000018;
    
    p.mass = 0.15;  // Light effective mass
    p.lattice_const = 3.31;  // Averaged
    p.bandgap = 1.5;  // Direct gap
    
    return p;
}

MaterialParams MaterialsDatabase::get_silicene() {
    MaterialParams p;
    p.name = "silicene";
    
    // Buckled honeycomb lattice
    p.alpha_ac = 0.010;
    p.omega_c_ac = 0.08;
    p.q_ac = 1.4;
    
    p.alpha_flex = 0.008;
    p.omega_c_flex = 0.015;
    p.s_flex = 0.25;
    p.q_flex = 1.9;
    
    p.omega_opt = {0.064};  // E_2g mode
    p.lambda_opt = {0.0012};
    p.gamma_opt = {0.0015};
    
    p.T_ref = 300.0;
    p.alpha_T = 0.0004;
    p.gamma_T = 0.00001;
    
    p.mass = 0.1;  // Light (Si-based)
    p.lattice_const = 3.86;
    p.bandgap = 0.0;  // Zero gap with SOC
    
    return p;
}

MaterialParams MaterialsDatabase::get_germanene() {
    MaterialParams p;
    p.name = "germanene";
    
    // More buckled than silicene
    p.alpha_ac = 0.012;
    p.omega_c_ac = 0.06;
    p.q_ac = 1.3;
    
    p.alpha_flex = 0.010;
    p.omega_c_flex = 0.012;
    p.s_flex = 0.28;
    p.q_flex = 1.85;
    
    p.omega_opt = {0.045};
    p.lambda_opt = {0.0015};
    p.gamma_opt = {0.0018};
    
    p.T_ref = 300.0;
    p.alpha_T = 0.0005;
    p.gamma_T = 0.000012;
    
    p.mass = 0.12;
    p.lattice_const = 4.02;
    p.bandgap = 0.0;
    
    return p;
}

MaterialParams MaterialsDatabase::get_stanene() {
    MaterialParams p;
    p.name = "stanene";
    
    // Strong SOC (topological insulator)
    p.alpha_ac = 0.014;
    p.omega_c_ac = 0.05;
    p.q_ac = 1.25;
    
    p.alpha_flex = 0.012;
    p.omega_c_flex = 0.010;
    p.s_flex = 0.30;
    p.q_flex = 1.8;
    
    p.omega_opt = {0.030};
    p.lambda_opt = {0.0020};
    p.gamma_opt = {0.0020};
    
    p.T_ref = 300.0;
    p.alpha_T = 0.0006;
    p.gamma_T = 0.000015;
    
    p.mass = 0.15;
    p.lattice_const = 4.67;
    p.bandgap = 0.1;  // Small gap from SOC
    
    return p;
}

// ============================================================================
// Database lookup
// ============================================================================

MaterialParams MaterialsDatabase::get_material(const std::string& name) {
    if (name == "MoS2") return get_MoS2();
    if (name == "WSe2") return get_WSe2();
    if (name == "WS2") return get_WS2();
    if (name == "MoSe2") return get_MoSe2();
    if (name == "MoTe2") return get_MoTe2();
    if (name == "WSe2_multilayer") return get_WSe2_multilayer();
    if (name == "graphene") return get_graphene();
    if (name == "hBN") return get_hBN();
    if (name == "GaN") return get_GaN();
    if (name == "phosphorene") return get_phosphorene();
    if (name == "silicene") return get_silicene();
    if (name == "germanene") return get_germanene();
    if (name == "stanene") return get_stanene();
    
    throw std::invalid_argument("Unknown material: " + name + 
        "\nAvailable materials: MoS2, WSe2, WS2, MoSe2, MoTe2, WSe2_multilayer, "
        "graphene, hBN, GaN, phosphorene, silicene, germanene, stanene");
}

// ============================================================================
// Temperature-dependent spectral density
// ============================================================================

std::vector<double> SpectralDensity2D::build_material_spectrum_T(
    const std::vector<double>& omega,
    const std::string& material,
    double temperature_K) {
    
    MaterialParams params = MaterialsDatabase::get_material(material);
    
    // Temperature scaling factors
    double T_ratio = temperature_K / params.T_ref;
    double alpha_scale = 1.0 + params.alpha_T * (temperature_K - params.T_ref);
    double gamma_scale = 1.0 + params.gamma_T * (temperature_K - params.T_ref);
    
    // Scale parameters
    double alpha_ac_T = params.alpha_ac * alpha_scale;
    double alpha_flex_T = params.alpha_flex * alpha_scale;
    
    // Build components
    auto J_ac = acoustic(omega, alpha_ac_T, params.omega_c_ac, params.q_ac);
    auto J_flex = flexural(omega, alpha_flex_T, params.omega_c_flex, params.s_flex, params.q_flex);
    
    // Combine contributions
    std::vector<double> J_total(omega.size(), 0.0);
    
    // Combine acoustic and flexural in a single parallel loop
    #pragma omp parallel for
    for (size_t i = 0; i < omega.size(); ++i) {
        J_total[i] = J_ac[i] + J_flex[i];
    }

    // Accumulate all optical peaks safely without nested parallel regions
    if (!params.omega_opt.empty()) {
        // Precompute sum of optical contributions per omega (no nested parallel regions)
        std::vector<double> J_opt_sum(omega.size(), 0.0);
        for (size_t j = 0; j < params.omega_opt.size(); ++j) {
            double gamma_T = params.gamma_opt[j] * gamma_scale;
            auto J_opt = lorentzian_peak(omega, params.omega_opt[j], params.lambda_opt[j], gamma_T);
            // Accumulate serially per peak into J_opt_sum; this avoids data races
            for (size_t i = 0; i < omega.size(); ++i) {
                J_opt_sum[i] += J_opt[i];
            }
        }
        // Single parallel add to J_total
        #pragma omp parallel for
        for (size_t i = 0; i < omega.size(); ++i) {
            J_total[i] += J_opt_sum[i];
        }
    }
    
    return J_total;
}
// Simple JSON parser (no external dependencies) — improved numeric parsing and validation
bool SpectralDensity2D::load_material_from_json(
    const std::string& json_filename,
    std::string& material_name,
    std::unordered_map<std::string, double>& params,
    std::string* error_message) {

    std::ifstream file(json_filename);
    if (!file.is_open()) {
        if (error_message) {
            *error_message = "Failed to open file: " + json_filename;
        }
        return false;
    }

    params.clear();
    material_name.clear();
    std::string line;

    while (std::getline(file, line)) {
        // Remove whitespace
        line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
        if (line.empty() || line[0] == '#') continue;

        // Parse "material_name": "value"
        if (line.find("\"material_name\"") != std::string::npos) {
            size_t start = line.find(':');
            if (start != std::string::npos) {
                size_t quote1 = line.find('"', start + 1);
                size_t quote2 = (quote1 != std::string::npos) ? line.find('"', quote1 + 1) : std::string::npos;
                if (quote1 != std::string::npos && quote2 != std::string::npos) {
                    material_name = line.substr(quote1 + 1, quote2 - quote1 - 1);
                }
            }
            continue;
        }

        // Parse "key": value
        size_t colon = line.find(':');
        if (colon == std::string::npos) continue;

        size_t quote1 = line.find('"');
        size_t quote2 = (quote1 != std::string::npos) ? line.find('"', quote1 + 1) : std::string::npos;
        if (quote1 == std::string::npos || quote2 == std::string::npos || quote2 >= colon) continue;

        std::string key = line.substr(quote1 + 1, quote2 - quote1 - 1);
        if (key == "material_name") continue;

        // Extract numeric value substring
        std::string value_str = line.substr(colon + 1);
        // Strip trailing commas/brackets/braces
        value_str.erase(std::remove(value_str.begin(), value_str.end(), ','), value_str.end());
        value_str.erase(std::remove(value_str.begin(), value_str.end(), '}'), value_str.end());
        value_str.erase(std::remove(value_str.begin(), value_str.end(), ']'), value_str.end());

        // Robust numeric parsing: ensure full consumption
        const char* cstr = value_str.c_str();
        char* endptr = nullptr;
        errno = 0;
        double value = std::strtod(cstr, &endptr);
        bool ok = (errno == 0) && (endptr != cstr);
        // Ensure remaining chars are only whitespace
        while (ok && *endptr != '\0') {
            if (!std::isspace(static_cast<unsigned char>(*endptr))) {
                ok = false;
                break;
            }
            ++endptr;
        }

        if (ok) {
            params[key] = value;
        }
    }
    
    file.close();
    return !material_name.empty();
}

std::vector<double> SpectralDensity2D::build_custom_material_spectrum(
    const std::vector<double>& omega,
    const std::unordered_map<std::string, double>& params) {
    
    std::vector<double> J_total(omega.size(), 0.0);
    
    // Extract acoustic parameters
    double alpha_ac = params.count("alpha_ac") ? params.at("alpha_ac") : 0.01;
    double omega_c_ac = params.count("omega_c_ac") ? params.at("omega_c_ac") : 0.04;
    double q_ac = params.count("q_ac") ? params.at("q_ac") : 1.5;
    
    // Extract flexural parameters
    double alpha_flex = params.count("alpha_flex") ? params.at("alpha_flex") : 0.005;
    double omega_c_flex = params.count("omega_c_flex") ? params.at("omega_c_flex") : 0.02;
    double s_flex = params.count("s_flex") ? params.at("s_flex") : 0.3;
    double q_flex = params.count("q_flex") ? params.at("q_flex") : 2.0;
    
    // Build acoustic and flexural contributions
    auto J_ac = acoustic(omega, alpha_ac, omega_c_ac, q_ac);
    auto J_flex = flexural(omega, alpha_flex, omega_c_flex, s_flex, q_flex);
    
    #pragma omp parallel for
    for (size_t i = 0; i < omega.size(); ++i) {
        J_total[i] = J_ac[i] + J_flex[i];
    }
    
    // Add optical peaks if specified
    if (params.count("omega_opt_1")) {
        double omega_opt = params.at("omega_opt_1");
        double lambda_opt = params.count("lambda_opt_1") ? params.at("lambda_opt_1") : 0.002;
        double gamma_opt = params.count("gamma_opt_1") ? params.at("gamma_opt_1") : 0.001;
        
        auto J_opt = lorentzian_peak(omega, omega_opt, lambda_opt, gamma_opt);
        #pragma omp parallel for
        for (size_t i = 0; i < omega.size(); ++i) {
            J_total[i] += J_opt[i];
        }
    }
    
    if (params.count("omega_opt_2")) {
        double omega_opt = params.at("omega_opt_2");
        double lambda_opt = params.count("lambda_opt_2") ? params.at("lambda_opt_2") : 0.002;
        double gamma_opt = params.count("gamma_opt_2") ? params.at("gamma_opt_2") : 0.001;
        
        auto J_opt = lorentzian_peak(omega, omega_opt, lambda_opt, gamma_opt);
        #pragma omp parallel for
        for (size_t i = 0; i < omega.size(); ++i) {
            J_total[i] += J_opt[i];
        }
    }
    
    return J_total;
}

// ============================================================================
// Utility functions
// ============================================================================

std::vector<std::string> SpectralDensity2D::list_available_materials() {
    return {
        "MoS2", "WSe2", "WS2", "MoSe2", "MoTe2", "WSe2_multilayer",
        "graphene", "hBN", "GaN", "phosphorene",
        "silicene", "germanene", "stanene"
    };
}

std::unordered_map<std::string, double> SpectralDensity2D::get_material_properties(
    const std::string& material) {
    
    MaterialParams params = MaterialsDatabase::get_material(material);
    
    std::unordered_map<std::string, double> props;
    props["mass"] = params.mass;
    props["lattice_const"] = params.lattice_const;
    props["bandgap"] = params.bandgap;
    props["alpha_ac"] = params.alpha_ac;
    props["omega_c_ac"] = params.omega_c_ac;
    props["alpha_flex"] = params.alpha_flex;
    props["T_ref"] = params.T_ref;
    
    return props;
}

} // namespace PseudomodeSolver
