/*
 * Semiconductor Material Analysis for Room Temperature Operation
 * Analyzes all materials to find optimal semiconductor for 300K
 */

#include "pseudomode_solver.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

using namespace PseudomodeSolver;

struct MaterialScore {
    std::string name;
    double bandgap;
    double effective_mass;
    double spectral_density_integral;
    double coherence_metric;
    double overall_score;
    
    void print() const {
        std::cout << std::setw(20) << name 
                  << std::setw(12) << std::fixed << std::setprecision(3) << bandgap
                  << std::setw(12) << effective_mass
                  << std::setw(15) << std::scientific << std::setprecision(2) << spectral_density_integral
                  << std::setw(15) << std::fixed << std::setprecision(3) << coherence_metric
                  << std::setw(12) << std::setprecision(1) << overall_score << "\n";
    }
};

double calculate_coherence_metric(const std::vector<double>& J, const std::vector<double>& omega) {
    // Lower spectral density at low frequencies = better coherence
    // Focus on 0-50 meV range (relevant for quantum operations)
    double low_freq_integral = 0.0;
    int count = 0;
    
    for (size_t i = 0; i < omega.size() && omega[i] < 0.05; ++i) {
        low_freq_integral += J[i] * (omega[i+1] - omega[i]);
        count++;
    }
    
    return count > 0 ? low_freq_integral / count : 1e6;
}

MaterialScore analyze_material(const std::string& material_name, double temperature) {
    MaterialScore score;
    score.name = material_name;
    
    try {
        // Get material properties
        auto props = SpectralDensity2D::get_material_properties(material_name);
        score.bandgap = props["bandgap"];
        score.effective_mass = props["mass"];
        
        // Generate spectral density
        std::vector<double> omega(200);
        for (size_t i = 0; i < omega.size(); ++i) {
            omega[i] = 0.0001 + i * 0.001;  // 0.1 to 200 meV
        }
        
        auto J = SpectralDensity2D::build_material_spectrum_T(omega, material_name, temperature);
        
        // Calculate metrics
        score.spectral_density_integral = 0.0;
        for (size_t i = 0; i < J.size() - 1; ++i) {
            score.spectral_density_integral += J[i] * (omega[i+1] - omega[i]);
        }
        
        score.coherence_metric = calculate_coherence_metric(J, omega);
        
        // Scoring criteria for room temperature semiconductor:
        // 1. Bandgap: 1.0 - 3.5 eV optimal (0-40 points)
        // 2. Effective mass: 0.1 - 0.5 optimal (0-25 points)
        // 3. Low spectral density = low decoherence (0-25 points)
        // 4. Good coherence metric (0-10 points)
        
        double bandgap_score = 0.0;
        if (score.bandgap >= 1.0 && score.bandgap <= 3.5) {
            bandgap_score = 40.0;
        } else if (score.bandgap > 0.5 && score.bandgap < 5.0) {
            bandgap_score = 30.0 * std::max(0.0, 1.0 - std::abs(score.bandgap - 2.0) / 3.0);
        }
        
        double mass_score = 0.0;
        if (score.effective_mass >= 0.1 && score.effective_mass <= 0.5) {
            mass_score = 25.0;
        } else if (score.effective_mass > 0.0 && score.effective_mass < 1.0) {
            mass_score = 20.0 * std::max(0.0, 1.0 - std::abs(score.effective_mass - 0.3) / 0.7);
        }
        
        // Lower is better for spectral density
        double sd_score = 25.0 / (1.0 + score.spectral_density_integral * 100);
        
        // Lower coherence metric is better
        double coh_score = 10.0 / (1.0 + score.coherence_metric * 1000);
        
        score.overall_score = bandgap_score + mass_score + sd_score + coh_score;
        
    } catch (const std::exception& e) {
        score.bandgap = -1;
        score.effective_mass = -1;
        score.spectral_density_integral = -1;
        score.coherence_metric = -1;
        score.overall_score = 0;
    }
    
    return score;
}

int main() {
    std::cout << "\n" << std::string(100, '=') << "\n";
    std::cout << "SEMICONDUCTOR MATERIAL ANALYSIS FOR ROOM TEMPERATURE (300K)\n";
    std::cout << std::string(100, '=') << "\n\n";
    
    double temperature = 300.0;  // Room temperature
    
    auto materials = SpectralDensity2D::list_available_materials();
    std::vector<MaterialScore> scores;
    
    std::cout << "Analyzing " << materials.size() << " materials...\n\n";
    
    for (const auto& mat : materials) {
        auto score = analyze_material(mat, temperature);
        if (score.bandgap >= 0) {  // Valid material
            scores.push_back(score);
        }
    }
    
    // Sort by overall score (descending)
    std::sort(scores.begin(), scores.end(), 
              [](const MaterialScore& a, const MaterialScore& b) {
                  return a.overall_score > b.overall_score;
              });
    
    // Print results
    std::cout << std::string(100, '-') << "\n";
    std::cout << std::setw(20) << "Material"
              << std::setw(12) << "Bandgap(eV)"
              << std::setw(12) << "Eff.Mass"
              << std::setw(15) << "Spectral Dens"
              << std::setw(15) << "Coherence"
              << std::setw(12) << "Score" << "\n";
    std::cout << std::string(100, '-') << "\n";
    
    for (const auto& score : scores) {
        score.print();
    }
    
    std::cout << std::string(100, '-') << "\n\n";
    
    // Detailed analysis of top 3
    std::cout << "\n" << std::string(100, '=') << "\n";
    std::cout << "TOP 3 SEMICONDUCTORS FOR ROOM TEMPERATURE OPERATION\n";
    std::cout << std::string(100, '=') << "\n";
    
    int rank = 1;
    for (size_t i = 0; i < std::min(size_t(3), scores.size()); ++i) {
        const auto& s = scores[i];
        auto props = SpectralDensity2D::get_material_properties(s.name);
        
        std::cout << "\n#" << rank++ << " - " << s.name << "\n";
        std::cout << std::string(80, '-') << "\n";
        std::cout << "  Bandgap:              " << s.bandgap << " eV\n";
        std::cout << "  Effective mass:       " << s.effective_mass << " m_e\n";
        std::cout << "  Lattice constant:     " << props["lattice_const"] << " Å\n";
        std::cout << "  Acoustic coupling:    " << props["alpha_ac"] << "\n";
        std::cout << "  Acoustic cutoff:      " << props["omega_c_ac"] << " eV\n";
        std::cout << "  Spectral density:     " << std::scientific << s.spectral_density_integral << "\n";
        std::cout << "  Coherence metric:     " << std::fixed << s.coherence_metric << "\n";
        std::cout << "  Overall score:        " << s.overall_score << " / 100\n";
        
        // Application assessment
        std::cout << "\n  Applications:\n";
        if (s.bandgap >= 1.5 && s.bandgap <= 2.0) {
            std::cout << "    ✓ Solar cells (optimal bandgap)\n";
        }
        if (s.bandgap >= 2.0 && s.bandgap <= 4.0) {
            std::cout << "    ✓ Optoelectronics\n";
            std::cout << "    ✓ High-power devices\n";
        }
        if (s.effective_mass >= 0.1 && s.effective_mass <= 0.5) {
            std::cout << "    ✓ High-mobility transistors\n";
        }
        if (s.coherence_metric < 0.01) {
            std::cout << "    ✓ Quantum computing (good coherence)\n";
        }
        std::cout << "\n";
    }
    
    std::cout << "\n" << std::string(100, '=') << "\n";
    std::cout << "RECOMMENDATION\n";
    std::cout << std::string(100, '=') << "\n\n";
    
    if (!scores.empty()) {
        const auto& best = scores[0];
        std::cout << "🏆 BEST MATERIAL: " << best.name << "\n\n";
        std::cout << "This material offers the optimal combination of:\n";
        std::cout << "  • Appropriate bandgap for room temperature operation\n";
        std::cout << "  • Good carrier mobility (effective mass)\n";
        std::cout << "  • Low phonon-induced decoherence\n";
        std::cout << "  • Excellent coherence properties for quantum applications\n\n";
    }
    
    return 0;
}
