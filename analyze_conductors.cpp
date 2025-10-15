/*
 * Conductor/Metallic Material Analysis for Room Temperature
 * Analyzes materials for best conducting properties at 300K
 * Note: True room-temperature superconductivity remains an open physics problem
 */

#include "pseudomode_solver.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

using namespace PseudomodeSolver;

struct ConductorScore {
    std::string name;
    double bandgap;
    double effective_mass;
    double lattice_constant;
    double acoustic_coupling;
    double spectral_density_integral;
    double conductivity_score;
    double metallic_score;
    
    void print() const {
        std::cout << std::setw(20) << name 
                  << std::setw(12) << std::fixed << std::setprecision(3) << bandgap
                  << std::setw(12) << effective_mass
                  << std::setw(15) << std::scientific << std::setprecision(2) << spectral_density_integral
                  << std::setw(15) << std::fixed << std::setprecision(1) << conductivity_score
                  << std::setw(12) << std::setprecision(1) << metallic_score << "\n";
    }
};

ConductorScore analyze_conductor(const std::string& material_name, double temperature) {
    ConductorScore score;
    score.name = material_name;
    
    try {
        // Get material properties
        auto props = SpectralDensity2D::get_material_properties(material_name);
        score.bandgap = props["bandgap"];
        score.effective_mass = props["mass"];
        score.lattice_constant = props["lattice_const"];
        score.acoustic_coupling = props["alpha_ac"];
        
        // Generate spectral density
        std::vector<double> omega(200);
        for (size_t i = 0; i < omega.size(); ++i) {
            omega[i] = 0.0001 + i * 0.001;  // 0.1 to 200 meV
        }
        
        auto J = SpectralDensity2D::build_material_spectrum_T(omega, material_name, temperature);
        
        // Calculate spectral density integral
        score.spectral_density_integral = 0.0;
        for (size_t i = 0; i < J.size() - 1; ++i) {
            score.spectral_density_integral += J[i] * (omega[i+1] - omega[i]);
        }
        
        // Scoring for good conductor/metallic properties:
        // 1. Zero or near-zero bandgap (0 = best, metallic)
        // 2. Low effective mass (higher mobility)
        // 3. Good lattice structure
        // 4. Low phonon scattering (low acoustic coupling)
        
        // Bandgap score: 0 eV is best for conductors (50 points max)
        double bandgap_score = 0.0;
        if (score.bandgap <= 0.1) {
            bandgap_score = 50.0 * (1.0 - score.bandgap / 0.1);
        }
        
        // Effective mass score: lower is better (30 points max)
        // For 2D materials, very low mass means high mobility
        double mass_score = 0.0;
        if (score.effective_mass <= 0.01) {
            mass_score = 30.0;  // Graphene-like (massless Dirac fermions)
        } else if (score.effective_mass < 0.3) {
            mass_score = 30.0 * (1.0 - score.effective_mass / 0.3);
        }
        
        // Low acoustic coupling = less scattering (10 points max)
        double coupling_score = 10.0 / (1.0 + score.acoustic_coupling * 100);
        
        // Low spectral density = less decoherence (10 points max)
        double sd_score = 10.0 / (1.0 + score.spectral_density_integral * 100);
        
        score.conductivity_score = bandgap_score + mass_score + coupling_score + sd_score;
        
        // Metallic character score (0 bandgap + low mass)
        if (score.bandgap == 0.0 && score.effective_mass < 0.2) {
            score.metallic_score = 100.0;
        } else if (score.bandgap < 0.1) {
            score.metallic_score = 80.0 - score.effective_mass * 50;
        } else {
            score.metallic_score = std::max(0.0, 50.0 - score.bandgap * 10);
        }
        
    } catch (const std::exception& e) {
        score.bandgap = -1;
        score.effective_mass = -1;
        score.conductivity_score = 0;
        score.metallic_score = 0;
    }
    
    return score;
}

int main() {
    std::cout << "\n" << std::string(100, '=') << "\n";
    std::cout << "METALLIC/CONDUCTOR MATERIAL ANALYSIS FOR ROOM TEMPERATURE (300K)\n";
    std::cout << std::string(100, '=') << "\n\n";
    
    std::cout << "⚠️  IMPORTANT NOTE:\n";
    std::cout << "True room-temperature superconductivity (zero electrical resistance)\n";
    std::cout << "remains an unsolved problem in physics. This analysis identifies the\n";
    std::cout << "best CONDUCTORS (metallic materials) with highest conductivity at 300K.\n\n";
    
    double temperature = 300.0;  // Room temperature
    
    auto materials = SpectralDensity2D::list_available_materials();
    std::vector<ConductorScore> scores;
    
    std::cout << "Analyzing " << materials.size() << " materials...\n\n";
    
    for (const auto& mat : materials) {
        auto score = analyze_conductor(mat, temperature);
        if (score.bandgap >= 0) {  // Valid material
            scores.push_back(score);
        }
    }
    
    // Sort by conductivity score (descending)
    std::sort(scores.begin(), scores.end(), 
              [](const ConductorScore& a, const ConductorScore& b) {
                  return a.conductivity_score > b.conductivity_score;
              });
    
    // Print results
    std::cout << std::string(100, '-') << "\n";
    std::cout << std::setw(20) << "Material"
              << std::setw(12) << "Bandgap(eV)"
              << std::setw(12) << "Eff.Mass"
              << std::setw(15) << "Spectral Dens"
              << std::setw(15) << "Conduct.Score"
              << std::setw(12) << "Metal%" << "\n";
    std::cout << std::string(100, '-') << "\n";
    
    for (const auto& score : scores) {
        score.print();
    }
    
    std::cout << std::string(100, '-') << "\n\n";
    
    // Detailed analysis of top 3
    std::cout << "\n" << std::string(100, '=') << "\n";
    std::cout << "TOP 3 CONDUCTORS FOR ROOM TEMPERATURE OPERATION\n";
    std::cout << std::string(100, '=') << "\n";
    
    int rank = 1;
    for (size_t i = 0; i < std::min(size_t(3), scores.size()); ++i) {
        const auto& s = scores[i];
        auto props = SpectralDensity2D::get_material_properties(s.name);
        
        std::cout << "\n#" << rank++ << " - " << s.name << "\n";
        std::cout << std::string(80, '-') << "\n";
        std::cout << "  Bandgap:              " << s.bandgap << " eV ";
        if (s.bandgap == 0.0) std::cout << "(METALLIC ✓)";
        std::cout << "\n";
        std::cout << "  Effective mass:       " << s.effective_mass << " m_e ";
        if (s.effective_mass < 0.1) std::cout << "(HIGH MOBILITY ✓)";
        std::cout << "\n";
        std::cout << "  Lattice constant:     " << props["lattice_const"] << " Å\n";
        std::cout << "  Acoustic coupling:    " << props["alpha_ac"] << "\n";
        std::cout << "  Acoustic cutoff:      " << props["omega_c_ac"] << " eV\n";
        std::cout << "  Spectral density:     " << std::scientific << s.spectral_density_integral << "\n";
        std::cout << "  Conductivity score:   " << std::fixed << s.conductivity_score << " / 100\n";
        std::cout << "  Metallic character:   " << s.metallic_score << "%\n";
        
        // Calculate approximate conductivity metrics
        double mobility_factor = s.effective_mass > 0 ? 1.0 / s.effective_mass : 1000.0;
        std::cout << "  Mobility factor:      " << std::scientific << mobility_factor << "\n";
        
        // Application assessment
        std::cout << "\n  Properties:\n";
        if (s.bandgap == 0.0) {
            std::cout << "    ✓ Zero bandgap (metallic conductor)\n";
        }
        if (s.effective_mass < 0.1) {
            std::cout << "    ✓ Ultra-high carrier mobility\n";
        }
        if (s.bandgap == 0.0 && s.effective_mass < 0.1) {
            std::cout << "    ✓ Massless Dirac fermions (graphene-like)\n";
        }
        if (s.acoustic_coupling < 0.01) {
            std::cout << "    ✓ Low phonon scattering\n";
        }
        
        std::cout << "\n  Applications:\n";
        if (s.bandgap == 0.0) {
            std::cout << "    • Transparent conductors\n";
            std::cout << "    • High-frequency electronics\n";
            std::cout << "    • Interconnects and electrodes\n";
        }
        if (s.effective_mass < 0.1) {
            std::cout << "    • Ultrafast transistors\n";
            std::cout << "    • THz devices\n";
        }
        std::cout << "\n";
    }
    
    std::cout << "\n" << std::string(100, '=') << "\n";
    std::cout << "RECOMMENDATION\n";
    std::cout << std::string(100, '=') << "\n\n";
    
    if (!scores.empty()) {
        const auto& best = scores[0];
        std::cout << "🏆 BEST CONDUCTOR: " << best.name << "\n\n";
        
        if (best.bandgap == 0.0) {
            std::cout << "This material is a TRUE METAL (zero bandgap) with:\n";
            std::cout << "  • Metallic conductivity at all temperatures\n";
            std::cout << "  • High carrier mobility (effective mass = " << best.effective_mass << " m_e)\n";
            std::cout << "  • Excellent transport properties\n";
            
            if (best.effective_mass < 0.01) {
                std::cout << "\n✨ SPECIAL: Contains massless Dirac fermions!\n";
                std::cout << "  • Linear energy-momentum dispersion\n";
                std::cout << "  • Ballistic transport over micron scales\n";
                std::cout << "  • Record-breaking room temperature mobility\n";
            }
        } else {
            std::cout << "This material has the lowest bandgap (" << best.bandgap << " eV)\n";
            std::cout << "and best conductivity among available materials.\n";
        }
        
        std::cout << "\n⚠️  NOTE: For true SUPERCONDUCTIVITY (zero resistance),\n";
        std::cout << "    cryogenic cooling is still required with current materials.\n";
        std::cout << "    Room-temp superconductors remain a major physics challenge.\n";
    }
    
    std::cout << "\n" << std::string(100, '=') << "\n";
    std::cout << "SUPERCONDUCTOR CONTEXT\n";
    std::cout << std::string(100, '=') << "\n\n";
    std::cout << "Known superconducting materials (require cooling):\n";
    std::cout << "  • Conventional (BCS): Aluminum, Niobium, Lead (T_c < 20K)\n";
    std::cout << "  • High-Tc cuprates: YBa₂Cu₃O₇ (T_c ~90K)\n";
    std::cout << "  • High-pressure: LaH₁₀ (T_c ~250K at 170 GPa)\n";
    std::cout << "  • Room-temp claim: Still unverified/controversial\n\n";
    std::cout << "Among 2D materials in this database:\n";
    std::cout << "  • None are known superconductors at room temperature\n";
    std::cout << "  • Graphene: Best metallic conductor available\n";
    std::cout << "  • Some TMDs show superconductivity below 10K when doped\n\n";
    
    return 0;
}
