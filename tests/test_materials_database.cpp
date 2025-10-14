/*
 * Test Materials Database - Phase 5 Validation
 * Tests the extended materials database with temperature dependence
 */

#include "pseudomode_solver.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>

using namespace PseudomodeSolver;

void print_section(const std::string& title) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(70, '=') << "\n";
}

bool test_material_listing() {
    print_section("TEST 1: List Available Materials");
    
    auto materials = SpectralDensity2D::list_available_materials();
    
    std::cout << "Found " << materials.size() << " materials:\n";
    for (const auto& mat : materials) {
        std::cout << "  - " << mat << "\n";
    }
    
    bool passed = materials.size() >= 13;
    std::cout << "\n✓ Status: " << (passed ? "PASS" : "FAIL") << "\n";
    return passed;
}

bool test_material_properties() {
    print_section("TEST 2: Material Properties Retrieval");
    
    std::vector<std::string> test_materials = {"MoS2", "graphene", "hBN", "GaN"};
    bool all_passed = true;
    
    for (const auto& mat : test_materials) {
        try {
            auto props = SpectralDensity2D::get_material_properties(mat);
            
            std::cout << "\n" << mat << ":\n";
            std::cout << "  Effective mass: " << props["mass"] << " m_e\n";
            std::cout << "  Lattice constant: " << props["lattice_const"] << " Å\n";
            std::cout << "  Band gap: " << props["bandgap"] << " eV\n";
            std::cout << "  Acoustic coupling: " << props["alpha_ac"] << "\n";
            std::cout << "  Acoustic cutoff: " << props["omega_c_ac"] << " eV\n";
            
        } catch (const std::exception& e) {
            std::cout << "\n✗ Failed for " << mat << ": " << e.what() << "\n";
            all_passed = false;
        }
    }
    
    std::cout << "\n✓ Status: " << (all_passed ? "PASS" : "FAIL") << "\n";
    return all_passed;
}

bool test_temperature_dependence() {
    print_section("TEST 3: Temperature-Dependent Spectral Density");
    
    std::vector<double> omega(100);
    for (size_t i = 0; i < omega.size(); ++i) {
        omega[i] = 0.0001 + i * 0.001;  // 0.1 to 100 meV
    }
    
    std::string material = "MoS2";
    std::vector<double> temps = {77.0, 300.0, 500.0};  // K
    
    std::cout << "\nTesting " << material << " at different temperatures:\n";
    
    bool all_passed = true;
    double J_prev = 0.0;
    
    for (double T : temps) {
        try {
            auto J = SpectralDensity2D::build_material_spectrum_T(omega, material, T);
            
            // Check that spectral density is non-negative
            bool valid = true;
            double J_sum = 0.0;
            for (size_t i = 0; i < J.size(); ++i) {
                if (J[i] < 0.0) {
                    valid = false;
                    break;
                }
                J_sum += J[i];
            }
            
            std::cout << "  T = " << std::setw(5) << T << " K: ";
            std::cout << "Sum J(ω) = " << std::scientific << std::setprecision(4) << J_sum;
            std::cout << " (" << (valid ? "valid" : "INVALID") << ")\n";
            
            if (!valid) all_passed = false;
            J_prev = J_sum;
            
        } catch (const std::exception& e) {
            std::cout << "  ✗ Error at T=" << T << "K: " << e.what() << "\n";
            all_passed = false;
        }
    }
    
    std::cout << "\n✓ Status: " << (all_passed ? "PASS" : "FAIL") << "\n";
    return all_passed;
}

bool test_spectral_density_components() {
    print_section("TEST 4: Spectral Density Components");
    
    std::vector<double> omega(50);
    for (size_t i = 0; i < omega.size(); ++i) {
        omega[i] = 0.001 + i * 0.002;  // 1 to 100 meV
    }
    
    std::vector<std::string> materials = {"MoS2", "WSe2", "graphene", "hBN"};
    bool all_passed = true;
    
    std::cout << "\nSpectral density peak analysis:\n";
    std::cout << std::setw(15) << "Material" 
              << std::setw(15) << "Max J(ω)" 
              << std::setw(15) << "Peak ω (eV)\n";
    std::cout << std::string(45, '-') << "\n";
    
    for (const auto& mat : materials) {
        try {
            auto J = SpectralDensity2D::build_material_spectrum_T(omega, mat, 300.0);
            
            // Find peak
            double max_J = 0.0;
            size_t max_idx = 0;
            for (size_t i = 0; i < J.size(); ++i) {
                if (J[i] > max_J) {
                    max_J = J[i];
                    max_idx = i;
                }
            }
            
            std::cout << std::setw(15) << mat
                      << std::setw(15) << std::scientific << std::setprecision(4) << max_J
                      << std::setw(15) << std::fixed << std::setprecision(6) << omega[max_idx]
                      << "\n";
            
        } catch (const std::exception& e) {
            std::cout << std::setw(15) << mat << " ✗ ERROR: " << e.what() << "\n";
            all_passed = false;
        }
    }
    
    std::cout << "\n✓ Status: " << (all_passed ? "PASS" : "FAIL") << "\n";
    return all_passed;
}

bool test_all_materials() {
    print_section("TEST 5: All Materials Accessibility");
    
    auto materials = SpectralDensity2D::list_available_materials();
    
    std::vector<double> omega(20);
    for (size_t i = 0; i < omega.size(); ++i) {
        omega[i] = 0.01 + i * 0.005;
    }
    
    bool all_passed = true;
    int success_count = 0;
    
    std::cout << "\nTesting spectral density generation for all materials:\n";
    
    for (const auto& mat : materials) {
        try {
            auto J = SpectralDensity2D::build_material_spectrum_T(omega, mat, 300.0);
            
            // Verify non-negative
            bool valid = true;
            for (double val : J) {
                if (val < 0.0 || std::isnan(val) || std::isinf(val)) {
                    valid = false;
                    break;
                }
            }
            
            std::cout << "  " << std::setw(20) << mat << ": " 
                      << (valid ? "✓ PASS" : "✗ FAIL (invalid values)") << "\n";
            
            if (valid) success_count++;
            else all_passed = false;
            
        } catch (const std::exception& e) {
            std::cout << "  " << std::setw(20) << mat << ": ✗ FAIL (" << e.what() << ")\n";
            all_passed = false;
        }
    }
    
    std::cout << "\nSuccess rate: " << success_count << "/" << materials.size() << "\n";
    std::cout << "\n✓ Status: " << (all_passed ? "PASS" : "FAIL") << "\n";
    return all_passed;
}

bool test_custom_material() {
    print_section("TEST 6: Custom Material Parameters");
    
    std::vector<double> omega(30);
    for (size_t i = 0; i < omega.size(); ++i) {
        omega[i] = 0.005 + i * 0.003;
    }
    
    // Define custom material
    std::unordered_map<std::string, double> custom_params = {
        {"alpha_ac", 0.015},
        {"omega_c_ac", 0.05},
        {"q_ac", 1.6},
        {"alpha_flex", 0.008},
        {"omega_c_flex", 0.025},
        {"s_flex", 0.4},
        {"q_flex", 2.0},
        {"omega_opt_1", 0.040},
        {"lambda_opt_1", 0.003},
        {"gamma_opt_1", 0.001}
    };
    
    try {
        auto J = SpectralDensity2D::build_custom_material_spectrum(omega, custom_params);
        
        // Verify
        bool valid = true;
        double max_J = 0.0;
        for (double val : J) {
            if (val < 0.0 || std::isnan(val)) {
                valid = false;
                break;
            }
            if (val > max_J) max_J = val;
        }
        
        std::cout << "\nCustom material spectral density:\n";
        std::cout << "  Max J(ω): " << std::scientific << max_J << "\n";
        std::cout << "  Validity: " << (valid ? "✓ Valid" : "✗ Invalid") << "\n";
        
        std::cout << "\n✓ Status: " << (valid ? "PASS" : "FAIL") << "\n";
        return valid;
        
    } catch (const std::exception& e) {
        std::cout << "\n✗ ERROR: " << e.what() << "\n";
        std::cout << "\n✓ Status: FAIL\n";
        return false;
    }
}

int main() {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "SPINTRONIC QUANTUM FRAMEWORK - Phase 5 Materials Database Tests\n";
    std::cout << std::string(70, '=') << "\n";
    
    int passed = 0;
    int total = 6;
    
    if (test_material_listing()) passed++;
    if (test_material_properties()) passed++;
    if (test_temperature_dependence()) passed++;
    if (test_spectral_density_components()) passed++;
    if (test_all_materials()) passed++;
    if (test_custom_material()) passed++;
    
    print_section("FINAL RESULTS");
    std::cout << "\nTests passed: " << passed << "/" << total << "\n";
    std::cout << "Success rate: " << (100.0 * passed / total) << "%\n\n";
    
    if (passed == total) {
        std::cout << "🎉 ALL TESTS PASSED - Phase 5 Materials Database Complete!\n\n";
        return 0;
    } else {
        std::cout << "⚠️  Some tests failed - see details above\n\n";
        return 1;
    }
}
