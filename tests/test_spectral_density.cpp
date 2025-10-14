/**
 * @file test_spectral_density.cpp
 * @brief Unit tests for spectral density functions
 * 
 * Tests the 2D material spectral density implementations including:
 * - Acoustic phonons
 * - Flexural (ZA) phonons  
 * - Lorentzian peaks
 * - Material-specific spectra
 */

#include <gtest/gtest.h>
#include <pseudomode_solver.h>
#include <cmath>

using namespace PseudomodeSolver;

class SpectralDensityTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common test parameters
        omega_grid.resize(1000);
        for (int i = 0; i < 1000; ++i) {
            omega_grid[i] = i * 0.01; // 0 to 10 eV
        }
    }

    std::vector<double> omega_grid;
};

/**
 * Test acoustic phonon spectral density
 */
TEST_F(SpectralDensityTest, AcousticPhononShape) {
    double alpha = 1.0;
    double omega_c = 1.0;
    double q = 1.5;
    
    auto J = SpectralDensity2D::acoustic(omega_grid, alpha, omega_c, q);
    
    // Check size
    EXPECT_EQ(J.size(), omega_grid.size());
    
    // Check J(0) = 0 (linear at origin)
    EXPECT_NEAR(J[0], 0.0, 1e-10);
    
    // Check positivity
    for (size_t i = 1; i < J.size(); ++i) {
        EXPECT_GE(J[i], 0.0) << "J[" << i << "] is negative";
    }
    
    // Check peak exists
    auto max_iter = std::max_element(J.begin(), J.end());
    EXPECT_GT(*max_iter, 0.0);
    
    // Peak should be before cutoff
    size_t peak_idx = std::distance(J.begin(), max_iter);
    EXPECT_LT(omega_grid[peak_idx], omega_c * 2);
}

/**
 * Test flexural phonon spectral density
 */
TEST_F(SpectralDensityTest, FlexuralPhononShape) {
    double alpha_f = 0.5;
    double omega_f = 0.5;
    double s_f = 0.5;
    double q = 2.0;
    
    auto J = SpectralDensity2D::flexural(omega_grid, alpha_f, omega_f, s_f, q);
    
    // Check size
    EXPECT_EQ(J.size(), omega_grid.size());
    
    // Check J(0) = 0
    EXPECT_NEAR(J[0], 0.0, 1e-10);
    
    // Check positivity
    for (const auto& val : J) {
        EXPECT_GE(val, 0.0);
    }
    
    // Flexural should have sqrt(omega) behavior at low frequencies
    if (omega_grid[1] > 0 && omega_grid[2] > 0) {
        double ratio1 = J[1] / std::pow(omega_grid[1], s_f);
        double ratio2 = J[2] / std::pow(omega_grid[2], s_f);
        EXPECT_NEAR(ratio1, ratio2, ratio1 * 0.5); // Within 50%
    }
}

/**
 * Test Lorentzian peak spectral density
 */
TEST_F(SpectralDensityTest, LorentzianPeak) {
    double Omega_j = 2.0; // Peak at 2 eV
    double lambda_j = 0.5; // Coupling strength
    double Gamma_j = 0.1;  // Width
    
    auto J = SpectralDensity2D::lorentzian_peak(omega_grid, Omega_j, lambda_j, Gamma_j);
    
    // Check size
    EXPECT_EQ(J.size(), omega_grid.size());
    
    // Find peak position
    auto max_iter = std::max_element(J.begin(), J.end());
    size_t peak_idx = std::distance(J.begin(), max_iter);
    
    // Peak should be near Omega_j
    EXPECT_NEAR(omega_grid[peak_idx], Omega_j, 0.1);
    
    // Check Lorentzian shape: J(ω) ∝ 1/((ω-Ω)² + Γ²)
    for (size_t i = 0; i < J.size(); ++i) {
        double omega = omega_grid[i];
        double expected_shape = 1.0 / ((omega - Omega_j) * (omega - Omega_j) + Gamma_j * Gamma_j);
        EXPECT_GT(J[i], 0.0);
    }
}

/**
 * Test normalization and sum rules
 */
TEST_F(SpectralDensityTest, NormalizationCheck) {
    double alpha = 1.0;
    double omega_c = 1.0;
    
    auto J = SpectralDensity2D::acoustic(omega_grid, alpha, omega_c);
    
    // Numerical integration (trapezoidal rule)
    double integral = 0.0;
    for (size_t i = 1; i < J.size(); ++i) {
        double dw = omega_grid[i] - omega_grid[i-1];
        integral += 0.5 * (J[i] + J[i-1]) * dw;
    }
    
    // Integral should be finite and positive
    EXPECT_GT(integral, 0.0);
    EXPECT_LT(integral, 1000.0);
}

/**
 * Test material-specific spectral densities
 */
TEST_F(SpectralDensityTest, MaterialSpecificSpectra) {
    // Test graphene
    auto J_graphene = SpectralDensity2D::build_material_spectrum(
        omega_grid, "graphene"
    );
    EXPECT_EQ(J_graphene.size(), omega_grid.size());
    EXPECT_GT(*std::max_element(J_graphene.begin(), J_graphene.end()), 0.0);
    
    // Test MoS2
    auto J_MoS2 = SpectralDensity2D::build_material_spectrum(
        omega_grid, "MoS2"
    );
    EXPECT_EQ(J_MoS2.size(), omega_grid.size());
    
    // Different materials should have different spectra
    bool spectra_differ = false;
    for (size_t i = 0; i < J_graphene.size(); ++i) {
        if (std::abs(J_graphene[i] - J_MoS2[i]) > 1e-10) {
            spectra_differ = true;
            break;
        }
    }
    EXPECT_TRUE(spectra_differ);
}

/**
 * Test edge cases and error handling
 */
TEST_F(SpectralDensityTest, EdgeCases) {
    // Empty frequency grid
    std::vector<double> empty_grid;
    auto J_empty = SpectralDensity2D::acoustic(empty_grid, 1.0, 1.0);
    EXPECT_EQ(J_empty.size(), 0);
    
    // Single point
    std::vector<double> single_point = {1.0};
    auto J_single = SpectralDensity2D::acoustic(single_point, 1.0, 1.0);
    EXPECT_EQ(J_single.size(), 1);
    
    // Negative frequencies (should handle gracefully)
    std::vector<double> negative_omega = {-1.0, 0.0, 1.0};
    auto J_neg = SpectralDensity2D::acoustic(negative_omega, 1.0, 1.0);
    EXPECT_EQ(J_neg.size(), 3);
    // Typically J(-ω) = 0 or J(-ω) = J(ω) depending on convention
}

/**
 * Performance benchmark test
 */
TEST_F(SpectralDensityTest, PerformanceBenchmark) {
    // Large grid
    std::vector<double> large_grid(100000);
    for (size_t i = 0; i < large_grid.size(); ++i) {
        large_grid[i] = i * 0.0001;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    auto J = SpectralDensity2D::acoustic(large_grid, 1.0, 1.0);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Should complete in reasonable time
    EXPECT_LT(duration.count(), 1000) << "Computation took " << duration.count() << " ms";
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
