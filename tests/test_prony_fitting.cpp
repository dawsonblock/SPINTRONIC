/**
 * @file test_prony_fitting.cpp
 * @brief Unit tests for Prony fitting algorithm
 * 
 * Tests the correlation function decomposition using Prony method.
 */

#include <gtest/gtest.h>
#include "../include/pseudomode_solver.h"
#include <cmath>
#include <random>

using namespace PseudomodeSolver;

class PronyFittingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create synthetic test data
        n_points = 1000;
        t_max = 10.0; // ps
        
        t_grid.resize(n_points);
        for (int i = 0; i < n_points; ++i) {
            t_grid[i] = i * t_max / (n_points - 1);
        }
    }
    
    // Generate synthetic correlation function
    std::vector<Complex> generate_test_correlation(
        const std::vector<PseudomodeParams>& true_modes
    ) {
        std::vector<Complex> C(t_grid.size());
        
        for (size_t i = 0; i < t_grid.size(); ++i) {
            Complex sum(0.0, 0.0);
            for (const auto& mode : true_modes) {
                Complex exponent = -Complex(mode.gamma_eV, mode.omega_eV) * t_grid[i];
                sum += mode.g_eV * mode.g_eV * std::exp(exponent);
            }
            C[i] = sum;
        }
        
        return C;
    }
    
    int n_points;
    double t_max;
    std::vector<double> t_grid;
};

/**
 * Test single-mode fitting
 */
TEST_F(PronyFittingTest, SingleModeFitting) {
    // True parameters
    PseudomodeParams true_mode;
    true_mode.omega_eV = 1.0;
    true_mode.gamma_eV = 0.1;
    true_mode.g_eV = 0.5;
    true_mode.mode_type = "test";
    
    std::vector<PseudomodeParams> true_modes = {true_mode};
    auto C_data = generate_test_correlation(true_modes);
    
    // Fit with max_modes = 1
    auto result = PronyFitter::fit_correlation(C_data, t_grid, 1, 300.0);
    
    // Check convergence
    EXPECT_TRUE(result.converged) << "Fitting did not converge: " << result.message;
    
    // Check number of modes
    EXPECT_EQ(result.modes.size(), 1);
    
    // Check fitted parameters (within 10% tolerance)
    if (result.modes.size() >= 1) {
        EXPECT_NEAR(result.modes[0].omega_eV, true_mode.omega_eV, 0.1);
        EXPECT_NEAR(result.modes[0].gamma_eV, true_mode.gamma_eV, 0.01);
        EXPECT_GT(result.modes[0].g_eV, 0.0);
    }
    
    // Check fit quality
    EXPECT_LT(result.rmse, 0.1);
}

/**
 * Test multi-mode fitting
 */
TEST_F(PronyFittingTest, MultiModeFitting) {
    // Two well-separated modes
    std::vector<PseudomodeParams> true_modes(2);
    
    true_modes[0].omega_eV = 0.5;
    true_modes[0].gamma_eV = 0.05;
    true_modes[0].g_eV = 0.3;
    
    true_modes[1].omega_eV = 2.0;
    true_modes[1].gamma_eV = 0.2;
    true_modes[1].g_eV = 0.4;
    
    auto C_data = generate_test_correlation(true_modes);
    
    // Fit with max_modes = 3 (should select 2)
    auto result = PronyFitter::fit_correlation(C_data, t_grid, 3, 300.0);
    
    EXPECT_TRUE(result.converged);
    EXPECT_GE(result.modes.size(), 2);
    EXPECT_LE(result.modes.size(), 3);
    
    // Should fit well
    EXPECT_LT(result.rmse, 0.2);
}

/**
 * Test noise robustness
 */
TEST_F(PronyFittingTest, NoiseRobustness) {
    PseudomodeParams true_mode;
    true_mode.omega_eV = 1.0;
    true_mode.gamma_eV = 0.1;
    true_mode.g_eV = 0.5;
    
    auto C_clean = generate_test_correlation({true_mode});
    
    // Add Gaussian noise
    std::vector<Complex> C_noisy = C_clean;
    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.01);
    
    for (auto& val : C_noisy) {
        val += Complex(noise(rng), noise(rng));
    }
    
    auto result = PronyFitter::fit_correlation(C_noisy, t_grid, 2, 300.0);
    
    // Should still converge with some noise
    EXPECT_TRUE(result.converged);
    EXPECT_GE(result.modes.size(), 1);
    
    // Parameters should be reasonable
    if (result.modes.size() >= 1) {
        EXPECT_NEAR(result.modes[0].omega_eV, 1.0, 0.3);
    }
}

/**
 * Test BIC model selection
 */
TEST_F(PronyFittingTest, ModelSelection) {
    // Single true mode
    PseudomodeParams true_mode;
    true_mode.omega_eV = 1.0;
    true_mode.gamma_eV = 0.1;
    true_mode.g_eV = 0.5;
    
    auto C_data = generate_test_correlation({true_mode});
    
    // Fit with different max_modes
    auto result_1 = PronyFitter::fit_correlation(C_data, t_grid, 1, 300.0);
    auto result_2 = PronyFitter::fit_correlation(C_data, t_grid, 2, 300.0);
    auto result_3 = PronyFitter::fit_correlation(C_data, t_grid, 3, 300.0);
    
    // BIC should prefer simpler model (1 mode)
    // Lower BIC is better
    EXPECT_LT(result_1.bic, result_2.bic);
    EXPECT_LT(result_1.bic, result_3.bic);
}

/**
 * Test parameter validation
 */
TEST_F(PronyFittingTest, ParameterValidation) {
    PseudomodeParams true_mode;
    true_mode.omega_eV = 1.0;
    true_mode.gamma_eV = 0.1;
    true_mode.g_eV = 0.5;
    
    auto C_data = generate_test_correlation({true_mode});
    auto result = PronyFitter::fit_correlation(C_data, t_grid, 2, 300.0);
    
    // All fitted modes should pass validation
    for (const auto& mode : result.modes) {
        EXPECT_TRUE(mode.is_valid()) << "Mode validation failed";
        EXPECT_GT(mode.gamma_eV, 0.0);
        EXPECT_GE(mode.g_eV, 0.0);
    }
}

/**
 * Test edge cases
 */
TEST_F(PronyFittingTest, EdgeCases) {
    // Empty data
    std::vector<Complex> empty_data;
    std::vector<double> empty_time;
    auto result_empty = PronyFitter::fit_correlation(empty_data, empty_time, 1, 300.0);
    EXPECT_FALSE(result_empty.converged);
    
    // Insufficient data points
    std::vector<Complex> short_data = {Complex(1.0, 0.0), Complex(0.5, 0.0)};
    std::vector<double> short_time = {0.0, 1.0};
    auto result_short = PronyFitter::fit_correlation(short_data, short_time, 2, 300.0);
    // Should handle gracefully
}

/**
 * Test temperature dependence
 */
TEST_F(PronyFittingTest, TemperatureDependence) {
    PseudomodeParams mode;
    mode.omega_eV = 1.0;
    mode.gamma_eV = 0.1;
    mode.g_eV = 0.5;
    
    auto C_data = generate_test_correlation({mode});
    
    // Fit at different temperatures
    auto result_low_T = PronyFitter::fit_correlation(C_data, t_grid, 2, 10.0);
    auto result_high_T = PronyFitter::fit_correlation(C_data, t_grid, 2, 1000.0);
    
    // Both should converge
    EXPECT_TRUE(result_low_T.converged);
    EXPECT_TRUE(result_high_T.converged);
    
    // Results may differ slightly due to temperature-dependent corrections
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
