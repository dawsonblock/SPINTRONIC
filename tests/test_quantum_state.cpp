/**
 * @file test_quantum_state.cpp
 * @brief Unit tests for quantum state representation and operations
 */

#include <gtest/gtest.h>
#include "../include/pseudomode_solver.h"
#include <cmath>

using namespace PseudomodeSolver;

class QuantumStateTest : public ::testing::Test {
protected:
    void SetUp() override {
        system_dim = 2;      // Qubit
        n_pseudomodes = 2;
        n_max = 3;
    }
    
    int system_dim;
    int n_pseudomodes;
    int n_max;
};

/**
 * Test state construction
 */
TEST_F(QuantumStateTest, Construction) {
    QuantumState state(system_dim, n_pseudomodes, n_max);
    
    // State should be created
    EXPECT_NO_THROW({
        auto tr = state.trace();
    });
}

/**
 * Test normalization
 */
TEST_F(QuantumStateTest, Normalization) {
    QuantumState state(system_dim, n_pseudomodes, n_max);
    state.set_initial_state("ground");
    state.normalize();
    
    auto tr = state.trace();
    EXPECT_NEAR(std::abs(tr), 1.0, 1e-10) << "Trace = " << tr;
}

/**
 * Test purity
 */
TEST_F(QuantumStateTest, Purity) {
    QuantumState state(system_dim, n_pseudomodes, n_max);
    state.set_initial_state("ground");
    state.normalize();
    
    double purity = state.purity();
    
    // Pure state should have Tr(ρ²) = 1
    EXPECT_GE(purity, 0.0);
    EXPECT_LE(purity, 1.0 + 1e-10);
}

/**
 * Test partial trace
 */
TEST_F(QuantumStateTest, PartialTrace) {
    QuantumState state(system_dim, n_pseudomodes, n_max);
    state.set_initial_state("ground");
    
    auto reduced_state = state.partial_trace_system();
    
    ASSERT_NE(reduced_state, nullptr);
    
    // Reduced state should be normalized
    auto tr = reduced_state->trace();
    EXPECT_NEAR(std::abs(tr), 1.0, 1e-8);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
