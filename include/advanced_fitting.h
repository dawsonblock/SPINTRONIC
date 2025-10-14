/*
 * Advanced Fitting Framework - Enterprise-Grade Parameter Optimization
 * Apache License 2.0 - Copyright (c) 2025 Aetheron Research
 */

#pragma once
#include "pseudomode_solver_extended.h"
#include "lbfgs_optimizer.h"
#include "fit_cache.h"
#include <random>
#include <optional>

namespace PseudomodeFramework {
namespace Fitting {

// Enhanced fit variable with regularization
struct EnhancedFitVariable {
    std::string key;               // JSONPath to parameter
    double lo, hi, init;          // Bounds and initial value
    std::optional<double> prior_mu;     // Prior mean (log space)
    std::optional<double> prior_lambda; // Prior precision (1/variance)
    std::string units;            // Physical units for reporting
    std::string description;      // Human-readable description
};

// Enhanced target with uncertainty
struct EnhancedFitTarget {
    std::string metric;           // "T2star_ps" or "T1_ps"
    double T_K;                  // Temperature
    double value;                // Target value
    double weight;               // Fitting weight
    std::optional<double> sigma; // Experimental uncertainty
    bool validation;             // True for validation set
    std::string source;          // Data source reference
};

// Enhanced simulation configuration
struct EnhancedSimConfig {
    int max_modes = 4;
    bool use_gpu = false;
    int n_max = 3;
    double abs_tol = 1e-6;
    double rel_tol = 1e-5;
    int omp_threads = 0;         // 0 = auto-detect
    double val_split = 0.0;      // Fraction for validation (0-0.5)
    bool cache_enabled = true;   // Enable objective caching
};

// Loss function configuration
struct LossConfig {
    std::string type = "huber_log";  // "huber_log", "mse_log", "robust_log"
    double delta = 0.1;              // Huber threshold
    double regularization_weight = 0.0; // L2 regularization strength
};

// Enhanced fit pack with v2 features
struct EnhancedFitPack {
    int version = 2;
    std::string material;
    std::string dim;             // "2D" or "3D"
    std::vector<std::string> channels;
    std::string materials_json;

    std::vector<EnhancedFitVariable> variables;
    std::vector<EnhancedFitTarget> targets;

    EnhancedSimConfig sim;
    LossConfig loss;

    std::string output_file;
    std::optional<std::string> checkpoint_file;
    std::optional<std::string> bootstrap_output;

    // Metadata
    std::string description;
    std::string created_by;
    std::string created_date;
};

// Fit result with comprehensive diagnostics
struct EnhancedFitResult {
    bool success = false;
    std::string message;

    // Optimization results
    std::vector<double> optimal_parameters;
    std::vector<double> optimal_physical_values;
    double final_loss = 0.0;
    double training_loss = 0.0;
    double validation_loss = 0.0;
    double regularization_loss = 0.0;

    // Convergence diagnostics
    int iterations = 0;
    int function_evaluations = 0;
    double gradient_norm = 0.0;
    double parameter_change = 0.0;
    bool converged = false;

    // Performance metrics
    double computation_time_s = 0.0;
    double cache_hit_rate = 0.0;
    size_t cache_size = 0;

    // Statistical analysis
    std::vector<double> residuals;
    double rmse = 0.0;
    double mae = 0.0;
    double r_squared = 0.0;

    // Bootstrap uncertainty (if requested)
    struct BootstrapStats {
        std::vector<double> parameter_means;
        std::vector<double> parameter_stds;
        std::vector<double> parameter_ci_lower; // 95% CI
        std::vector<double> parameter_ci_upper;
        int bootstrap_samples = 0;
    };
    std::optional<BootstrapStats> bootstrap;
};

// Advanced fitting engine
class AdvancedFitter {
public:
    // Constructor
    AdvancedFitter(const EnhancedFitPack& fit_pack);

    // Main fitting interface
    EnhancedFitResult fit();

    // Bootstrap uncertainty analysis
    EnhancedFitResult bootstrap_fit(int num_bootstrap = 100, int seed = 42);

    // Cross-validation
    std::vector<double> cross_validate(int k_folds = 5, int seed = 42);

    // Prediction and diagnostics
    std::vector<double> predict(const std::vector<double>& parameters);
    std::vector<double> compute_residuals(const std::vector<double>& parameters);
    double compute_loss(const std::vector<double>& parameters);

    // Parameter space exploration
    void parameter_sweep(
        const std::string& param1, int n1,
        const std::string& param2, int n2,
        const std::string& output_file
    );

    // Save/load checkpoint
    void save_checkpoint(const std::string& filename, const std::vector<double>& parameters, double loss);
    std::pair<std::vector<double>, double> load_checkpoint(const std::string& filename);

    // Configuration
    void set_optimizer_options(const Optimization::LBFGSOptions& options);
    void enable_verbose_output(bool verbose = true);

private:
    EnhancedFitPack pack_;
    Optimization::LBFGSOptions optimizer_options_;
    bool verbose_ = false;

    // Internal state
    MaterialSpec material_spec_;
    ChannelToggles channels_;
    std::unique_ptr<ExtendedPseudomodeFramework> framework_;
    mutable std::mt19937_64 rng_;

    // Helper methods
    void initialize_framework();
    void validate_fit_pack();

    // Parameter transformation (physical <-> optimizer space)
    std::vector<double> physical_to_optimizer(const std::vector<double>& physical) const;
    std::vector<double> optimizer_to_physical(const std::vector<double>& optimizer) const;

    // JSON parameter updates
    void update_material_parameters(const std::vector<double>& physical_params);

    // Objective function components
    double compute_simulation_loss(const std::vector<double>& physical_params);
    double compute_regularization_loss(const std::vector<double>& physical_params) const;

    // Statistical utilities
    double compute_r_squared(const std::vector<double>& predicted, const std::vector<double>& observed) const;
    void compute_confidence_intervals(
        const std::vector<std::vector<double>>& bootstrap_samples,
        std::vector<double>& ci_lower,
        std::vector<double>& ci_upper,
        double confidence_level = 0.95
    ) const;

    // Validation utilities
    std::vector<size_t> create_validation_split(double val_fraction, int seed) const;
    std::vector<std::vector<size_t>> create_cv_folds(int k, int seed) const;
};

// Utility functions for fit pack management
EnhancedFitPack load_enhanced_fitpack(const std::string& filename);
void save_enhanced_fitpack(const EnhancedFitPack& pack, const std::string& filename);
void validate_fitpack_schema(const std::string& filename, int expected_version = 2);

// Report generation
void generate_fit_report(const EnhancedFitResult& result, const EnhancedFitPack& pack, const std::string& output_file);

} // namespace Fitting
} // namespace PseudomodeFramework
