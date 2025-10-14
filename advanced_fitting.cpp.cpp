/*
 * Advanced Fitting Framework Implementation
 * Apache License 2.0 - Copyright (c) 2025 Aetheron Research
 */

#include "advanced_fitting.h"
#include <fstream>
#include <chrono>
#include <algorithm>
#include <numeric>

#ifdef USE_JSON
#include <nlohmann/json.hpp>
using json = nlohmann::json;
#endif

namespace PseudomodeFramework {
namespace Fitting {

AdvancedFitter::AdvancedFitter(const EnhancedFitPack& fit_pack) 
    : pack_(fit_pack), rng_(42) {

    validate_fit_pack();
    initialize_framework();

    // Set default optimizer options
    optimizer_options_.max_evaluations = 1000;
    optimizer_options_.gradient_tolerance = 1e-4;
    optimizer_options_.parameter_tolerance = 1e-3;
    optimizer_options_.verbose = false;
}

EnhancedFitResult AdvancedFitter::fit() {
    auto start_time = std::chrono::high_resolution_clock::now();

    EnhancedFitResult result;
    result.success = false;

    try {
        if (verbose_) {
            std::cout << "=== Advanced Parameter Fitting Started ===" << std::endl;
            std::cout << "Material: " << pack_.material << " (" << pack_.dim << ")" << std::endl;
            std::cout << "Variables: " << pack_.variables.size() << std::endl;
            std::cout << "Targets: " << pack_.targets.size() << std::endl;
        }

        // Initialize parameters
        std::vector<double> initial_physical(pack_.variables.size());
        std::vector<double> lower_bounds(pack_.variables.size());
        std::vector<double> upper_bounds(pack_.variables.size());

        for (size_t i = 0; i < pack_.variables.size(); ++i) {
            const auto& var = pack_.variables[i];
            initial_physical[i] = var.init;
            lower_bounds[i] = var.lo;
            upper_bounds[i] = var.hi;
        }

        // Transform to optimizer space (log scale with sigmoid bounds)
        auto initial_optimizer = physical_to_optimizer(initial_physical);
        std::vector<double> opt_lower(initial_optimizer.size(), -5.0);
        std::vector<double> opt_upper(initial_optimizer.size(), 5.0);

        // Try to load checkpoint
        if (pack_.checkpoint_file && std::ifstream(*pack_.checkpoint_file)) {
            try {
                auto [checkpoint_params, checkpoint_loss] = load_checkpoint(*pack_.checkpoint_file);
                if (checkpoint_params.size() == initial_optimizer.size()) {
                    initial_optimizer = checkpoint_params;
                    if (verbose_) {
                        std::cout << "Loaded checkpoint with loss = " << checkpoint_loss << std::endl;
                    }
                }
            } catch (...) {
                if (verbose_) {
                    std::cout << "Warning: Could not load checkpoint, using initial guess" << std::endl;
                }
            }
        }

        // Clear cache if needed
        if (pack_.sim.cache_enabled) {
            global_objective_cache.clear();
        }

        // Define objective function with gradient
        auto objective_with_grad = [this](const std::vector<double>& optimizer_params, 
                                        std::vector<double>& grad) -> double {
            const double eps = 1e-5;

            // Convert to physical space
            auto physical_params = optimizer_to_physical(optimizer_params);

            // Compute function value
            double f = compute_simulation_loss(physical_params) + 
                      compute_regularization_loss(physical_params);

            // Compute gradient by finite differences
            grad.resize(optimizer_params.size());
            for (size_t i = 0; i < optimizer_params.size(); ++i) {
                auto params_plus = optimizer_params;
                params_plus[i] += eps;

                auto physical_plus = optimizer_to_physical(params_plus);
                double f_plus = compute_simulation_loss(physical_plus) + 
                               compute_regularization_loss(physical_plus);

                grad[i] = (f_plus - f) / eps;
            }

            return f;
        };

        // Run optimization
        Optimization::LBFGSBOptimizer optimizer(optimizer_options_);
        auto opt_result = optimizer.minimize(
            objective_with_grad, initial_optimizer, opt_lower, opt_upper
        );

        // Convert results back to physical space
        result.optimal_parameters = opt_result.x;
        result.optimal_physical_values = optimizer_to_physical(opt_result.x);
        result.final_loss = opt_result.f;
        result.iterations = opt_result.iterations;
        result.function_evaluations = opt_result.function_evaluations;
        result.gradient_norm = 0.0; // Would need to compute
        result.converged = opt_result.converged;
        result.success = opt_result.converged;
        result.message = opt_result.message;

        // Compute diagnostics
        auto predictions = predict(result.optimal_physical_values);
        result.residuals = compute_residuals(result.optimal_physical_values);

        // RMSE
        double sum_sq_residuals = 0.0;
        for (double r : result.residuals) {
            sum_sq_residuals += r * r;
        }
        result.rmse = std::sqrt(sum_sq_residuals / result.residuals.size());

        // MAE
        double sum_abs_residuals = 0.0;
        for (double r : result.residuals) {
            sum_abs_residuals += std::abs(r);
        }
        result.mae = sum_abs_residuals / result.residuals.size();

        // R-squared
        std::vector<double> observed;
        for (const auto& target : pack_.targets) {
            if (!target.validation) {
                observed.push_back(target.value);
            }
        }
        result.r_squared = compute_r_squared(predictions, observed);

        // Cache statistics
        if (pack_.sim.cache_enabled) {
            auto [hits, misses, hit_rate] = global_objective_cache.get_stats();
            result.cache_hit_rate = hit_rate;
            result.cache_size = global_objective_cache.size();
        }

        // Save checkpoint
        if (pack_.checkpoint_file) {
            save_checkpoint(*pack_.checkpoint_file, result.optimal_parameters, result.final_loss);
        }

        // Timing
        auto end_time = std::chrono::high_resolution_clock::now();
        result.computation_time_s = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();

        if (verbose_) {
            std::cout << "\n=== Fitting Completed ===" << std::endl;
            std::cout << "Status: " << result.message << std::endl;
            std::cout << "Final loss: " << result.final_loss << std::endl;
            std::cout << "RMSE: " << result.rmse << std::endl;
            std::cout << "R²: " << result.r_squared << std::endl;
            std::cout << "Iterations: " << result.iterations << std::endl;
            std::cout << "Time: " << result.computation_time_s << " s" << std::endl;
            if (pack_.sim.cache_enabled) {
                std::cout << "Cache hit rate: " << (result.cache_hit_rate * 100) << "%" << std::endl;
            }
        }

    } catch (const std::exception& e) {
        result.success = false;
        result.message = "Fitting failed: " + std::string(e.what());

        if (verbose_) {
            std::cerr << "Error: " << result.message << std::endl;
        }
    }

    return result;
}

double AdvancedFitter::compute_simulation_loss(const std::vector<double>& physical_params) {
    // Update material parameters
    update_material_parameters(physical_params);

    // Check cache first
    if (pack_.sim.cache_enabled) {
        FitCacheKey cache_key;
        cache_key.u = physical_params; // Use physical params as key
        cache_key.temps.reserve(pack_.targets.size());

        for (const auto& target : pack_.targets) {
            cache_key.temps.push_back(target.T_K);
        }

        if (global_objective_cache.has_result(cache_key)) {
            const auto& cached = global_objective_cache.get_result(cache_key);

            if (cached.success) {
                double loss = 0.0;
                for (size_t i = 0; i < pack_.targets.size(); ++i) {
                    const auto& target = pack_.targets[i];
                    double predicted = (target.metric == "T2star_ps") ? 
                                     cached.T2star_ps[i] : cached.T1_ps[i];

                    double residual = std::log(std::max(predicted, 1e-30)) - 
                                    std::log(std::max(target.value, 1e-30));

                    if (pack_.loss.type == "huber_log") {
                        double abs_residual = std::abs(residual);
                        if (abs_residual <= pack_.loss.delta) {
                            loss += target.weight * 0.5 * residual * residual;
                        } else {
                            loss += target.weight * pack_.loss.delta * (abs_residual - 0.5 * pack_.loss.delta);
                        }
                    } else {
                        loss += target.weight * 0.5 * residual * residual;
                    }
                }
                return loss;
            }
        }
    }

    // Run simulations
    FitCacheValue cache_value;
    cache_value.T2star_ps.reserve(pack_.targets.size());
    cache_value.T1_ps.reserve(pack_.targets.size());
    cache_value.success = true;

    double loss = 0.0;

    for (const auto& target : pack_.targets) {
        UnifiedLindbladSolver::SystemParams sys_params;
        sys_params.temperature_K = target.T_K;
        sys_params.n_max = pack_.sim.n_max;

        try {
            auto result = framework_->simulate_material_spec(material_spec_, sys_params);

            if (result.success) {
                double predicted = (target.metric == "T2star_ps") ? 
                                 result.coherence_times.T2_star_ps : 
                                 result.coherence_times.T1_ps;

                cache_value.T2star_ps.push_back(result.coherence_times.T2_star_ps);
                cache_value.T1_ps.push_back(result.coherence_times.T1_ps);

                // Compute loss contribution
                double residual = std::log(std::max(predicted, 1e-30)) - 
                                std::log(std::max(target.value, 1e-30));

                if (pack_.loss.type == "huber_log") {
                    double abs_residual = std::abs(residual);
                    if (abs_residual <= pack_.loss.delta) {
                        loss += target.weight * 0.5 * residual * residual;
                    } else {
                        loss += target.weight * pack_.loss.delta * (abs_residual - 0.5 * pack_.loss.delta);
                    }
                } else {
                    loss += target.weight * 0.5 * residual * residual;
                }
            } else {
                cache_value.success = false;
                return 1e9; // Large penalty for failed simulation
            }

        } catch (...) {
            cache_value.success = false;
            return 1e9;
        }
    }

    // Cache the result
    if (pack_.sim.cache_enabled && cache_value.success) {
        FitCacheKey cache_key;
        cache_key.u = physical_params;
        cache_key.temps.reserve(pack_.targets.size());
        for (const auto& target : pack_.targets) {
            cache_key.temps.push_back(target.T_K);
        }

        global_objective_cache.store_result(cache_key, cache_value);
    }

    return loss;
}

double AdvancedFitter::compute_regularization_loss(const std::vector<double>& physical_params) const {
    double reg_loss = 0.0;

    for (size_t i = 0; i < physical_params.size(); ++i) {
        const auto& var = pack_.variables[i];

        if (var.prior_mu && var.prior_lambda) {
            double log_param = std::log(std::max(physical_params[i], 1e-30));
            double diff = log_param - *var.prior_mu;
            reg_loss += *var.prior_lambda * diff * diff;
        }
    }

    return pack_.loss.regularization_weight * reg_loss;
}

std::vector<double> AdvancedFitter::physical_to_optimizer(const std::vector<double>& physical) const {
    std::vector<double> optimizer(physical.size());

    for (size_t i = 0; i < physical.size(); ++i) {
        const auto& var = pack_.variables[i];

        // Map to [0,1] then to unbounded space via inverse sigmoid
        double fraction = (physical[i] - var.lo) / (var.hi - var.lo);
        fraction = std::max(0.001, std::min(0.999, fraction)); // Avoid infinities

        optimizer[i] = std::log(fraction / (1.0 - fraction));
    }

    return optimizer;
}

std::vector<double> AdvancedFitter::optimizer_to_physical(const std::vector<double>& optimizer) const {
    std::vector<double> physical(optimizer.size());

    for (size_t i = 0; i < optimizer.size(); ++i) {
        const auto& var = pack_.variables[i];

        // Sigmoid mapping from unbounded to [0,1] then to [lo,hi]
        double sigmoid = 1.0 / (1.0 + std::exp(-optimizer[i]));
        physical[i] = var.lo + sigmoid * (var.hi - var.lo);
    }

    return physical;
}

double AdvancedFitter::compute_r_squared(const std::vector<double>& predicted, const std::vector<double>& observed) const {
    if (predicted.size() != observed.size() || predicted.empty()) {
        return 0.0;
    }

    // Compute mean of observed values
    double mean_observed = std::accumulate(observed.begin(), observed.end(), 0.0) / observed.size();

    // Compute total sum of squares and residual sum of squares
    double ss_tot = 0.0, ss_res = 0.0;
    for (size_t i = 0; i < observed.size(); ++i) {
        ss_tot += (observed[i] - mean_observed) * (observed[i] - mean_observed);
        ss_res += (observed[i] - predicted[i]) * (observed[i] - predicted[i]);
    }

    return (ss_tot > 1e-12) ? (1.0 - ss_res / ss_tot) : 0.0;
}

void AdvancedFitter::initialize_framework() {
    // Setup simulation configuration
    SimulationConfig config;
    config.dim = (pack_.dim == "3D") ? Dimensionality::D3 : Dimensionality::D2;
    config.max_modes = pack_.sim.max_modes;
    config.use_gpu = pack_.sim.use_gpu;
    config.materials_json = pack_.materials_json;

    // Parse channels
    for (const auto& channel : pack_.channels) {
        if (channel == "dp") config.channels.dp = true;
        else if (channel == "pe") config.channels.pe = true;
        else if (channel == "polar") config.channels.polar = true;
        else if (channel == "raman") config.channels.raman = true;
        else if (channel == "orbach") config.channels.orbach = true;
        else if (channel == "acoustic") config.channels.acoustic_2d = true;
        else if (channel == "flexural") config.channels.flexural_2d = true;
        else if (channel == "optical") config.channels.optical_2d = true;
    }

    channels_ = config.channels;

    // Load material specification
    material_spec_.dim = config.dim;
    material_spec_.name = pack_.material;

    if (config.dim == Dimensionality::D3) {
        material_spec_ = MaterialDatabase::load_material(pack_.material, config.dim, pack_.materials_json);
    }

    // Initialize framework
    framework_ = std::make_unique<ExtendedPseudomodeFramework>(config);
}

void AdvancedFitter::validate_fit_pack() {
    if (pack_.variables.empty()) {
        throw std::invalid_argument("No variables specified for fitting");
    }

    if (pack_.targets.empty()) {
        throw std::invalid_argument("No targets specified for fitting");
    }

    // Validate bounds
    for (const auto& var : pack_.variables) {
        if (var.hi <= var.lo) {
            throw std::invalid_argument("Invalid bounds for variable: " + var.key);
        }
        if (var.init < var.lo || var.init > var.hi) {
            throw std::invalid_argument("Initial value outside bounds for variable: " + var.key);
        }
    }

    // Validate targets
    for (const auto& target : pack_.targets) {
        if (target.value <= 0) {
            throw std::invalid_argument("Target values must be positive");
        }
        if (target.T_K <= 0) {
            throw std::invalid_argument("Temperatures must be positive");
        }
        if (target.metric != "T2star_ps" && target.metric != "T1_ps") {
            throw std::invalid_argument("Unknown target metric: " + target.metric);
        }
    }
}

} // namespace Fitting
} // namespace PseudomodeFramework
