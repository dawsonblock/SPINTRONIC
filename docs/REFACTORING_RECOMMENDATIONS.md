# Function Refactoring Recommendations

**Generated:** 2025-10-14  
**Analysis:** Functions exceeding 100 lines identified for refactoring

## Overview

This document identifies functions in the codebase that exceed 100 lines and provides recommendations for refactoring them into smaller, more maintainable units following SOLID principles.

---

## Functions Requiring Refactoring

### 1. `AdvancedFitter::fit()` 
**File:** `src/advanced_fitting.cpp`  
**Lines:** 184 (starts at line 33)  
**Severity:** HIGH

**Current Structure:**
```cpp
EnhancedFitResult AdvancedFitter::fit() {
    // Line 33-217: Single monolithic function
    // - Parameter initialization
    // - Multiple optimization stages
    // - Cross-validation
    // - Result aggregation
}
```

**Recommended Refactoring:**
```cpp
// Break into smaller functions:

EnhancedFitResult AdvancedFitter::fit() {
    auto initial_params = initialize_parameters();
    auto optimized = run_optimization_stages(initial_params);
    auto validated = perform_cross_validation(optimized);
    return aggregate_results(validated);
}

private:
    FitParams initialize_parameters();
    FitParams run_optimization_stages(const FitParams& initial);
    FitParams perform_cross_validation(const FitParams& params);
    EnhancedFitResult aggregate_results(const FitParams& params);
```

**Benefits:**
- ✅ Each function has single responsibility
- ✅ Easier to unit test individual stages
- ✅ Improved readability
- ✅ Better error handling per stage

---

### 2. `AdvancedFitter::compute_simulation_loss()`
**File:** `src/advanced_fitting.cpp`  
**Lines:** 105 (starts at line 219)  
**Severity:** MEDIUM

**Current Structure:**
```cpp
double AdvancedFitter::compute_simulation_loss(
    const std::vector<double>& physical_params) {
    // Lines 219-324
    // - Parameter unpacking
    // - Simulation setup
    // - Time evolution
    // - Observable computation
    // - Error calculation
}
```

**Recommended Refactoring:**
```cpp
double AdvancedFitter::compute_simulation_loss(
    const std::vector<double>& physical_params) {
    
    auto system = setup_simulation_from_params(physical_params);
    auto evolution = run_time_evolution(system);
    auto observables = compute_observables(evolution);
    return calculate_loss_metric(observables);
}

private:
    SimulationSystem setup_simulation_from_params(
        const std::vector<double>& params);
    
    TimeEvolution run_time_evolution(const SimulationSystem& system);
    
    Observables compute_observables(const TimeEvolution& evolution);
    
    double calculate_loss_metric(const Observables& obs);
```

**Benefits:**
- ✅ Separation of concerns
- ✅ Reusable simulation components
- ✅ Testable loss calculation
- ✅ Clear data flow

---

### 3. `Optimization namespace` 
**File:** `src/lbfgs_optimizer.cpp`  
**Lines:** 125 (starts at line 14)  
**Severity:** MEDIUM

**Current Structure:**
```cpp
namespace Optimization {
    // Lines 14-139: Large namespace with multiple responsibilities
    // - L-BFGS algorithm implementation
    // - Line search
    // - Gradient computation
    // - Convergence checking
}
```

**Recommended Refactoring:**
```cpp
namespace Optimization {
    
    class LBFGSOptimizer {
    public:
        OptimizationResult optimize(
            const ObjectiveFunction& f,
            const Vector& initial_guess
        );
        
    private:
        class LineSearch {
            double find_step_size(/*...*/);
        };
        
        class ConvergenceChecker {
            bool is_converged(/*...*/);
        };
        
        Vector compute_search_direction(/*...*/);
        Vector compute_gradient(/*...*/);
        void update_history(/*...*/);
    };
}
```

**Benefits:**
- ✅ Encapsulation of optimization logic
- ✅ Modular line search strategies
- ✅ Pluggable convergence criteria
- ✅ Better testability

---

### 4. `main()` function
**File:** `src/main.cpp`  
**Lines:** 119 (starts at line 34)  
**Severity:** LOW (acceptable for entry point)

**Current Structure:**
```cpp
int main(int argc, char* argv[]) {
    // Lines 34-153
    // - Argument parsing
    // - Configuration loading
    // - Simulation setup
    // - Execution
    // - Results output
}
```

**Recommended Refactoring:**
```cpp
int main(int argc, char* argv[]) {
    try {
        auto config = parse_arguments(argc, argv);
        auto simulator = setup_simulator(config);
        auto results = run_simulation(simulator, config);
        export_results(results, config.output_path);
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        log_error(e.what());
        return EXIT_FAILURE;
    }
}

// Helper functions:
Config parse_arguments(int argc, char* argv[]);
Simulator setup_simulator(const Config& config);
Results run_simulation(const Simulator& sim, const Config& config);
void export_results(const Results& results, const std::string& path);
```

**Benefits:**
- ✅ Clear program flow
- ✅ Testable components
- ✅ Better error handling
- ✅ Reusable setup logic

---

### 5. `PYBIND11_MODULE()` 
**File:** `src/python_bindings.cpp`  
**Lines:** 143 (starts at line 17)  
**Severity:** LOW (acceptable for bindings)

**Current Structure:**
```cpp
PYBIND11_MODULE(pseudomode_py, m) {
    // Lines 17-160
    // - Class bindings
    // - Function exports
    // - Documentation strings
}
```

**Recommendation:**
This is acceptable for Python bindings. However, can be organized:

```cpp
PYBIND11_MODULE(pseudomode_py, m) {
    bind_config_classes(m);
    bind_spectral_density(m);
    bind_prony_fitter(m);
    bind_quantum_state(m);
    bind_framework(m);
}

// Separate file: python_bindings_config.cpp
void bind_config_classes(py::module& m) { /*...*/ }

// Separate file: python_bindings_spectral.cpp
void bind_spectral_density(py::module& m) { /*...*/ }

// etc.
```

**Benefits:**
- ✅ Modular binding organization
- ✅ Easier to maintain
- ✅ Parallel development possible

---

### 6. `main()` in scan_main.cpp
**File:** `src/scan_main.cpp`  
**Lines:** 197 (starts at line 77)  
**Severity:** MEDIUM

**Current Structure:**
```cpp
int main(int argc, char** argv) {
    // Lines 77-274
    // - Parameter grid generation
    // - Nested loops for parameter sweep
    // - Simulation execution
    // - Results aggregation
    // - Output formatting
}
```

**Recommended Refactoring:**
```cpp
int main(int argc, char** argv) {
    auto config = parse_scan_config(argc, argv);
    auto param_grid = generate_parameter_grid(config);
    auto results = execute_parameter_sweep(param_grid, config);
    export_scan_results(results, config.output_format);
    return EXIT_SUCCESS;
}

private:
    ParameterGrid generate_parameter_grid(const ScanConfig& config);
    
    SweepResults execute_parameter_sweep(
        const ParameterGrid& grid,
        const ScanConfig& config
    );
    
    void export_scan_results(
        const SweepResults& results,
        const std::string& format
    );
```

**Benefits:**
- ✅ Clear separation of concerns
- ✅ Parallel execution easier to implement
- ✅ Testable sweep logic
- ✅ Multiple export formats supported

---

## Refactoring Priority

### High Priority (Immediate)
1. ✅ `AdvancedFitter::fit()` - Core algorithm, heavily used
2. ✅ `AdvancedFitter::compute_simulation_loss()` - Performance critical

### Medium Priority (Next Sprint)
3. ⚠️ `Optimization namespace` - Affects convergence
4. ⚠️ `scan_main.cpp main()` - Used for batch jobs

### Low Priority (Nice to Have)
5. 📝 `main.cpp main()` - Acceptable for entry point
6. 📝 `PYBIND11_MODULE()` - Standard practice for bindings

---

## General Refactoring Principles

### 1. Single Responsibility Principle (SRP)
Each function should have one well-defined purpose.

**Bad:**
```cpp
void process_data() {
    // Load data
    // Validate data
    // Transform data
    // Save data
}
```

**Good:**
```cpp
Data load_data(const std::string& path);
void validate_data(const Data& data);
Data transform_data(const Data& data);
void save_data(const Data& data, const std::string& path);

void process_data(const std::string& input, const std::string& output) {
    auto data = load_data(input);
    validate_data(data);
    auto transformed = transform_data(data);
    save_data(transformed, output);
}
```

### 2. Extract Method Pattern
When a function is too long, extract logical blocks into named methods.

### 3. Cyclomatic Complexity
Target: Maximum 10 decision points per function

### 4. Line Count
Target: Maximum 50-75 lines per function  
Hard Limit: 100 lines

---

## Inline Documentation Examples

### Complex Algorithm Documentation

**Before:**
```cpp
void algorithm() {
    // Complex math here
    for (int i = 0; i < n; ++i) {
        x[i] = (a[i] + b[i]) / c[i];
    }
}
```

**After:**
```cpp
void algorithm() {
    /**
     * Compute normalized weighted average:
     * 
     * x_i = (a_i + b_i) / c_i
     * 
     * Where:
     * - a_i: acoustic phonon contribution
     * - b_i: optical phonon contribution  
     * - c_i: normalization factor
     * 
     * This implements Eq. (15) from Smith et al., PRB 2024.
     */
    for (int i = 0; i < n; ++i) {
        x[i] = (a[i] + b[i]) / c[i];
    }
}
```

### Physics-Based Comments

```cpp
/**
 * Compute pseudomode parameters via Prony decomposition.
 * 
 * Physical Interpretation:
 * Each pseudomode represents a collective bath degree of freedom
 * with effective frequency ω_k and decay rate γ_k. The coupling
 * strength g_k determines the system-bath interaction strength.
 * 
 * Mathematical Background:
 * We decompose the bath correlation function:
 * C(t) = ∑_k |g_k|² exp[-(γ_k + iω_k)t]
 * 
 * This is equivalent to replacing the continuum bath with K
 * discrete harmonic oscillators (pseudomodes).
 * 
 * Reference: Prior et al., PRL 105, 050404 (2010)
 */
std::vector<PseudomodeParams> fit_correlation_function(
    const std::vector<Complex>& C_t,
    const std::vector<double>& t_grid
) {
    // Implementation...
}
```

---

## Testing Requirements for Refactored Code

Each refactored function must have:

1. **Unit Tests**
   - Test each extracted function independently
   - Cover edge cases
   - Validate numerical accuracy

2. **Integration Tests**
   - Ensure refactored code produces identical results
   - Test with production data

3. **Performance Tests**
   - Benchmark before/after refactoring
   - Ensure no performance regression

---

## Checklist for Refactoring

- [ ] Function does one thing (SRP)
- [ ] Function name describes what it does
- [ ] Function is < 75 lines
- [ ] Cyclomatic complexity < 10
- [ ] Well-documented (Doxygen comments)
- [ ] Unit tests exist
- [ ] No side effects (pure function where possible)
- [ ] Error handling is clear
- [ ] No magic numbers (use named constants)

---

## Conclusion

Refactoring the identified long functions will:
- ✅ Improve code maintainability
- ✅ Enhance testability
- ✅ Reduce bug introduction risk
- ✅ Make code easier to understand for new developers
- ✅ Enable better parallelization opportunities

**Next Steps:**
1. Create refactoring tickets for high-priority functions
2. Write tests for existing behavior
3. Refactor incrementally
4. Run regression tests
5. Update documentation

---

**Document Status:** DRAFT  
**Reviewed By:** [Pending]  
**Approved By:** [Pending]
