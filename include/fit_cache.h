/*
 * Fit Cache System - Accelerates Parameter Optimization
 * Apache License 2.0 - Copyright (c) 2025 Aetheron Research
 */

#pragma once
#include <unordered_map>
#include <vector>
#include <tuple>
#include <cstring>
#include <cmath>

namespace PseudomodeFramework {
namespace Fitting {

struct FitCacheKey {
    std::vector<double> u;     // Parameter vector in optimizer space
    std::vector<double> temps; // Temperature points

    bool operator==(const FitCacheKey& other) const {
        if (u.size() != other.u.size() || temps.size() != other.temps.size()) {
            return false;
        }

        // Compare with tolerance for numerical stability
        const double tol = 1e-3;

        for (size_t i = 0; i < u.size(); ++i) {
            if (std::abs(u[i] - other.u[i]) > tol) return false;
        }

        for (size_t i = 0; i < temps.size(); ++i) {
            if (std::abs(temps[i] - other.temps[i]) > tol) return false;
        }

        return true;
    }
};

struct FitCacheValue {
    std::vector<double> T2star_ps; // T2* values for each temperature
    std::vector<double> T1_ps;     // T1 values for each temperature
    bool success;                  // Whether simulation succeeded
    double computation_time_s;     // Time taken for computation
};

struct FitCacheHash {
    size_t operator()(const FitCacheKey& key) const noexcept {
        size_t hash = 1469598103934665603ULL;

        auto mix_double = [&hash](double x) {
            // Round to remove small numerical differences
            double rounded = std::round(x * 1000.0) / 1000.0;

            uint64_t bits;
            static_assert(sizeof(double) == 8, "Assumes 64-bit double");
            std::memcpy(&bits, &rounded, sizeof(double));

            hash ^= bits;
            hash *= 1099511628211ULL; // FNV prime
        };

        // Hash parameter vector
        for (double x : key.u) {
            mix_double(x);
        }

        // Hash temperature vector
        for (double t : key.temps) {
            mix_double(t);
        }

        return hash;
    }
};

class ObjectiveCache {
private:
    mutable std::unordered_map<FitCacheKey, FitCacheValue, FitCacheHash> cache_;
    mutable size_t hits_ = 0;
    mutable size_t misses_ = 0;

public:
    // Check if result exists in cache
    bool has_result(const FitCacheKey& key) const {
        return cache_.find(key) != cache_.end();
    }

    // Get cached result
    const FitCacheValue& get_result(const FitCacheKey& key) const {
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            hits_++;
            return it->second;
        }

        misses_++;
        throw std::runtime_error("Cache miss - result not found");
    }

    // Store result in cache
    void store_result(const FitCacheKey& key, const FitCacheValue& value) {
        cache_[key] = value;
    }

    // Clear cache
    void clear() {
        cache_.clear();
        hits_ = 0;
        misses_ = 0;
    }

    // Get cache statistics
    std::tuple<size_t, size_t, double> get_stats() const {
        size_t total = hits_ + misses_;
        double hit_rate = (total > 0) ? (double(hits_) / double(total)) : 0.0;
        return {hits_, misses_, hit_rate};
    }

    // Get cache size
    size_t size() const {
        return cache_.size();
    }
};

// Global cache instance
extern ObjectiveCache global_objective_cache;

} // namespace Fitting
} // namespace PseudomodeFramework
