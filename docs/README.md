# 2D Pseudomode Framework Documentation

## Overview

The 2D Pseudomode Framework is a high-performance C++/CUDA library for simulating non-Markovian quantum dynamics in 2D materials using the pseudomode method.

## Key Features

- **GPU Acceleration**: CUDA-accelerated kernels for large Hilbert spaces
- **Adaptive Truncation**: Physics-based Hilbert space reduction
- **Material Library**: Built-in spectral densities for common 2D materials
- **Python Bindings**: Easy-to-use Python interface via pybind11
- **Performance**: 10-100x speedup over QuTiP for large systems

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/pseudomode-framework.git
cd pseudomode-framework

# Build
mkdir build && cd build
cmake .. -DBUILD_PYTHON_BINDINGS=ON -DUSE_CUDA=ON
make -j$(nproc)
sudo make install
```

### Basic Usage (C++)

```cpp
#include <pseudomode_solver.h>

using namespace PseudomodeSolver;

// Configure simulation
SimulationConfig config;
config.max_pseudomodes = 5;
config.use_gpu = true;

// Create framework
PseudomodeFramework2D framework(config);

// Define system
System2DParams system;
system.omega0_eV = 1.4;
system.temperature_K = 300.0;

// Run simulation
auto result = framework.simulate_material(
    "graphene", system, omega_grid, time_grid
);

// Extract results
std::cout << "T1 = " << result.coherence_times.T1_ps << " ps\n";
```

### Basic Usage (Python)

```python
import pseudomode_py as pm
import numpy as np

# Configure
config = pm.SimulationConfig()
config.max_pseudomodes = 5
config.use_gpu = True

# Run simulation
framework = pm.PseudomodeFramework2D(config)
result = framework.simulate_material("graphene", system, omega, times)

print(f"T1 = {result.coherence_times.T1_ps} ps")
```

## Documentation

- [API Reference](api/index.html) - Doxygen-generated API docs
- [Tutorial Notebooks](../examples/) - Jupyter notebook tutorials
- [Security Audit](../SECURITY_AUDIT.md) - CUDA security analysis

## Directory Structure

```
pseudomode-framework/
├── src/                    # C++ source files
│   ├── spectral_density_2d.cpp
│   ├── prony_fitting.cpp
│   ├── quantum_state.cpp
│   ├── lindblad_evolution.cpp
│   └── cuda_kernels.cu
├── include/                # Header files
│   ├── pseudomode_solver.h
│   └── ...
├── tests/                  # Unit tests (Google Test)
│   ├── test_spectral_density.cpp
│   ├── test_prony_fitting.cpp
│   └── test_quantum_state.cpp
├── benchmarks/            # Performance benchmarks
│   └── benchmark_main.cpp
├── examples/              # Tutorial notebooks
│   ├── tutorial_basic_usage.ipynb
│   └── tutorial_advanced_features.ipynb
├── k8s/                   # Kubernetes deployment
│   ├── deployment.yaml
│   ├── gpu-deployment.yaml
│   └── ...
└── docs/                  # Documentation
    ├── Doxyfile.in
    └── README.md
```

## Building the Documentation

```bash
cd build
make docs
```

Documentation will be generated in `build/docs/html/`.

## Running Tests

```bash
cd build
ctest --output-on-failure
```

## Running Benchmarks

```bash
cd build
./benchmark_pseudomode --benchmark_filter=BM_*
```

## Kubernetes Deployment

Deploy to Kubernetes cluster:

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/gpu-deployment.yaml
kubectl apply -f k8s/ingress.yaml
```

For GPU nodes:
```bash
kubectl apply -f k8s/gpu-deployment.yaml
```

## Performance Tips

1. **Use GPU for large systems** (>6 pseudomodes, n_max > 5)
2. **Enable adaptive truncation** to reduce Hilbert space size
3. **Optimize time step** based on energy scales
4. **Use sparse matrices** for large Hamiltonians
5. **Batch simulations** for materials screening

## Troubleshooting

### CUDA Out of Memory

Reduce `max_pseudomodes` or `adaptive_n_max`:

```cpp
config.max_pseudomodes = 4;  // Reduce from 6
config.adaptive_n_max = 3;   // Reduce from 5
```

### Slow Convergence

Check spectral density and increase fitting tolerance:

```cpp
config.convergence_tol = 1e-6;  // Relax from 1e-8
```

### Build Errors

Ensure dependencies are installed:

```bash
# Ubuntu/Debian
sudo apt-get install libeigen3-dev libhdf5-dev

# Fedora/RHEL
sudo dnf install eigen3-devel hdf5-devel
```

## Citation

If you use this software in your research, please cite:

```bibtex
@software{pseudomode_framework_2025,
  title = {2D Pseudomode Framework},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/your-org/pseudomode-framework},
  version = {1.0.0}
}
```

## License

Apache License 2.0 - see LICENSE for details

## Support

- GitHub Issues: https://github.com/your-org/pseudomode-framework/issues
- Documentation: https://docs.pseudomode-framework.org
- Email: support@pseudomode-framework.org
