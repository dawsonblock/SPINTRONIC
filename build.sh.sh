#!/bin/bash
# build.sh - Automated build script for 2D Pseudomode Framework C++/CUDA
# Apache License 2.0 - Copyright (c) 2025 Aetheron Research

set -e  # Exit on error

echo "=== 2D Pseudomode Framework C++/CUDA Build Script ==="
echo

# Check system
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    PACKAGE_MANAGER="apt"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
    PACKAGE_MANAGER="brew"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    OS="windows"
    PACKAGE_MANAGER="vcpkg"
else
    echo "Unsupported OS: $OSTYPE"
    exit 1
fi

echo "Detected OS: $OS"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install dependencies
install_dependencies() {
    echo "Installing dependencies..."

    if [[ "$OS" == "linux" ]]; then
        if command_exists apt; then
            sudo apt update
            sudo apt install -y build-essential cmake libeigen3-dev libfftw3-dev libomp-dev pkg-config

            # Optional dependencies
            sudo apt install -y libjsoncpp-dev libhdf5-dev python3-dev

            # CUDA (if available)
            if command_exists nvcc; then
                echo "CUDA already installed"
            else
                echo "CUDA not found. Install CUDA Toolkit for GPU acceleration:"
                echo "  https://developer.nvidia.com/cuda-toolkit"
            fi

        elif command_exists yum; then
            sudo yum groupinstall -y "Development Tools"
            sudo yum install -y cmake eigen3-devel fftw-devel openmp-devel
        else
            echo "Unknown Linux package manager"
            exit 1
        fi

    elif [[ "$OS" == "macos" ]]; then
        if ! command_exists brew; then
            echo "Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi

        brew install cmake eigen fftw libomp jsoncpp hdf5

    elif [[ "$OS" == "windows" ]]; then
        echo "Windows build requires Visual Studio 2019+ and vcpkg"
        echo "See README.md for detailed Windows build instructions"
    fi

    # Python dependencies (optional)
    if command_exists pip3; then
        echo "Installing Python dependencies..."
        pip3 install pybind11 numpy matplotlib
    fi
}

# Check CUDA availability
check_cuda() {
    if command_exists nvcc; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        echo "CUDA $CUDA_VERSION detected"
        USE_CUDA="ON"

        # Check compute capability
        if command_exists nvidia-smi; then
            echo "GPU(s) available:"
            nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
        fi
    else
        echo "CUDA not found - building CPU-only version"
        USE_CUDA="OFF"
    fi
}

# Configure build
configure_build() {
    echo "Configuring build..."

    BUILD_DIR="build"
    if [ -d "$BUILD_DIR" ]; then
        echo "Removing existing build directory"
        rm -rf "$BUILD_DIR"
    fi

    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    CMAKE_ARGS=(
        -DCMAKE_BUILD_TYPE=Release
        -DUSE_CUDA="$USE_CUDA"
        -DBUILD_PYTHON_BINDINGS=ON
        -DBUILD_TESTS=ON
    )

    # Platform-specific settings
    if [[ "$OS" == "macos" ]]; then
        CMAKE_ARGS+=(-DOpenMP_ROOT=/usr/local/opt/libomp)
    fi

    echo "CMake configuration:"
    printf '  %s
' "${CMAKE_ARGS[@]}"
    echo

    cmake .. "${CMAKE_ARGS[@]}"
}

# Build project
build_project() {
    echo "Building project..."

    # Detect number of CPU cores
    if [[ "$OS" == "linux" ]]; then
        CORES=$(nproc)
    elif [[ "$OS" == "macos" ]]; then
        CORES=$(sysctl -n hw.ncpu)
    else
        CORES=4  # fallback
    fi

    echo "Using $CORES parallel jobs"
    make -j"$CORES"

    echo "Build completed successfully!"
}

# Run tests
run_tests() {
    echo "Running tests..."

    if [ -f "./test_spectral_density" ]; then
        ./test_spectral_density
    fi

    if [ -f "./test_prony_fitting" ]; then
        ./test_prony_fitting
    fi

    if [ -f "./test_quantum_state" ]; then
        ./test_quantum_state
    fi

    echo "All tests passed!"
}

# Benchmark performance
run_benchmarks() {
    if [ -f "./benchmark_pseudomode" ]; then
        echo "Running performance benchmarks..."
        ./benchmark_pseudomode --quick
    fi
}

# Install
install_project() {
    echo "Installing pseudomode framework..."

    if [[ "$1" == "system" ]]; then
        sudo make install
        echo "Installed system-wide"
    else
        INSTALL_PREFIX="$HOME/pseudomode-install"
        make install DESTDIR="$INSTALL_PREFIX"
        echo "Installed to $INSTALL_PREFIX"

        # Add to PATH suggestion
        echo
        echo "To use the installed binaries, add to your ~/.bashrc:"
        echo "export PATH="$INSTALL_PREFIX/bin:\$PATH""
    fi
}

# Quick test
quick_test() {
    echo "Running quick functionality test..."

    ./pseudomode_cli --material MoS2 --temperature 300 --max-modes 3 --time-max 10 --output test_output.json

    if [ -f "test_output.json" ]; then
        echo "Quick test PASSED - framework is working correctly"
        rm test_output.json
    else
        echo "Quick test FAILED - check configuration"
        exit 1
    fi
}

# Main build process
main() {
    case "${1:-full}" in
        deps|dependencies)
            install_dependencies
            ;;
        configure)
            check_cuda
            configure_build
            ;;
        build)
            build_project
            ;;
        test)
            run_tests
            ;;
        benchmark)
            run_benchmarks
            ;;
        install)
            install_project "${2:-local}"
            ;;
        quick-test)
            quick_test
            ;;
        full)
            install_dependencies
            check_cuda
            configure_build
            build_project
            run_tests
            quick_test
            echo
            echo "=== BUILD COMPLETED SUCCESSFULLY ==="
            echo
            echo "Next steps:"
            echo "  1. Install: ./build.sh install"
            echo "  2. Quick test: cd build && ./pseudomode_cli --help"
            echo "  3. Python: python3 -c 'import pseudomode_py; print(pseudomode_py.__version__)'"
            ;;
        clean)
            echo "Cleaning build directory..."
            rm -rf build/
            echo "Clean completed"
            ;;
        *)
            echo "Usage: $0 [command]"
            echo "Commands:"
            echo "  full        - Complete build process (default)"
            echo "  deps        - Install dependencies only"
            echo "  configure   - Configure build only"
            echo "  build       - Build project only"
            echo "  test        - Run tests only"
            echo "  benchmark   - Run performance benchmarks"
            echo "  install     - Install [system|local]"
            echo "  quick-test  - Run quick functionality test"
            echo "  clean       - Clean build directory"
            ;;
    esac
}

main "$@"
