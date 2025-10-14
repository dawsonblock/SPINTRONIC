# Multi-stage Docker build for Extended 2D/3D Pseudomode Framework
# Apache License 2.0 - Copyright (c) 2025 Aetheron Research

# =============================================================================
# Stage 1: Build environment with all dependencies
# =============================================================================
FROM ubuntu:22.04 AS builder

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    # Build tools
    build-essential cmake ninja-build git curl \
    # Required libraries
    libeigen3-dev libfftw3-dev libomp-dev \
    # Optional libraries for enhanced performance
    libhdf5-dev libopenmpi-dev \
    # Python support
    python3 python3-dev python3-pip \
    # JSON schema validation
    python3-jsonschema \
    # Testing utilities
    libgtest-dev libbenchmark-dev \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir pybind11[global] numpy scipy matplotlib jsonschema

# Set up build environment
WORKDIR /build

# Copy source code
COPY . /build/

# Configure build with all optimizations
RUN cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_EIGEN=ON \
    -DUSE_OPENMP=ON \
    -DUSE_JSON=ON \
    -DUSE_MPI=OFF \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DBUILD_TESTS=ON \
    -DBUILD_FITTING_TOOLS=ON \
    -DBUILD_ADVANCED_TOOLS=ON \
    -DBUILD_BENCHMARKS=ON \
    -DUSE_FAST_MATH=ON \
    -DCMAKE_INSTALL_PREFIX=/opt/pseudomode

# Build the framework
RUN cmake --build build --parallel $(nproc)

# Run tests to ensure build quality
RUN cd build && ctest --output-on-failure --parallel

# Install to staging area
RUN cmake --build build --target install

# =============================================================================
# Stage 2: Runtime environment (minimal)
# =============================================================================
FROM ubuntu:22.04 AS runtime

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    # Runtime libraries
    libeigen3-dev libfftw3-3 libgomp1 \
    libhdf5-103 libopenmpi3 \
    # Python runtime
    python3 python3-numpy python3-scipy python3-matplotlib \
    # JSON utilities
    jq \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy built framework from builder stage
COPY --from=builder /opt/pseudomode /opt/pseudomode

# Add framework to PATH
ENV PATH="/opt/pseudomode/bin:$PATH"
ENV LD_LIBRARY_PATH="/opt/pseudomode/lib:$LD_LIBRARY_PATH"
ENV PYTHONPATH="/opt/pseudomode/lib/python3/site-packages:$PYTHONPATH"

# Create working directory with proper permissions
RUN mkdir -p /workspace && chmod 777 /workspace
WORKDIR /workspace

# Copy essential data files
COPY --from=builder /build/materials_3d.json /workspace/
COPY --from=builder /build/data/ /workspace/data/
COPY --from=builder /build/examples/ /workspace/examples/

# Default user (non-root for security)
RUN useradd -m -s /bin/bash pseudomode && chown -R pseudomode:pseudomode /workspace
USER pseudomode

# Expose any ports (none needed for this application)
# EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD pseudomode_cli --help > /dev/null || exit 1

# Default command - run a simple demonstration
CMD ["bash", "-c", "\
    echo 'Extended 2D/3D Pseudomode Framework - Docker Container'; \
    echo 'Apache License 2.0 - Copyright (c) 2025 Aetheron Research'; \
    echo ''; \
    echo 'Available commands:'; \
    echo '  pseudomode_cli        - Main simulation interface'; \
    echo '  pseudomode_scan       - Temperature curve generation'; \
    echo '  pseudomode_fit        - Parameter fitting'; \
    echo '  bootstrap_fit         - Uncertainty analysis'; \
    echo '  parameter_sweep       - Parameter space exploration'; \
    echo ''; \
    echo 'Example usage:'; \
    echo '  pseudomode_cli GaAs 300 --dim 3D --channels dp,pe,polar'; \
    echo '  pseudomode_scan GaAs --dim 3D --Tmin 50 --Tmax 350 --n 31 --out gaas.csv'; \
    echo ''; \
    echo 'Running demonstration...'; \
    pseudomode_scan GaAs --dim 3D --materials materials_3d.json --channels dp,pe,polar --Tmin 100 --Tmax 300 --n 11 --out demo_scan.csv && \
    echo 'Results saved to demo_scan.csv:' && \
    head -5 demo_scan.csv && \
    echo '...' && \
    tail -2 demo_scan.csv \
    "]

# =============================================================================
# Labels for metadata
# =============================================================================
LABEL maintainer="Aetheron Research <technical-support@aetheron-research.com>"
LABEL version="2.0.0"
LABEL description="Extended 2D/3D Non-Markovian Pseudomode Framework"
LABEL license="Apache-2.0"
LABEL org.opencontainers.image.title="Pseudomode Framework Extended"
LABEL org.opencontainers.image.description="Complete implementation for 2D and 3D quantum materials coherence simulation"
LABEL org.opencontainers.image.vendor="Aetheron Research"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.source="https://github.com/aetheron-research/pseudomode-extended"
