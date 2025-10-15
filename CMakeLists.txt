# CMakeLists.txt for 2D Pseudomode Framework C++/CUDA Implementation
# Apache License 2.0 - Copyright (c) 2025 Aetheron Research

cmake_minimum_required(VERSION 3.18)
project(pseudomode_framework_cuda VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add local Eigen path if it exists
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/eigen-3.4.0")
    set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/external/eigen-3.4.0" ${CMAKE_PREFIX_PATH})
endif()

# Find required packages
find_package(Eigen3 QUIET)
find_package(OpenMP QUIET)

# Check for CUDA availability
include(CheckLanguage)
check_language(CUDA)

# Optional packages
find_package(PkgConfig)
pkg_check_modules(HDF5 hdf5)

# Build options
option(USE_CUDA "Enable CUDA GPU acceleration" ON)
option(USE_OPENMP "Enable OpenMP parallelization" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_BENCHMARKS "Build performance benchmarks" OFF)

# Compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")

# CUDA configuration
if(CMAKE_CUDA_COMPILER AND USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Try to detect CUDA architecture
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)  # Common architectures
    endif()

    message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

    add_definitions(-DUSE_CUDA)
    set(CUDA_ENABLED TRUE)
else()
    message(STATUS "CUDA not available or disabled, building CPU-only version")
    set(CUDA_ENABLED FALSE)
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/external/eigen-3.4.0
)

if(Eigen3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
    message(STATUS "Eigen3 found: ${EIGEN3_INCLUDE_DIR}")
else()
    # Use bundled Eigen
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/eigen-3.4.0")
        include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/eigen-3.4.0)
        message(STATUS "Using bundled Eigen from external/eigen-3.4.0")
        set(Eigen3_FOUND TRUE)
    else()
        message(WARNING "Eigen3 not found - some features will be disabled")
    endif()
endif()

# Source files
set(SOURCES
    src/spectral_density_2d.cpp
    src/materials_database.cpp
    src/prony_fitting.cpp
    src/quantum_state.cpp
    src/lindblad_evolution.cpp
    src/utils.cpp
    src/high_level_interface.cpp
)

set(CUDA_SOURCES
    src/cuda_kernels.cu
)

# Create main library
add_library(pseudomode_framework SHARED ${SOURCES})

# Special compilation flags for slow-compiling files
# Disable all optimizations and reduce Eigen template depth
set_source_files_properties(src/prony_fitting.cpp PROPERTIES 
    COMPILE_FLAGS "-O0 -g0 -ftemplate-depth=128 -fno-var-tracking"
)

# Link CUDA sources if available
if(CUDA_ENABLED)
    add_library(pseudomode_framework_cuda SHARED ${CUDA_SOURCES})
    set_target_properties(pseudomode_framework_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_link_libraries(pseudomode_framework PRIVATE pseudomode_framework_cuda)
    target_link_libraries(pseudomode_framework_cuda PRIVATE cusparse cublas curand)
endif()

# Link Eigen (header-only) if found
# Eigen is header-only, so we don't need to link anything when using bundled version
if(TARGET Eigen3::Eigen)
    target_link_libraries(pseudomode_framework PRIVATE Eigen3::Eigen)
endif()

# Link OpenMP if enabled and found
if(USE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(pseudomode_framework PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP enabled")
else()
    message(STATUS "OpenMP not available or disabled")
endif()

# Link optional HDF5
if(HDF5_FOUND)
    target_include_directories(pseudomode_framework PRIVATE ${HDF5_INCLUDE_DIRS})
    target_link_libraries(pseudomode_framework PRIVATE ${HDF5_LIBRARIES})
    add_definitions(-DUSE_HDF5)
endif()

# Set library properties
set_target_properties(pseudomode_framework PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER "include/pseudomode_solver.h"
)

# Install targets
install(TARGETS pseudomode_framework
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)

# Create command-line executable
add_executable(pseudomode_cli src/main.cpp)

# CLI needs to link the library and Eigen headers
if(TARGET Eigen3::Eigen)
    target_link_libraries(pseudomode_cli pseudomode_framework Eigen3::Eigen)
    target_compile_definitions(pseudomode_cli PRIVATE PSEUDOMODE_HAVE_EIGEN=1)
else()
    target_link_libraries(pseudomode_cli pseudomode_framework)
    target_compile_definitions(pseudomode_cli PRIVATE PSEUDOMODE_HAVE_EIGEN=1)
endif()

install(TARGETS pseudomode_cli
    RUNTIME DESTINATION bin
)

# Python bindings using pybind11
if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 CONFIG)
    if(pybind11_FOUND)
        pybind11_add_module(pseudomode_py src/python_bindings.cpp)
        target_link_libraries(pseudomode_py PRIVATE pseudomode_framework)

        # Install Python module
        install(TARGETS pseudomode_py
            LIBRARY DESTINATION python
        )
    else()
        message(WARNING "pybind11 not found, skipping Python bindings")
    endif()
endif()

# Unit tests
if(BUILD_TESTS)
    find_package(GTest)
    if(GTest_FOUND)
        enable_testing()

        add_executable(test_spectral_density tests/test_spectral_density.cpp)
        target_link_libraries(test_spectral_density pseudomode_framework GTest::GTest GTest::Main)

        add_executable(test_prony_fitting tests/test_prony_fitting.cpp)
        target_link_libraries(test_prony_fitting pseudomode_framework GTest::GTest GTest::Main)

        add_executable(test_quantum_state tests/test_quantum_state.cpp)
        target_link_libraries(test_quantum_state pseudomode_framework GTest::GTest GTest::Main)

        add_test(NAME SpectralDensityTest COMMAND test_spectral_density)
        add_test(NAME PronyFittingTest COMMAND test_prony_fitting)
        add_test(NAME QuantumStateTest COMMAND test_quantum_state)
    endif()
endif()

# Benchmarks
if(BUILD_BENCHMARKS)
    find_package(benchmark)
    if(benchmark_FOUND)
        add_executable(benchmark_pseudomode benchmarks/benchmark_main.cpp)
        target_link_libraries(benchmark_pseudomode pseudomode_framework benchmark::benchmark)
    endif()
endif()

# Documentation
find_package(Doxygen)
if(DOXYGEN_FOUND)
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
endif()

# Package configuration
include(CMakePackageConfigHelpers)
configure_package_config_file(
    cmake/pseudomode-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/pseudomode-config.cmake
    INSTALL_DESTINATION lib/cmake/pseudomode
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/pseudomode-config.cmake
    DESTINATION lib/cmake/pseudomode
)
