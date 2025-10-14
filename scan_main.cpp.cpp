/*
 * Temperature Scan CLI - High-throughput T(T) curve generation
 * Apache License 2.0 - Copyright (c) 2025 Aetheron Research
 */

#include "pseudomode_solver_extended.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>

using namespace PseudomodeFramework;

static void usage(){
    std::cerr << "Usage: pseudomode_scan <Material> [options]\n"
              << "\n"
              << "Options:\n"
              << "  --dim <2D|3D>          Material dimensionality\n"
              << "  --materials <json>     Materials database file\n"
              << "  --channels <csv>       Comma-separated channel list\n"
              << "  --Tmin <K>             Minimum temperature [K]\n"
              << "  --Tmax <K>             Maximum temperature [K]\n"
              << "  --n <N>                Number of temperature points\n"
              << "  --log-scale            Use logarithmic temperature spacing\n"
              << "  --out <csv>            Output CSV file\n"
              << "  --omega0 <eV>          System frequency [eV]\n"
              << "  --max-modes <K>        Maximum pseudomodes\n"
              << "  --use-gpu              Enable GPU acceleration\n"
              << "  --n-max <N>            Oscillator truncation\n"
              << "  --verbose              Verbose progress output\n"
              << "\n"
              << "Examples:\n"
              << "  pseudomode_scan GaAs --dim 3D --channels dp,pe,polar --Tmin 50 --Tmax 350 --n 61\n"
              << "  pseudomode_scan MoS2 --dim 2D --log-scale --Tmin 10 --Tmax 300 --n 50\n"
              << "  pseudomode_scan Diamond_NV --dim 3D --channels dp,orbach --out diamond_scan.csv\n";
}

static ChannelToggles parse_channels(const std::string& s, Dimensionality dim){
    ChannelToggles ch{};

    // Set defaults based on dimension
    if (dim == Dimensionality::D2) {
        ch.acoustic_2d = true;
        ch.flexural_2d = true;
        ch.optical_2d = true;
    } else {
        ch.dp = true;
        ch.pe = true;
        ch.polar = true;
    }

    if (s.empty()) return ch;

    // Reset all to false, then enable specified
    ch = ChannelToggles{};

    std::stringstream ss(s);
    std::string tok;
    while(std::getline(ss, tok, ',')){
        // Trim whitespace
        tok.erase(0, tok.find_first_not_of(" \t"));
        tok.erase(tok.find_last_not_of(" \t") + 1);

        if(tok == "dp") ch.dp = true;
        else if(tok == "pe") ch.pe = true;
        else if(tok == "polar") ch.polar = true;
        else if(tok == "raman") ch.raman = true;
        else if(tok == "orbach") ch.orbach = true;
        else if(tok == "acoustic") ch.acoustic_2d = true;
        else if(tok == "flexural") ch.flexural_2d = true;
        else if(tok == "optical") ch.optical_2d = true;
    }

    return ch;
}

int main(int argc, char** argv){
    if (argc < 2){ 
        usage(); 
        return 1; 
    }

    // Parse arguments
    std::string material = argv[1];
    std::string dimS = "auto";
    std::string materials_file = "materials_3d.json";
    std::string channels_str = "";
    double Tmin = 50.0;
    double Tmax = 300.0;
    int N = 50;
    bool log_scale = false;
    std::string output = "scan.csv";
    double omega0_eV = 1.6;
    int max_modes = 4;
    bool use_gpu = false;
    int n_max = 0; // Auto
    bool verbose = false;

    for (int i = 2; i < argc; i++){
        std::string arg = argv[i];

        if(arg == "--dim" && i + 1 < argc) {
            dimS = argv[++i];
        } else if(arg == "--materials" && i + 1 < argc) {
            materials_file = argv[++i];
        } else if(arg == "--channels" && i + 1 < argc) {
            channels_str = argv[++i];
        } else if(arg == "--Tmin" && i + 1 < argc) {
            Tmin = std::stod(argv[++i]);
        } else if(arg == "--Tmax" && i + 1 < argc) {
            Tmax = std::stod(argv[++i]);
        } else if(arg == "--n" && i + 1 < argc) {
            N = std::stoi(argv[++i]);
        } else if(arg == "--log-scale") {
            log_scale = true;
        } else if(arg == "--out" && i + 1 < argc) {
            output = argv[++i];
        } else if(arg == "--omega0" && i + 1 < argc) {
            omega0_eV = std::stod(argv[++i]);
        } else if(arg == "--max-modes" && i + 1 < argc) {
            max_modes = std::stoi(argv[++i]);
        } else if(arg == "--use-gpu") {
            use_gpu = true;
        } else if(arg == "--n-max" && i + 1 < argc) {
            n_max = std::stoi(argv[++i]);
        } else if(arg == "--verbose") {
            verbose = true;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            usage();
            return 2;
        }
    }

    try {
        // Auto-detect dimensionality if needed
        Dimensionality dim;
        if (dimS == "auto") {
            // Simple heuristic based on material name
            std::unordered_set<std::string> known_2d = {"MoS2", "WSe2", "MoSe2", "WS2", "graphene", "hBN"};
            std::unordered_set<std::string> known_3d = {"GaAs", "Diamond_NV", "4H_SiC", "Si_P", "InAs"};

            if (known_2d.count(material)) {
                dim = Dimensionality::D2;
                if (verbose) std::cout << "Auto-detected 2D material" << std::endl;
            } else if (known_3d.count(material)) {
                dim = Dimensionality::D3;
                if (verbose) std::cout << "Auto-detected 3D material" << std::endl;
            } else {
                dim = Dimensionality::D3; // Default
                if (verbose) std::cout << "Unknown material, defaulting to 3D" << std::endl;
            }
        } else {
            dim = (dimS == "3D") ? Dimensionality::D3 : Dimensionality::D2;
        }

        // Parse channels
        ChannelToggles channels = parse_channels(channels_str, dim);

        // Setup configuration
        SimulationConfig config;
        config.dim = dim;
        config.channels = channels;
        config.materials_json = materials_file;
        config.max_modes = max_modes;
        config.use_gpu = use_gpu;

        // Load material
        MaterialSpec spec;
        spec.dim = dim;
        if (dim == Dimensionality::D3) {
            spec.mat_3d = MaterialDatabase::load_material(material, dim, materials_file).mat_3d;
        } else {
            // Load 2D material
            spec.mat_2d = MaterialDatabase::load_material(material, dim, materials_file).mat_2d;
        }

        // Initialize framework
        ExtendedPseudomodeFramework framework(config);

        // System parameters
        UnifiedLindbladSolver::SystemParams system_params;
        system_params.omega0_eV = omega0_eV;
        if (n_max > 0) {
            system_params.n_max = n_max;
        }

        // Generate temperature points
        std::vector<double> temperatures(N);
        for (int i = 0; i < N; i++){
            double fraction = double(i) / double(N - 1);

            if (log_scale) {
                // Logarithmic spacing
                double log_min = std::log(Tmin);
                double log_max = std::log(Tmax);
                temperatures[i] = std::exp(log_min + fraction * (log_max - log_min));
            } else {
                // Linear spacing
                temperatures[i] = Tmin + fraction * (Tmax - Tmin);
            }
        }

        // Print configuration
        if (verbose) {
            std::cout << "=== Temperature Scan Configuration ===" << std::endl;
            std::cout << "Material: " << material << " (" << (dim == Dimensionality::D3 ? "3D" : "2D") << ")" << std::endl;
            std::cout << "Temperature range: " << Tmin << " - " << Tmax << " K (" << N << " points, ";
            std::cout << (log_scale ? "logarithmic" : "linear") << ")" << std::endl;
            std::cout << "System frequency: " << omega0_eV << " eV" << std::endl;
            std::cout << "Max modes: " << max_modes << std::endl;
            std::cout << "GPU: " << (use_gpu ? "enabled" : "disabled") << std::endl;
            std::cout << "Output: " << output << std::endl;
            std::cout << std::endl;
        }

        // Open output file
        std::ofstream outfile(output);
        if (!outfile.is_open()) {
            std::cerr << "Error: Cannot open output file " << output << std::endl;
            return 3;
        }

        // Write CSV header
        outfile << std::fixed << std::setprecision(6);
        outfile << "T_K,T2star_ps,T1_ps,n_modes,fit_rmse,status" << std::endl;

        // Temperature scan loop
        for (int i = 0; i < N; i++){
            double T = temperatures[i];
            system_params.temperature_K = T;

            if (verbose) {
                std::cout << "\rProgress: " << (i+1) << "/" << N 
                          << " (T = " << std::fixed << std::setprecision(1) << T << " K)" << std::flush;
            }

            try {
                auto result = framework.simulate_material_spec(spec, system_params);

                if (result.success) {
                    outfile << T << ","
                            << result.coherence_times.T2_star_ps << ","
                            << result.coherence_times.T1_ps << ","
                            << result.modes.size() << ","
                            << result.fit_rmse << ","
                            << "success" << std::endl;
                } else {
                    outfile << T << ",NaN,NaN,0,NaN,failed" << std::endl;
                    if (verbose) {
                        std::cerr << "\nWarning: Simulation failed at T=" << T << "K: " << result.status << std::endl;
                    }
                }

            } catch (const std::exception& e) {
                outfile << T << ",NaN,NaN,0,NaN,error" << std::endl;
                if (verbose) {
                    std::cerr << "\nError at T=" << T << "K: " << e.what() << std::endl;
                }
            }
        }

        if (verbose) {
            std::cout << std::endl << "✓ Temperature scan completed successfully!" << std::endl;
            std::cout << "Results written to: " << output << std::endl;
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
