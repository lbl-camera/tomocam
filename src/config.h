/* -------------------------------------------------------------------------------
 * Tomocam Copyright (c) 2018
 *
 * The Regents of the University of California, through Lawrence Berkeley
 * National Laboratory (subject to receipt of any required approvals from the
 * U.S. Dept. of Energy). All rights reserved.
 *
 * If you have questions about your rights to use or distribute this software,
 * please contact Berkeley Lab's Innovation & Partnerships Office at
 * IPO@lbl.gov.
 *
 * NOTICE. This Software was developed under funding from the U.S. Department of
 * Energy and the U.S. Government consequently retains certain rights. As such,
 * the U.S. Government has been granted for itself and others acting on its
 * behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software
 * to reproduce, distribute copies to the public, prepare derivative works, and
 * perform publicly and display publicly, and to permit other to do so.
 *---------------------------------------------------------------------------------
 */

#ifndef TOMOCAM_CONFIG__H
#define TOMOCAM_CONFIG__H

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include "common.h"

namespace tomocam {

    inline json load_config(const std::string &filename) {
        std::ifstream ifs(filename);
        json cfg = json::parse(ifs);
        return cfg;
    }

    inline ReconParams load_recon_params(json cfg) {
        ReconParams params;

        if (cfg.find("MBIR") == cfg.end()) {
            throw std::runtime_error("JSON file does not contain MBIR parameters");
        }
        json rec_cfg = cfg["MBIR"];
        if (rec_cfg.find("tol") != rec_cfg.end()) { params.tol = rec_cfg["tol"]; }
        if (rec_cfg.find("xtol") != rec_cfg.end()) { params.xtol = rec_cfg["xtol"]; }
        if (rec_cfg.find("max_iters") != rec_cfg.end()) {
            params.max_iters = rec_cfg["max_iters"];
        }
        if (rec_cfg.find("inner_iters") != rec_cfg.end()) {
            params.inner_iters = rec_cfg["inner_iters"];
        }
        if (rec_cfg.find("mu") != rec_cfg.end()) { params.mu = rec_cfg["mu"]; }
        if (rec_cfg.find("lambda") != rec_cfg.end()) {
            params.lambda = rec_cfg["lambda"];
        }
        if (rec_cfg.find("sigma") != rec_cfg.end()) {
            params.sigma = rec_cfg["sigma"];
        }
        return params;
    }

    inline OutputParams load_output_params(json cfg) {
        OutputParams params;
        if (cfg.find("output") == cfg.end()) {
            throw std::runtime_error("JSON file does not contain output parameters");
        }
        json out_cfg = cfg["output"];
        if (out_cfg.find("filename") != out_cfg.end()) {
            std::filesystem::path outpath(out_cfg["filename"].get<std::string>());
            // if parent directory does not exist, create it
            std::filesystem::create_directories(outpath.parent_path());
            params.filename = outpath.string();

            // default extension is HDF5, but can accept tiff
            if (outpath.extension() == ".tiff") {
                params.format = "tiff";
            } else {
                params.format = "hdf5";
            }
        } else {
            throw std::runtime_error("JSON file does not contain output filename");
        }
        return params;
    }

    inline void dump_config(const ReconParams &params, const OutputParams &out_params,
                             std::ostream &os = std::cout) {
        os << "MBIR parameters:\n"
           << "  max_iters   = " << params.max_iters << "\n"
           << "  inner_iters = " << params.inner_iters << "\n"
           << "  tol         = " << params.tol << "\n"
           << "  xtol        = " << params.xtol << "\n"
           << "  mu          = " << params.mu << "\n"
           << "  lambda      = " << params.lambda << "\n"
           << "  sigma       = " << params.sigma << "\n"
           << "output parameters:\n"
           << "  filename    = " << out_params.filename << "\n"
           << "  format      = " << out_params.format << "\n";
    }

} // namespace tomocam
#endif // TOMOCAM_CONFIG__H
