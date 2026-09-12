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

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>

#include "config.h"
#include "dist_array.h"
#include "hdf5/reader.h"
#include "hdf5/writer.h"
#include "timer.h"
#include "tomocam.h"

#ifdef MULTIPROC
#include <mpi.h>
#endif

using json = nlohmann::json;

int main(int argc, char **argv) {

    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <json>" << std::endl;
        return 1;
    }

// initialize MPI
#ifdef MULTIPROC
    int nprocs = tomocam::multiproc::mp.nprocs();
    int myrank = tomocam::multiproc::mp.myrank();
#else
    int nprocs = 1;
    int myrank = 0;
#endif

    // load JSON config
    json cfg = tomocam::load_config(argv[1]);

    // get parameters
    std::string filename = cfg["filename"];
    std::string dataset = cfg["dataset"];
    std::string angles = cfg["angles"];
    int center = cfg["axis"];

    int ibeg = 0, iend = -1;
    // chcek for "slices" key
    if (cfg.find("slices") != cfg.end()) {
        auto slcs = cfg["slices"];
        ibeg = slcs[0];
        iend = slcs[1];
    }

    auto params = tomocam::load_recon_params(cfg);
    auto out_params = tomocam::load_output_params(cfg);
    tomocam::dump_config(params, out_params);

    // load tomogrmaphic data
    tomocam::h5::Reader fp(filename.c_str());
    if (iend < 0) iend = fp.dims(dataset.c_str(), 1);
    int nslices = iend - ibeg;

#ifdef MULTIPROC
    int slcs_per_proc = nslices / nprocs;
    int extra_slcs = nslices % nprocs;
    if ((extra_slcs > 0) && (myrank < extra_slcs)) slcs_per_proc += 1;

    // set local ibegs and iends
    ibeg = myrank * slcs_per_proc;
    iend = ibeg + slcs_per_proc;
    if (myrank > extra_slcs) {
        ibeg =
            extra_slcs * (slcs_per_proc + 1) + (myrank - extra_slcs) * slcs_per_proc;
        iend = ibeg + slcs_per_proc;
    }
#endif

    auto sino = fp.read_sinogram<float>(dataset.c_str(), ibeg, iend);
    auto angs = fp.read<float>(angles.c_str());

    // if number of columns is even, drop one column
    if (sino.ncols() % 2 == 0) {
        sino.dropcol();
        // center -= 1;
    }

    float cen = static_cast<float>(center);

    tomocam::DArray<float> x0({0, 0, 0});
    // run MBIR (split-Bregman + CG)
    tomocam::Timer t2;
    t2.start();
    auto recon = tomocam::mbir_bregman(x0, sino, angs, cen, params);
    t2.stop();

#ifdef MULTIPROC
    if (myrank == 0)
#endif
        std::cout << "time taken(s): " << t2.seconds() << std::endl;

// save reconstruction
#ifdef MULTIPROC
    auto outf = out_params.insert_rank(myrank);
#else
    auto outf = out_params.filename;
#endif
    tomocam::h5::Writer writer(outf.c_str());
    writer.write("recon", recon);

    return 0;
}
