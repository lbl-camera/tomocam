#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <nlohmann/json.hpp>

#include "dist_array.h"
#include "hdf5/reader.h"
#include "hdf5/writer.h"
#include "timer.h"
#include "tomocam.h"

#ifdef MULTIPROC
#include <mpi.h>
#endif

using json = nlohmann::json;



// filtered backprojection
template<typename T>
tomocam::DArray<T> fbp(tomocam::DArray<T> &sino, std::vector<T> &angs, int center) {

    // save the original sino size
    int nrays = sino.ncols();

    // center and padding
    auto sino2 = tomocam::preproc(sino, (T) center);

    // do the filtered backprojection
    auto recon = tomocam::backproject(sino2, angs, true);

    // remove padding and return
    return tomocam::postproc(recon, nrays);
}

int main(int argc, char **argv) {

    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <json>" << std::endl;
        return 1;
    }

    // initialize MPI
    #ifdef MULTIPROC
    tomocam::multiproc::mp.init(argc, argv);
    int nprocs = tomocam::multiproc::mp.nprocs();
    int myrank = tomocam::multiproc::mp.myrank();
    #else
    int nprocs = 1;
    int myrank = 0;
    #endif

    // get JSON file
    std::ifstream json_file(argv[1]);
    if (!json_file.is_open()) {
        std::cerr << "Error: cannot open JSON file" << std::endl;
        return 1;
    }

    json cfg = json::parse(json_file);

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

    if (cfg.find("MBIR") == cfg.end()) {
        std::cerr << "Error: missing MBIR parameters" << std::endl;
        return 1;
    }

    // MBIR parameters
    auto params = cfg["MBIR"];
    int max_iters = params["num_iters"];
    float sigma = params["sigma"];

    float tol = 0.001;
    if (params.find("tol") != params.end()) tol = params["tol"];
    float xtol = 0.001;
    if (params.find("xtol") != params.end()) xtol = params["xtol"];

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
        ibeg = extra_slcs * (slcs_per_proc + 1) +
            (myrank - extra_slcs) * slcs_per_proc;
        iend = ibeg + slcs_per_proc;
    }
    #endif

    auto sino = fp.read_sinogram<float>(dataset.c_str(), ibeg, iend);
    auto angs = fp.read<float>(angles.c_str());

    // if number of columns is even, drop one column
    if (sino.ncols() % 2 == 0) {
        sino.dropcol();
        //center -= 1;
    }

    float cen = static_cast<float>(center);

    // inital guess
    auto x0 = tomocam::DArray<float>({sino.nslices(), sino.ncols(), sino.ncols()});
    x0.init(1.f);

    // run FBP
    auto x1 = fbp(sino, angs, center);
    // normalize
    x1.normalize();

    // run MBIR
    Timer t2;
    t2.start();
    auto recon2 = tomocam::mbir(x1, sino, angs, cen, max_iters, sigma, tol, xtol);
    t2.stop();
   

    #ifdef MULTIPROC
    if (myrank == 0)
    #endif
        std::cout << "time taken(s): " << t2.seconds() << std::endl;

    // save reconstruction
    #ifdef MULTIPROC
    auto fname = cfg["output"].get<std::string>();
    auto prefix = fname.substr(0, fname.find_last_of("."));
    auto suffix = fname.substr(fname.find_last_of("."));
    std::stringstream tag;
    tag << std::setw(3) << std::setfill('0') << myrank;
    auto outf = prefix + tag.str() + suffix;
    #else
    auto outf = cfg["output"].get<std::string>();
    #endif
    tomocam::h5::Writer writer(outf.c_str());
    writer.write("recon", recon2);

    #ifdef MULTIPROC
    tomocam::multiproc::mp.finalize();
    #endif

    return 0;
}
