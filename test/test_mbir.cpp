#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>

#include "dist_array.h"
#include "tomocam.h"

#include "optimize.h"

#include "reader.h"
#include "writer.h"
#include "timer.h"

using json = nlohmann::json;

const float PI = M_PI;

int main(int argc, char **argv) {


    if (argc == 2) {
        std::ifstream fin(argv[1]);
        json cfg = json::parse(fin);

        auto data = cfg["data"];
        auto param = cfg["recon"];
        
        std::string filename = data["filename"];
        std::string dataset = data["dataset"];
        std::string theta = data["angles"];

        int ibeg = 0;
        int iend = 4;
        if (!data["sinos"].is_null()) {
            std::vector<int> sinos = data["sinos"];
            ibeg = sinos[0]; 
            iend = sinos[1];
        }
        int nslices = iend - ibeg;

        // read data
	    tomocam::H5Reader fp(filename.c_str());
        fp.setDataset(dataset.c_str());
	    auto sino = fp.read_sinogram(nslices, ibeg);
        std::vector<float> angs = fp.read_angles(theta.c_str());
        float * angles = angs.data();

        tomocam::dim3_t d1 = sino.dims();
        tomocam::dim3_t d2(d1.x, d1.z, d1.z);

        int max_iters = param["num_iters"];
        float center = data["axis"];
        float smooth = param["smoothness"];
        float sigma = 1.0 / smooth;

        float p = 1.2;
        float oversample = 2;


        std::cout << "Input size: (" << d1.x << ", " << d1.y << ", " << d1.z <<" )" << std::endl;
	    std::cout << "Center: " << center << std::endl;
	    std::cout << "Oversampling: " << oversample << std::endl;
	    std::cout << "1 / Smoothness: " << sigma << std::endl;
	    std::cout << "No. of iterations: " << max_iters << std::endl;

        Timer t;
	    auto recon = tomocam::mbir(sino, angles, center, oversample, sigma, p, max_iters);
        t.stop();
        std::cout << "time taken(ms): " << t.millisec() << std::endl;

        std::string outfile = param["output_filename"];
        tomocam::H5Writer outf(outfile.c_str());
        outf.write("recon", recon);
    }
    return 0;
}
