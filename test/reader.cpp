#include <filesystem>
#include <fstream>
#include <iostream>

#include <nlohmann/json.hpp>

#include "hdf5/reader.h"
#include "hdf5/writer.h"

using json = nlohmann::json;

int main(int argc, char *argv[]) {

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <JSON file>" << std::endl;
        return 1;
    }

    // read JSON file
    std::ifstream json_file(argv[1]);
    if (!json_file.is_open()) {
        std::cerr << "Failed to open file: " << argv[1] << std::endl;
        return 1;
    }
    nlohmann::json cfg = json::parse(json_file);

    // get filename
    const std::string filename = cfg["filename"];
    const std::string dataset = cfg["dataset"];
    const std::string angles = cfg["angles"];
    int begin = 0;
    int end = -1;
    // check for slices key
    if (cfg.find("slices") != cfg.end()) {
        auto slices = cfg["slices"];
        begin = slices[0];
        end = slices[1];
    }

    // check if file exists
    if (!std::filesystem::exists(filename)) {
        std::cerr << "File does not exist: " << filename << std::endl;
        return 1;
    }
    tomocam::h5::Reader reader(filename.c_str());
    auto sino = reader.read_sinogram<float>(dataset.c_str(), begin, end);
    auto theta = reader.read<float>(angles.c_str());

    // write sinogram to file
    const std::string output_filename = cfg["output"];
    tomocam::h5::Writer writer(output_filename.c_str());
    writer.write<float>("sino", sino);

    return 0;
}
