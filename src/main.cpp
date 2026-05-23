#include <exception>
#include <iostream>

#include "scene/ply_loader.h"
#include "tinyply.h"

using namespace tinyply;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << "<path/to/pretrained_gaussin.py\n";
        return 1;
    }

    std::string filepath = argv[1];

    try {
        auto gaussian = scene::load(filepath);
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << "\n";
        return 1;
    }

    return 0;
}