#include <iostream>

#include "tinyply.h"

using namespace tinyply;

int main() {
    std::cout << " test \n";

    std::string filepath =
        "/home/user1/work/learn/3dgs/pretrainedModelsGaussian/counter/point_cloud/iteration_30000/"
        "point_cloud.ply";
    std::cout << "Reading file : " << filepath << std::endl;

    return 0;
}