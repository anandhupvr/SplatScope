#pragma once

#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"

#include <Eigen/Dense>

struct GaussianCloud {
    std::vector<Eigen::Vector3f> mean;
    std::vector<Eigen::Vector3f> scale;
    std::vector<Eigen::Quaternionf> rotation;
    std::vector<Eigen::Vector3f> color;  // used by cpu renderer
    std::vector<float> opacity;
    std::vector<float> sh;
    int sh_degree = 0;

    size_t size() const {
        return mean.size();
    }
};