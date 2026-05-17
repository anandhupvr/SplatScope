#pragma once

#include <Eigen/Dense>
#include <vector>
#include "Eigen/Core"
#include "Eigen/Geometry"

struct GaussianCloud{
    std::vector<Eigen::Vector3f> mean;
    std::vector<Eigen::Vector3f> scale;
    std::vector<Eigen::Quaternionf> rotation;
    std::vector<Eigen::Vector3f> color;
    std::vector<float> opacity;
};