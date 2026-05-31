#pragma once

#include "Eigen/Core"
#include "Eigen/Dense"

#include "scene/gaussian_cloud.h"

class Scene {
   public:
    Scene(GaussianCloud cloud);

    std::vector<Eigen::Matrix3f> covarience_3d() const;
    const GaussianCloud& gaussians() const;

   private:
    GaussianCloud cloud_;
    std::vector<Eigen::Matrix3f> cov_3d_;
};