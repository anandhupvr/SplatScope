#include "scene/scene.h"

#include "splat/splat.h"

Scene::Scene(GaussianCloud cloud) : cloud_(std::move(cloud)) {
    cov_3d_.reserve(cloud_.size());

    for (size_t i = 0; i < cloud_.size(); ++i) {
        cov_3d_.emplace_back(splat::compute_covariance(cloud_.scale[i], cloud_.rotation[i]));
    }
}

std::vector<Eigen::Matrix3f> Scene::covarience_3d() const {
    return cov_3d_;
}

const GaussianCloud& Scene::gaussians() const {
    return cloud_;
}