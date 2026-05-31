#pragma once

#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/Geometry"

namespace splat {

Eigen::Matrix3f compute_covariance(const Eigen::Vector3f& scale, Eigen::Quaternionf& rotation);

Eigen::Vector3f world_to_camera(const Eigen::Vector3f& world_pos,
                                const Eigen::Matrix4f& view_matrix);

Eigen::Vector2f camera_to_screen(const Eigen::Vector3f& camera_pos,
                                 const Eigen::Matrix3f& intrinsic);
};  // namespace splat