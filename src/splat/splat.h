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

Eigen::Matrix<float, 2, 3> compute_jacobian(const Eigen::Vector3f& camera_pos, float fx, float fy);

Eigen::Matrix2f project_covarience(const Eigen::Matrix3f& cov3d_world,
                                   const Eigen::Vector3f& camera_pos,
                                   const Eigen::Matrix3f& view_roataion,
                                   float fx,
                                   float fy);

Eigen::Vector4i compute_bounding_box(const Eigen::Matrix2f& cov2d,
                                     const Eigen::Vector2f& center_pixels,
                                     int w,
                                     int h);
};  // namespace splat