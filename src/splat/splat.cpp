#include "splat/splat.h"

#include "Eigen/Core"
#include "Eigen/Geometry"

Eigen::Matrix3f splat::compute_covariance(const Eigen::Vector3f& scale,
                                          Eigen::Quaternionf& rotation) {
    // E = R S S.T R.T , E = 3x3 symetric matrix describe any oriented
    // ellipsoid shape
    Eigen::Matrix3f s = scale.asDiagonal();
    Eigen::Matrix3f r = rotation.toRotationMatrix();

    auto covarience = r * s * s.transpose() * r.transpose();

    return covarience;
}

Eigen::Vector3f splat::world_to_camera(const Eigen::Vector3f& world_pos,
                                       const Eigen::Matrix4f& view_matrix) {
    // convert to homogenous coordinate
    Eigen::Vector4f cam_point = view_matrix * world_pos.homogeneous();
    Eigen::Vector3f cam_pos = cam_point.head<3>() / cam_point.w();
    return cam_pos;
}

Eigen::Vector2f splat::camera_to_screen(const Eigen::Vector3f& camera_pos,
                                        const Eigen::Matrix3f& intrinsic) {
    // project to screen space
    Eigen::Vector3f screen_point = intrinsic * camera_pos;
    // prespective divide
    return screen_point.head<2>() / screen_point.z();
}