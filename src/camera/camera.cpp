#include "camera.h"

#include "Eigen/Core"
#include "Eigen/Geometry"

namespace camera {

Eigen::Matrix3f Instrinsic::as_matrix() const {
    Eigen::Matrix3f K = Eigen::Matrix3f::Zero();
    K(0, 0) = fx;
    K(1, 1) = fy;
    K(0, 2) = cx;
    K(1, 2) = cy;
    K(2, 2) = 1.0f;
    return K;
}

Camera::Camera() : kWorldUp(0.0f, 1.0f, 0.0f), instrinsic_() {}

void Camera::set_target(const Eigen::Vector3f& target) {
    target_ = target;
}

Eigen::Vector3f Camera::target() const {
    return target_;
}

void Camera::set_distance(float distance) {
    distance_ = distance;
}

float Camera::distance() const {
    return distance_;
}

void Camera::set_yaw(float yaw) {
    yaw_ = yaw;
}

float Camera::yaw() const {
    return yaw_;
}

void Camera::set_pitch(float pitch) {
    pitch_ = pitch;
}

float Camera::pitch() const {
    return pitch_;
}

void Camera::set_intrinic(const Instrinsic& intrinsic) {
    instrinsic_ = intrinsic;
}

Eigen::Matrix3f Camera::instrinsic_as_matrix() const {
    return instrinsic_.as_matrix();
}

Eigen::Vector3f Camera::position() const {
    // Convert spherical coordinates to Cartesian coordinates
    // camera sits on a sphere with radius = distance_ , theta = yaw_ , phi = pitch_
    float x = distance_ * cos(pitch_) * sin(yaw_);
    float y = distance_ * sin(pitch_);
    float z = distance_ * cos(pitch_) * cos(yaw_);

    return Eigen::Vector3f(x, y, z) + target_;
}

Eigen::Matrix4f Camera::projection_matrix() const {
    const auto& K = instrinsic_;
    float n = 0.01f, f = 1000.0f;  // near far planes
    Eigen::Matrix4f P = Eigen::Matrix4f::Zero();
    P(0, 0) = 2 * K.fx / K.width;
    P(1, 1) = 2 * K.fy / K.height;
    P(0, 2) = 1 - 2 * K.cx / K.width;
    P(1, 2) = 2 * K.cy / K.height - 1;
    P(2, 2) = -(f + n) / (f - n);
    P(2, 3) = -2 * f * n / (f - n);
    P(3, 2) = -1.0f;
    return P;
}

Eigen::Matrix4f Camera::view_matrix() const {
    // LookAt matrix
    Eigen::Vector3f eye = position();
    Eigen::Vector3f center = target_;
    // Eigen::Vector3f up = kWorldUp;

    Eigen::Vector3f forward = (center - eye).normalized();  // forward
    // Eigen::Vector3f right = forward.cross(kWorldUp).normalized();  // right
    // this is to match the up direction with colabmap not using opengl convension (kWorldUp)
    Eigen::Vector3f right =
        forward.cross(Eigen::Vector3f(0.0f, -1.0f, 0.0f)).normalized();  // right
    Eigen::Vector3f up = right.cross(forward);                           // up

    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();
    view.block<1, 3>(0, 0) = right.transpose();
    view.block<1, 3>(1, 0) = up.transpose();
    view.block<1, 3>(2, 0) = forward.transpose();
    view.block<3, 1>(0, 3) = -view.block<3, 3>(0, 0) * eye;
    return view;
}
}  // namespace camera