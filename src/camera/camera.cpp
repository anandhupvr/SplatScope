#include "camera.h"

#include <cmath>

#include "Eigen/Core"
#include "Eigen/Geometry"

// namespace camera {

Eigen::Matrix3f Instrinsic::as_matrix() const {
    Eigen::Matrix3f K = Eigen::Matrix3f::Zero();
    K(0, 0) = fx;
    K(1, 1) = fy;
    K(0, 2) = cx;
    K(1, 2) = cy;
    K(2, 2) = 1.0f;
    return K;
}

Camera::Camera() : instrinsic_() {
    set_world_up(world_up_);  // builds e1_, e2_ from the default up
}

void Camera::set_world_up(const Eigen::Vector3f& up) {
    world_up_ = up.normalized();
    Eigen::Vector3f helper =
        std::abs(world_up_.x()) < 0.9f ? Eigen::Vector3f::UnitX() : Eigen::Vector3f::UnitZ();
    e1_ = world_up_.cross(helper).normalized();  // on the floor
    e2_ = world_up_.cross(e1_);                  // on the floor, perpendicular to e1_
}

Eigen::Vector3f Camera::world_up() const {
    return world_up_;
}

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

void Camera::orbit(float delta_yaw, float delta_pitch) {
    yaw_ += delta_yaw;
    pitch_ += delta_pitch;

    // prevent flipping / gimbal locak
    constexpr float kLimit = float(M_PI / 2.0) - 0.01f;
    pitch_ = std::clamp(pitch_, -kLimit, kLimit);
}

void Camera::zoom(float factor) {
    distance_ *= factor;
    distance_ = std::clamp(distance_, 0.01f, 1000.0f);
}

void Camera::pan(const Eigen::Vector2f& screen_delta) {
    Eigen::Vector3f right, down, forward;
    axes(right, down, forward);
    // screen x → camera right, screen y (grows downward) → camera down.
    // Move the target opposite to the drag so the scene follows the cursor.
    target_ -= right * screen_delta.x() + down * screen_delta.y();
}

void Camera::set_intrinic(const Instrinsic& intrinsic) {
    instrinsic_ = intrinsic;
}

Eigen::Matrix3f Camera::instrinsic_as_matrix() const {
    return instrinsic_.as_matrix();
}

Instrinsic Camera::intrinsic() const {
    return instrinsic_;
}

Eigen::Vector3f Camera::position() const {
    // yaw picks a direction on the floor (walking around the turntable)
    Eigen::Vector3f h = std::sin(yaw_) * e1_ + std::cos(yaw_) * e2_;
    // pitch lifts it toward world up (0 = level with the target, + = above)
    Eigen::Vector3f dir = std::cos(pitch_) * h + std::sin(pitch_) * world_up_;
    return target_ + distance_ * dir;
}

void Camera::axes(Eigen::Vector3f& right, Eigen::Vector3f& down, Eigen::Vector3f& forward) const {
    forward = (target_ - position()).normalized();
    right = forward.cross(world_up_).normalized();
    down = forward.cross(right);
}

Eigen::Matrix4f Camera::view_matrix() const {
    Eigen::Vector3f right, down, forward;
    axes(right, down, forward);
    const Eigen::Vector3f eye = position();

    // rows = camera axes in world coordinates (OpenCV: x right, y down, z forward). det = +1
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();
    view.block<1, 3>(0, 0) = right.transpose();
    view.block<1, 3>(1, 0) = down.transpose();
    view.block<1, 3>(2, 0) = forward.transpose();
    view.block<3, 1>(0, 3) = -view.block<3, 3>(0, 0) * eye;
    return view;
}
// }  // namespace camera