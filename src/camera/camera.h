#pragma once

#include "Eigen/Core"

// namespace camera {

struct Instrinsic {
    float fx = 800.0f, fy = 800.0f, cx = 640.0f, cy = 360.0f;
    int width = 1280, height = 720;

    Eigen::Matrix3f as_matrix() const;
};

class Camera {
   public:
    Camera();

    // target is the point that camera look at
    void set_target(const Eigen::Vector3f& target);
    Eigen::Vector3f target() const;

    // Distance from camera to target
    void set_distance(float distance);
    float distance() const;

    // Yaw (rotation around y-axis) and pitch (rotation around x-axis) in radians
    void set_yaw(float yaw);
    float yaw() const;
    void set_pitch(float pitch);
    float pitch() const;

    Eigen::Vector3f position() const;

    // mouse controls
    void orbit(float delta_yaw, float delta_pitch);
    void zoom(float delta);
    void pan(const Eigen::Vector2f& screen_delta);

    Eigen::Matrix4f view_matrix() const;
    Eigen::Matrix4f projection_matrix() const;

    // instrinic parameters
    // Instrinic& instrinic() const;
    Eigen::Matrix3f instrinsic_as_matrix() const;
    void set_intrinic(const Instrinsic& intrinsic);

   private:
    Eigen::Vector3f kWorldUp;
    Eigen::Vector3f target_ = Eigen::Vector3f::Zero();
    float distance_ = 5.0f;
    float yaw_ = 0.0f;
    float pitch_ = 0.0f;
    Instrinsic instrinsic_;
};
// }  // namespace camera
