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

    void set_world_up(const Eigen::Vector3f& up);
    Eigen::Vector3f world_up() const;

    // target is the point that camera look at
    void set_target(const Eigen::Vector3f& target);
    Eigen::Vector3f target() const;

    // Distance from camera to target
    void set_distance(float distance);
    float distance() const;

    // yaw: walk around world_up; pitch: height above the target (+ = above). Radians.
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

    // instrinic parameters
    Eigen::Matrix3f instrinsic_as_matrix() const;
    Instrinsic intrinsic() const;
    void set_intrinic(const Instrinsic& intrinsic);

   private:
    void axes(Eigen::Vector3f& right, Eigen::Vector3f& down, Eigen::Vector3f& forward) const;

    Eigen::Vector3f world_up_ = Eigen::Vector3f(0.0f, -1.0f, 0.0f);
    Eigen::Vector3f e1_, e2_;  // two floor directions, perpendicular to world_up_
    Eigen::Vector3f target_ = Eigen::Vector3f::Zero();
    float distance_ = 5.0f;
    float yaw_ = 0.0f;
    float pitch_ = 0.0f;
    Instrinsic instrinsic_;
};
// }  // namespace camera
