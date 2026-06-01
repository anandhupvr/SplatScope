#pragma once
#include "Eigen/Core"

#include "renderer/renderer.h"
#include "scene/scene.h"

class CpuRenderer : public Renderer {
   public:
    void set_scene(const Scene& scene) override;

    void render(const camera::Camera& cam, FrameBuffer& fb_target) override;

   private:
    const Scene* scene_;
    std::vector<Eigen::Vector2f> projected_pixels_;
    std::vector<float> projected_depths_;

    std::vector<float> projected_opacities_;
    std::vector<Eigen::Vector3f> projected_colors_;
};