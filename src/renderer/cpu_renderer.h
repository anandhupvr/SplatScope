#pragma once
#include "display.h"
#include "Eigen/Core"

#include "renderer/framebuffer.h"
#include "renderer/renderer.h"
#include "scene/scene.h"

class CpuRenderer : public Renderer {
   public:
    CpuRenderer(int w, int h);
    void set_scene(const Scene& scene) override;

    void render(const Camera& cam) override;

   private:
    const Scene* scene_;

    Display display_;
    FrameBuffer fb_;

    // make a struct ?
    std::vector<Eigen::Vector2f> projected_pixels_;
    std::vector<float> projected_depths_;
    std::vector<float> projected_opacities_;
    std::vector<Eigen::Vector3f> projected_colors_;
    std::vector<Eigen::Matrix2f> projected_cov2d_inv_;
    std::vector<Eigen::Vector4i> projected_bbox_;
};