#pragma once
#include "renderer/renderer.h"
#include "scene/scene.h"

class CpuRenderer : public Renderer {
   public:
    void set_scene(const Scene& scene) override;

    void render(const camera::Camera& cam, FrameBuffer& fb_target) override;

   private:
    const Scene* scene_;
};