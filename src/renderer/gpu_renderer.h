#pragma once
#include "camera/camera.h"
#include "renderer/renderer.h"
#include "scene/scene.h"

class GpuRenderer : public Renderer {
   public:
    void set_scene(const Scene& scene) override;
    void render(const Camera& cam) override;
};