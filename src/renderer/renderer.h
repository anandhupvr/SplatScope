#pragma once

#include "camera/camera.h"
#include "renderer/framebuffer.h"
#include "scene/scene.h"

class Renderer {
   public:
    virtual ~Renderer() = default;

    virtual void set_scene(const Scene& scene) = 0;

    virtual void render(const Camera& cam, FrameBuffer& fb_target) = 0;
};