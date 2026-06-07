#pragma once

#include "camera/camera.h"
#include "scene/scene.h"

class Renderer {
   public:
    virtual ~Renderer() = default;

    virtual void set_scene(const Scene& scene) = 0;

    virtual void render(const Camera& cam) = 0;
};