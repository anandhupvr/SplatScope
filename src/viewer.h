#pragma once
#include "window/window.h"

#include "renderer/renderer.h"
#include "scene/scene.h"

class Viewer {
   public:
    Viewer(const char* scene_path);
    void run();

   private:
    Scene scene_;
    Window window_;
    std::unique_ptr<Renderer> renderer_;
};