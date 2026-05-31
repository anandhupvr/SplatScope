#pragma once
#include "display.h"
#include "window/window.h"

#include "renderer/framebuffer.h"
#include "renderer/renderer.h"
#include "scene/scene.h"

class Viewer {
   public:
    Viewer(const char* scene_path);
    void run();

   private:
    Scene scene_;
    std::unique_ptr<Renderer> renderer_;
    Window window_;
    Display display_;
    FrameBuffer fb_;
};