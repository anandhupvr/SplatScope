#pragma once

#include "renderer/framebuffer.h"
#include "renderer/Shader.h"

class Display {
   public:
    Display(int width, int height);
    ~Display();

    void show(const FrameBuffer& fb);

   private:
    Shader shader_;
    unsigned int texture_id_;
    unsigned int vao_, vbo_, ebo_;
};