#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>

class Window {
   public:
    Window(int width, int height, const char* title);
    ~Window();

    int width() const;
    int height() const;

    bool should_close() const;
    void swap_buffers();
    void poll_events();
    void key_callback(int key, int scancode, int action, int mods);

   private:
    GLFWwindow* glfw_window_ = nullptr;
    int width_, height_;
};