#pragma once
#include "window/input_state.h"

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

    const InputState& input();

   private:
    static void on_mouse_move_(GLFWwindow*, double, double);
    static void on_mouse_button_(GLFWwindow*, int button, int action, int mods);
    static void on_scroll_(GLFWwindow*, double, double);
    static void on_key_(GLFWwindow*, int key, int sc, int action, int mods);
    static void on_resize_(GLFWwindow*, int w, int h);

    GLFWwindow* glfw_window_ = nullptr;
    int width_, height_;

    InputState input_;

    double last_mouse_x_ = 0.0;
    double last_mouse_y_ = 0.0;
    bool first_mouse_ = true;
};