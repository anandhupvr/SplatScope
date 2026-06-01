#include "window.h"

#include <stdexcept>

Window::Window(int width, int height, const char* title) : width_(width), height_(height) {
    glfwInit();

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    glfw_window_ = glfwCreateWindow(width_, height_, title, NULL, NULL);

    glfwMakeContextCurrent(glfw_window_);

    // int fb_w, fb_h;
    // glfwGetFramebufferSize(glfw_window_, &fb_w, &fb_h);
    // glViewport(0, 0, fb_w, fb_h);

    glfwSetWindowUserPointer(glfw_window_, this);
    glfwSetKeyCallback(glfw_window_, [](GLFWwindow* w, int key, int, int action, int) {
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
            glfwSetWindowShouldClose(w, true);
    });

    glfwSetFramebufferSizeCallback(glfw_window_,
                                   [](GLFWwindow*, int w, int h) { glViewport(0, 0, w, h); });
    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
        throw std::runtime_error("Failed to initialize GLAD");
    }
}

Window::~Window() {
    if (glfw_window_) {
        glfwDestroyWindow(glfw_window_);
    }
    glfwTerminate();
}

bool Window::should_close() const {
    return glfwWindowShouldClose(glfw_window_);
}

int Window::width() const {
    return width_;
}

int Window::height() const {
    return height_;
}

void Window::swap_buffers() {
    glfwSwapBuffers(glfw_window_);
}

void Window::poll_events() {
    glfwPollEvents();
}

void Window::key_callback(int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(glfw_window_, GL_TRUE);
    }
}