#include "window.h"

#include <stdexcept>

#include "window/input_state.h"

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
    // Store a back-pointer so callbacks can find this Window.
    glfwSetWindowUserPointer(glfw_window_, this);

    glfwSetCursorPosCallback(glfw_window_, &Window::on_mouse_move_);
    glfwSetMouseButtonCallback(glfw_window_, &Window::on_mouse_button_);
    glfwSetScrollCallback(glfw_window_, &Window::on_scroll_);
    // glfwSetKeyCallback(glfw_window_, &Window::on_key_);
    // glfwSetFramebufferSizeCallback(glfw_window_, &Window::on_resize_);

    // glfwGetFramebufferSize(glfw_window_, &input_.fb_width, &input_.fb_height);
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
    // Reset per-frame deltas BEFORE polling.
    input_.mouse_dx = 0.0;
    input_.mouse_dy = 0.0;
    input_.scroll_dy = 0.0;

    glfwPollEvents();
}

void Window::key_callback(int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(glfw_window_, GL_TRUE);
    }
}

void Window::on_mouse_move_(GLFWwindow* w, double x, double y) {
    auto* self = static_cast<Window*>(glfwGetWindowUserPointer(w));

    if (self->first_mouse_) {
        self->last_mouse_x_ = x;
        self->last_mouse_y_ = y;
        self->first_mouse_ = false;
    }

    self->input_.mouse_dx += x - self->last_mouse_x_;
    self->input_.mouse_dy += y - self->last_mouse_y_;
    self->input_.mouse_x = x;
    self->input_.mouse_y = y;
    self->last_mouse_x_ = x;
    self->last_mouse_y_ = y;
}

void Window::on_mouse_button_(GLFWwindow* w, int button, int action, int /*mods*/) {
    auto* self = static_cast<Window*>(glfwGetWindowUserPointer(w));
    bool down = (action == GLFW_PRESS);

    if (button == GLFW_MOUSE_BUTTON_LEFT)
        self->input_.mouse_left_down = down;
    else if (button == GLFW_MOUSE_BUTTON_RIGHT)
        self->input_.mouse_right_down = down;
    else if (button == GLFW_MOUSE_BUTTON_MIDDLE)
        self->input_.mouse_middle_down = down;
}

void Window::on_scroll_(GLFWwindow* w, double /*dx*/, double dy) {
    auto* self = static_cast<Window*>(glfwGetWindowUserPointer(w));
    self->input_.scroll_dy += dy;
}

void Window::on_key_(GLFWwindow* w, int key, int /*sc*/, int action, int /*mods*/) {
    auto* self = static_cast<Window*>(glfwGetWindowUserPointer(w));
    bool down = (action != GLFW_RELEASE);

    if (key == GLFW_KEY_LEFT_SHIFT)
        self->input_.key_shift = down;
    else if (key == GLFW_KEY_LEFT_CONTROL)
        self->input_.key_ctrl = down;
    else if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(w, GLFW_TRUE);
    }
}

void Window::on_resize_(GLFWwindow* w, int width, int height) {
    auto* self = static_cast<Window*>(glfwGetWindowUserPointer(w));
    self->input_.fb_width = width;
    self->input_.fb_height = height;
    glViewport(0, 0, width, height);
}

const InputState& Window::input() {
    return input_;
}