#include "viewer.h"

#include <memory>

#include "camera/camera.h"
#include "camera/controls.h"
#include "renderer/cpu_renderer.h"
#include "scene/ply_loader.h"

Viewer::Viewer(const char* scene_path)
    : scene_(Scene(scene::load(scene_path))),
      window_(Window(800, 600, "SplatScope")),
      display_(Display(window_.width(), window_.height())),
      fb_(FrameBuffer(window_.width(), window_.height())),
      renderer_(std::make_unique<CpuRenderer>()) {
    renderer_->set_scene(scene_);
}
void Viewer::run() {
    Camera camera_;
    camera_.set_target(Eigen::Vector3f(0.0f, 0.0f, 0.0f));
    camera_.set_distance(5.0f);

    while (!window_.should_close()) {
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        window_.poll_events();
        update_camera(camera_, window_.input());
        fb_.clear(0, 0, 0);
        renderer_->render(camera_, fb_);
        display_.show(fb_);

        window_.swap_buffers();
    }
}