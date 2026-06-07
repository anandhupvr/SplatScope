#include "viewer.h"

#include <memory>

#include "camera/camera.h"
#include "camera/controls.h"
#include "renderer/cpu_renderer.h"
#include "scene/ply_loader.h"

Viewer::Viewer(const char* scene_path)
    : scene_(Scene(scene::load(scene_path))),
      window_(Window(800, 600, "SplatScope")),
      renderer_(std::make_unique<CpuRenderer>(window_.width(), window_.height())) {
    renderer_->set_scene(scene_);
}
void Viewer::run() {
    Camera camera_;
    camera_.set_target(Eigen::Vector3f(0.0f, 0.0f, 0.0f));
    camera_.set_distance(5.0f);
    Instrinsic intr;
    intr.fx = 800.0f;
    intr.fy = 800.0f;
    intr.cx = window_.width() / 2.0f;
    intr.cy = window_.height() / 2.0f;
    intr.width = window_.width();
    intr.height = window_.height();
    camera_.set_intrinic(intr);

    while (!window_.should_close()) {
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        window_.poll_events();
        update_camera(camera_, window_.input());
        renderer_->render(camera_);

        window_.swap_buffers();
    }
}