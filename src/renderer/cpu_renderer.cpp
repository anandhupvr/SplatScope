#include "renderer/cpu_renderer.h"

#include <algorithm>

#include "iostream"

#include "splat/splat.h"

int to_bytes(float value) {
    return static_cast<int>(std::clamp(value * 255.0f, 0.0f, 255.0f));
}

void CpuRenderer::set_scene(const Scene& scene) {
    scene_ = &scene;
    // reserve for projected splats
    projected_pixels_.reserve(scene_->gaussians().size());
    projected_depths_.reserve(scene_->gaussians().size());
    projected_opacities_.reserve(scene_->gaussians().size());
    projected_colors_.reserve(scene_->gaussians().size());
}

void CpuRenderer::render(const camera::Camera& cam, FrameBuffer& fb_target) {
    // generate checkboard pattern for testing
    // size_t width = fb_target.width();
    // size_t height = fb_target.height();
    // int tile = 10;
    // for (size_t i = 0; i < height; i++) {
    //     for (size_t j = 0; j < width; j++) {
    //         bool white = ((i / tile) % 2 == (j / tile) % 2);
    //         fb_target.set_pixel(j, i, white ? 255 : 0, 0, 0);
    //     }
    // }

    // project
    /*
    inputs : cloud.mean[i](position x, y, z), cloud.color[i](rgb)
            : cloud.opacity[i](0-1)
    second input in camera related : came.view_matrix(), cam.projection_matrix()
    step 1 :  transform 3d point (cloud.mean[i]) to camera space by view matrx
    step 2 : project 3d point (camera space) to 2d screen space by intrinic matrix (projection
            matrix) we get (u, v) and depth (z value in camera space)
            store the depth (z value in camera space) for later use

    */

    // clear previous frame's projected splats
    projected_pixels_.clear();
    projected_depths_.clear();
    projected_opacities_.clear();
    projected_colors_.clear();

    const auto& cloud = scene_->gaussians();
    auto view_mat = cam.view_matrix();
    auto intrinsic = cam.instrinsic_as_matrix();
    size_t width = fb_target.width();
    size_t height = fb_target.height();

    for (size_t i = 0; i < cloud.size(); ++i) {
        // TODO : render first 1000 points for testing

        // project to screen space
        auto cam_pos = splat::world_to_camera(cloud.mean[i], view_mat);
        auto point_2d = splat::camera_to_screen(cam_pos, intrinsic);
        // std::cout << "3D point: " << cloud.mean[i].transpose()
        //           << " -> 2D point: " << point_2d.transpose() << " with depth: " << cam_pos.z()
        //           << "\n";
        projected_pixels_.push_back(point_2d);
        projected_depths_.push_back(cam_pos.z());
        projected_opacities_.push_back(cloud.opacity[i]);
        projected_colors_.push_back(cloud.color[i]);
    }

    // cull
    std::vector<int> visible_;
    for (size_t i = 0; i < projected_pixels_.size(); i++) {
        if (projected_depths_[i] > 0 && projected_depths_[i] < 1000) {
            if (projected_pixels_[i].x() >= 0 && projected_pixels_[i].x() < width &&
                projected_pixels_[i].y() >= 0 && projected_pixels_[i].y() < height) {
                visible_.push_back(i);
            }
        }
    }

    // sort (by depth)

    std::sort(visible_.begin(), visible_.end(), [&](int a, int b) {
        return projected_depths_[a] > projected_depths_[b];
    });

    // rasterize
    for (auto i : visible_) {
        int x = static_cast<int>(projected_pixels_[i].x());
        int y = static_cast<int>(projected_pixels_[i].y());
        // float opacity = projected_opacities_[i];
        Eigen::Vector3f color = projected_colors_[i];

        // add a simple alpha blending

        fb_target.set_pixel(x, y, to_bytes(color.x()), to_bytes(color.y()), to_bytes(color.z()));
    }
    std::cout << "Rendered " << visible_.size() << " points\n";
}