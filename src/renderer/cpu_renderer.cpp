#include "renderer/cpu_renderer.h"

#include <algorithm>

#include "Eigen/Core"

#include "splat/splat.h"

// int to_bytes(float value) {
//     return static_cast<int>(std::clamp(value * 255.0f, 0.0f, 255.0f));
// }
CpuRenderer::CpuRenderer(int w, int h) : display_(Display(w, h)), fb_(FrameBuffer(w, h)) {}
void CpuRenderer::set_scene(const Scene& scene) {
    scene_ = &scene;
    // reserve for projected splats
    projected_pixels_.reserve(scene_->gaussians().size());
    projected_depths_.reserve(scene_->gaussians().size());
    projected_opacities_.reserve(scene_->gaussians().size());
    projected_colors_.reserve(scene_->gaussians().size());
    projected_bbox_.reserve(scene_->gaussians().size());
    projected_cov2d_inv_.reserve(scene_->gaussians().size());
}

void CpuRenderer::render(const Camera& cam) {
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

    fb_.clear(0, 0, 0);

    // clear previous frame's projected splats
    projected_pixels_.clear();
    projected_depths_.clear();
    projected_opacities_.clear();
    projected_colors_.clear();
    projected_cov2d_inv_.clear();
    projected_bbox_.clear();

    const auto& cloud = scene_->gaussians();
    auto view_mat = cam.view_matrix();
    auto intrinsic = cam.instrinsic_as_matrix();
    size_t width = fb_.width();
    size_t height = fb_.height();
    const auto& cov3d_world = scene_->covarience_3d();

    for (size_t i = 0; i < cloud.size(); ++i) {
        // project to screen space
        auto cam_pos = splat::world_to_camera(cloud.mean[i], view_mat);
        auto point_2d = splat::camera_to_screen(cam_pos, intrinsic);
        auto cov2d = splat::project_covarience(
            cov3d_world[i], cam_pos, view_mat.block(0, 0, 3, 3), intrinsic(0, 0), intrinsic(1, 1));
        // std::cout << "3D point: " << cloud.mean[i].transpose()
        //           << " -> 2D point: " << point_2d.transpose() << " with depth: " <<
        //           cam_pos.z()
        //           << "\n";
        projected_pixels_.push_back(point_2d);
        projected_depths_.push_back(cam_pos.z());
        projected_opacities_.push_back(cloud.opacity[i]);
        projected_colors_.push_back(cloud.color[i]);

        // numberical stablity , regularization
        cov2d(0, 0) += 1e-4f;
        cov2d(1, 1) += 1e-4f;
        projected_cov2d_inv_.push_back(cov2d.inverse());
        auto bbox = splat::compute_bounding_box(cov2d, point_2d, width, height);
        projected_bbox_.push_back(bbox);
    }

    // cull
    std::vector<int> visible_;
    for (size_t i = 0; i < projected_pixels_.size(); i++) {
        if (projected_depths_[i] < 0 || projected_depths_[i] > 1000)
            continue;

        const auto& bbox = projected_bbox_[i];

        if (bbox.x() > width || bbox.y() > height)
            continue;
        if (bbox.z() < 0 || bbox.w() < 0)
            continue;

        visible_.push_back(i);
    }

    // sort (by depth)
    std::sort(visible_.begin(), visible_.end(), [&](int a, int b) {
        return projected_depths_[a] > projected_depths_[b];
    });

    // rasterize
    // for (auto i : visible_) {
    //     int x = static_cast<int>(projected_pixels_[i].x());
    //     int y = static_cast<int>(projected_pixels_[i].y());
    //     // float opacity = projected_opacities_[i];
    //     Eigen::Vector3f color = projected_colors_[i];

    //     fb_.set_pixel(x, y, to_bytes(color.x()), to_bytes(color.y()), to_bytes(color.z()));
    // }

    // rasterization
    // std::cout << "Rendered " << visible_.size() << " points\n";
    for (auto i : visible_) {
        Eigen::Vector4i box = projected_bbox_[i];  // x_min, y_min, x_max, y_max
        auto center = projected_pixels_[i];
        auto cov2d_inv = projected_cov2d_inv_[i];
        auto opacity = projected_opacities_[i];
        auto color = projected_colors_[i];

        for (auto y = box.y(); y < box.w(); y++) {
            for (auto x = box.x(); x <= box.z(); x++) {
                // Mahalanobis: how far in ellipse units
                // d2 = 0 at the center, d2 = 1 at 1-sigma, d3 = 9 boundary
                Eigen::Vector2f delta(x - center.x(), y - center.y());
                float d2 = delta.dot(cov2d_inv * delta);
                if (d2 > 9.0)
                    continue;
                float alpha = std::min(0.99f, opacity * std::exp(-0.5f * d2));

                // only used fc1/2/3, more SH coefficients can be added for better lighting effect
                auto e_color = fb_.get_pixel(x, y);
                Eigen::Vector3f existing_color;
                existing_color << e_color[0] / 255.0f, e_color[1] / 255.0f, e_color[2] / 255.0f;
                Eigen::Vector3f blended = alpha * color + (1 - alpha) * existing_color;
                Eigen::Vector3i final_color = (blended * 255.0f).cast<int>();
                fb_.set_pixel(x, y, final_color.x(), final_color.y(), final_color.z());
            }
        }
    }

    //
    display_.show(fb_);
}