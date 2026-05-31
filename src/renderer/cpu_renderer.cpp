#include "renderer/cpu_renderer.h"

#include "splat/splat.h"

void CpuRenderer::set_scene(const Scene& scene) {
    scene_ = &scene;
    // reserve for projected splats
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

    auto cloud = scene_->gaussians();
    auto view_mat = cam.view_matrix();
    auto intrinsic = cam.instrinsic_as_matrix();

    for (size_t i = 0; i < 1000; ++i) {
        // TODO : render first 1000 points for testing

        // project to screen space
        auto cam_pos = splat::world_to_camera(cloud.mean[i], view_mat);
        auto point_2d = splat::camera_to_screen(cam_pos, intrinsic);
    }

    // cull

    // sort (by depth)

    // rasterize
}