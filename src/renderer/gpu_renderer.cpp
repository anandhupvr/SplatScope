#include "renderer/gpu_renderer.h"

#include "Eigen/Core"

#include "camera/camera.h"
#include "renderer/kernels/gaussian_pipeline.cuh"
#include "scene/scene.h"

GpuRenderer::GpuRenderer(int w, int h) : width(w), height(h) {}
GpuRenderer::~GpuRenderer() {
    rasterizer::free_scratch(scratch_bufs_);
    rasterizer::free_scene(scene_bufs_);
}

void GpuRenderer::set_scene(const Scene& scene) {
    scene_ = &scene;
    // leaks if set_scene called twice , tc later , guard ?
    rasterizer::upload(scene_bufs_,
                       reinterpret_cast<const float*>(scene_->gaussians().mean.data()),
                       reinterpret_cast<const float*>(scene_->gaussians().scale.data()),
                       reinterpret_cast<const float*>(scene_->gaussians().rotation.data()),
                       reinterpret_cast<const float*>(scene_->gaussians().color.data()),
                       scene_->gaussians().opacity.data(),
                       scene_->gaussians().size());

    rasterizer::alloc_scratch(scratch_bufs_, scene_bufs_.N, width, height);
}

void GpuRenderer::render(const Camera& cam) {
    const int N = scene_->gaussians().size();
    Eigen::Matrix4f vm = cam.view_matrix();
    auto intrinsic = cam.intrinsic();
    rasterizer::forward(scene_bufs_,
                        scratch_bufs_,
                        vm.data(),
                        intrinsic.fx,
                        intrinsic.fy,
                        intrinsic.cx,
                        intrinsic.cy);
    // cudaMemcpy D->H-> glTexSubImage2D -> draw quad
}