#include "renderer/gpu_renderer.h"

#include "camera/camera.h"
#include "renderer/kernels/gaussian_pipeline.cuh"
#include "scene/scene.h"

void GpuRenderer::set_scene(const Scene& scene) {
    scene_ = &scene;
    rasterizer::upload(scene_bufs_,
                       reinterpret_cast<const float*>(scene_->gaussians().mean.data()),
                       reinterpret_cast<const float*>(scene_->gaussians().scale.data()),
                       reinterpret_cast<const float*>(scene_->gaussians().rotation.data()),
                       reinterpret_cast<const float*>(scene_->gaussians().color.data()),
                       scene_->gaussians().opacity.data(),
                       scene_->gaussians().size());
}

void GpuRenderer::render(const Camera& cam) {
    const int N = scene_->gaussians().size();

    rasterizer::forward(
        scene_bufs_, scratch_bufs_, cam.view_matrix().data(), cam.instrinsic_as_matrix().data());
    // cudaMemcpy D->H-> glTexSubImage2D -> draw quad
}