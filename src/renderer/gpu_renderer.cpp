#include "renderer/gpu_renderer.h"

#include "camera/camera.h"
#include "scene/scene.h"

void GpuRenderer::set_scene(const Scene& scene) {}

void GpuRenderer::render(const Camera& cam) {
    // launch_project_cull(...);
    // launch_sor(...);
    // tile
    // launch_rasterize(..);
    // cudaMemcpy D->H-> glTexSubImage2D -> draw quad
}