#pragma once

namespace rasterizer {

struct SceneBuffers {
    void* d_means = nullptr;
    void* d_scales = nullptr;
    void* d_rotations = nullptr;
    void* d_colors = nullptr;
    void* d_opacities = nullptr;
    int N = 0;
};

struct ScratchBuffers {
    void* d_proj_xy = nullptr;
    void* d_framebuf = nullptr;
    int width, height;
};

void upload(SceneBuffers& scene,
            const float* h_means,
            const float* h_scales,
            const float* h_rotations,
            const float* h_color,
            const float* h_opacity,
            const int size);

void forward(const SceneBuffers& scene,
             const ScratchBuffers& scratch,
             const float* h_view,
             const float* h_K);
};  // namespace rasterizer