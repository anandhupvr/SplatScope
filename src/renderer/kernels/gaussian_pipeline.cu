#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <vector_functions.h>
#include <vector_types.h>

// #include <glm/glm.hpp>

#include "renderer/kernels/cuda_utils.cuh"
#include "renderer/kernels/gaussian_math.cuh"
#include "renderer/kernels/gaussian_pipeline.cuh"

namespace rasterizer {

constexpr int TILE = 16;

// Tiles covered by the square center ± radius, as [mn, mx) clamped to the grid.
// The count pass and the write pass MUST both use this, or writes overflow into the next Gaussian's
// slots.
__device__ inline void get_tile_rect(float2 p, float r, dim3 grid, int2& mn, int2& mx) {
    mn.x = min((int)grid.x, max(0, (int)floorf((p.x - r) / TILE)));
    mn.y = min((int)grid.y, max(0, (int)floorf((p.y - r) / TILE)));
    mx.x = min((int)grid.x, max(0, (int)floorf((p.x + r + TILE - 1) / TILE)));
    mx.y = min((int)grid.y, max(0, (int)floorf((p.y + r + TILE - 1) / TILE)));
}

__global__ void duplicate_and_generate_keys(int N,
                                            const float* __restrict__ d_proj_xy,
                                            const float* __restrict__ d_radii,
                                            const float* __restrict__ d_depths,
                                            const int* __restrict__ d_tiles_touched,
                                            const int* __restrict__ d_point_offsets,
                                            uint64_t* d_keys_unsorted,
                                            uint32_t* d_vals_unsorted,
                                            dim3 tile_grid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;
    if (d_tiles_touched[idx] == 0)  // culled or off screen: owns no slots
        return;

    // my slots are [offsets[idx-1], offsets[idx])
    uint32_t off = (idx == 0) ? 0 : d_point_offsets[idx - 1];
    float2 point = make_float2(d_proj_xy[2 * idx], d_proj_xy[2 * idx + 1]);

    int2 mn, mx;
    get_tile_rect(point, d_radii[idx], tile_grid, mn, mx);

    // depth > near plane > 0, so its raw float bits sort in the same order as the float values
    uint32_t depth_bits = __float_as_uint(d_depths[idx]);

    // key = [ tile_id : 32 | depth bits : 32 ], value = gaussian id
    for (int y = mn.y; y < mx.y; y++) {
        for (int x = mn.x; x < mx.x; x++) {
            uint64_t key = (uint64_t)(y * tile_grid.x + x) << 32;
            d_keys_unsorted[off] = key | depth_bits;
            d_vals_unsorted[off] = idx;
            off++;
        }
    }
    assert(
        off ==
        (uint32_t)d_point_offsets[idx]);  // wrote exactly tiles_touched[idx] entries (Debug only)
}

__global__ void project_gaussians(int N,
                                  const float* __restrict__ d_means,
                                  const float* __restrict__ d_view,
                                  const float* __restrict__ d_cov3d,
                                  const float* __restrict__ d_opacity,
                                  float fx,
                                  float fy,
                                  float cx,
                                  float cy,
                                  float* d_means2d,
                                  float* d_depths,
                                  int* d_tiles_touched,
                                  float* d_radii,
                                  float4* d_conic_opacities,
                                  uchar4* d_framebuf,
                                  int W,
                                  int H,
                                  dim3 tile_grid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    // Write these BEFORE any early return: the prefix sum reads all N slots every frame,
    // and cudaMalloc memory is garbage / last frame's values otherwise.
    d_radii[idx] = 0;
    d_tiles_touched[idx] = 0;

    float4 p = reinterpret_cast<const float4*>(d_means)[idx];
    float3 cam = gmath::world_to_camera(p, d_view);

    if (cam.z < 0.2f)  // near plane
        return;

    // keep subpixel precision; don't cull on the center, a big splat centered off-screen still
    // overlaps it
    float inv_z = 1.0f / cam.z;
    float u = fx * cam.x * inv_z + cx;
    float v = fy * cam.y * inv_z + cy;

    float3 cov2d = gmath::project_cov2d(d_cov3d + 6 * idx, cam, d_view, fx, fy);

    // low-pass filter: ADD 0.3 px² (convolving two Gaussians adds covariances); also keeps it
    // invertible
    float a = cov2d.x + 0.3f;
    float b = cov2d.y;
    float c = cov2d.z + 0.3f;

    float det = a * c - b * b;
    if (det <= 0.0f)
        return;
    float inv_det = 1.0f / det;

    // radius = 3 sigma along the longest axis = 3 * sqrt(largest eigenvalue) of the SAME (dilated)
    // matrix
    float mid = 0.5f * (a + c);
    float lambda1 = mid + sqrtf(fmaxf(0.0f, mid * mid - det));
    float radius = ceilf(3.0f * sqrtf(lambda1));

    int2 mn, mx;
    get_tile_rect(make_float2(u, v), radius, tile_grid, mn, mx);
    int tiles = (mx.x - mn.x) * (mx.y - mn.y);
    if (tiles == 0)  // entirely off screen
        return;

    d_means2d[2 * idx] = u;
    d_means2d[2 * idx + 1] = v;
    d_depths[idx] = cam.z;
    d_radii[idx] = radius;
    d_tiles_touched[idx] = tiles;
    // conic = inverse 2D covariance [[c, -b], [-b, a]] / det, packed with opacity for the render
    // kernel
    d_conic_opacities[idx] = make_float4(c * inv_det, -b * inv_det, a * inv_det, d_opacity[idx]);

    // debug view until the render kernel exists: white dot at the center
    if (u >= 0 && u < W && v >= 0 && v < H)
        d_framebuf[(int)v * W + (int)u] = make_uchar4(255, 255, 255, 255);
}

__global__ void identify_tile_ranges(int L, const uint64_t* keys_sorted, uint2* ranges)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L)
        return;

    uint32_t tile = keys_sorted[idx] >> 32;
    if (idx == 0){
        ranges[tile].x = 0;
    }
    else {
        uint32_t prev = keys_sorted[idx - 1] >> 32;
        if (tile != prev) {
            ranges[prev].y = idx;
            ranges[tile].x = idx;
        }
    } 
    if (idx == L -1 ){
        ranges[tile].y = L;
    }
}

constexpr int BLOCK_SIZE = TILE * TILE;  // 256 threads: one per pixel of a tile

// Block = tile, thread = pixel. Each thread blends its tile's depth-sorted list front to back.
// The list is fetched in batches of BLOCK_SIZE into shared memory, one entry per thread, so each
// entry is read from global memory once per block instead of once per pixel.
__global__ void __launch_bounds__(BLOCK_SIZE)
    render_tiles(const uint2* __restrict__ ranges,
                 const uint32_t* __restrict__ point_list,
                 const float2* __restrict__ xy,
                 const float4* __restrict__ conic_opacity,
                 const float* __restrict__ colors,  // 3 floats (rgb) per Gaussian
                 int W,
                 int H,
                 float3 background,
                 uchar4* __restrict__ out) {
    // who am I
    const int px = blockIdx.x * TILE + threadIdx.x;
    const int py = blockIdx.y * TILE + threadIdx.y;
    const int tile = blockIdx.y * gridDim.x + blockIdx.x;
    const int rank = threadIdx.y * TILE + threadIdx.x;  // 0 .. 255, slot this thread loads
    const bool inside = px < W && py < H;               // bottom tile row hangs off screen
    const float2 pix = make_float2((float)px, (float)py);

    // my tile's slice of the sorted list (same address for all 256 threads -> broadcast)
    const uint2 range = ranges[tile];

    __shared__ uint32_t s_id[BLOCK_SIZE];
    __shared__ float2 s_xy[BLOCK_SIZE];
    __shared__ float4 s_co[BLOCK_SIZE];

    // NO early return from here on: every thread must reach every barrier.
    // Off-screen threads start "done": they help load but skip the blending.
    bool done = !inside;
    float T = 1.0f;  // transmittance: how much light still gets through
    float3 C = make_float3(0.0f, 0.0f, 0.0f);

    // range is the same for the whole block, so every thread runs the same number of rounds
    for (uint32_t base = range.x; base < range.y; base += BLOCK_SIZE) {
        // barrier + count: if every pixel is saturated, the whole block leaves together.
        // Also acts as the "finished reading the previous batch" barrier before we overwrite it.
        if (__syncthreads_count(done) == BLOCK_SIZE)
            break;

        // cooperative load: thread `rank` fetches entry base + rank (coalesced on point_list)
        const uint32_t i = base + rank;
        if (i < range.y) {
            const uint32_t g = point_list[i];
            s_id[rank] = g;
            s_xy[rank] = xy[g];
            s_co[rank] = conic_opacity[g];
        }
        __syncthreads();  // batch fully loaded before anyone reads it

        if (!done) {
            const int count = min(BLOCK_SIZE, (int)(range.y - base));
            for (int j = 0; j < count; j++) {
                const float2 d = make_float2(pix.x - s_xy[j].x, pix.y - s_xy[j].y);
                const float4 co = s_co[j];  // (conic a, conic b, conic c, opacity)
                const float power = -0.5f * (co.x * d.x * d.x + co.z * d.y * d.y) - co.y * d.x * d.y;
                if (power > 0.0f)
                    continue;

                const float alpha = fminf(0.99f, co.w * __expf(power));
                if (alpha < 1.0f / 255.0f)
                    continue;

                const float T_next = T * (1.0f - alpha);
                if (T_next < 1e-4f) {  // saturated: nothing behind can show through
                    done = true;
                    break;
                }

                // color is only fetched for Gaussians that actually contribute
                const uint32_t g = s_id[j];
                const float w = alpha * T;
                C.x += colors[3 * g + 0] * w;
                C.y += colors[3 * g + 1] * w;
                C.z += colors[3 * g + 2] * w;
                T = T_next;
            }
        }
    }

    if (inside) {
        const float r = fminf(fmaxf(C.x + T * background.x, 0.0f), 1.0f);
        const float g = fminf(fmaxf(C.y + T * background.y, 0.0f), 1.0f);
        const float b = fminf(fmaxf(C.z + T * background.z, 0.0f), 1.0f);
        out[py * W + px] = make_uchar4((unsigned char)(r * 255.0f + 0.5f),
                                       (unsigned char)(g * 255.0f + 0.5f),
                                       (unsigned char)(b * 255.0f + 0.5f),
                                       255);
    }
}

void upload(SceneBuffers& scene,
            const float* h_means,
            const float* h_scales,
            const float* h_rotations,
            const float* h_colors,
            const float* h_opacity,
            const float* h_cov3d,
            const int N) {
    // this 32(warp) x 16 (float4) = 512 bytest = 4 transactions and multiple of 128
    const size_t means_bytes = N * 4 * sizeof(float);  // change from 3 -> 4 then pad w=0
    const size_t scales_bytes = N * 3 * sizeof(float);
    const size_t rot_bytes = N * 4 * sizeof(float);
    const size_t col_bytes = N * 3 * sizeof(float);
    const size_t opacity_bytes = N * sizeof(float);
    const size_t cov3d_bytes = N * 6 * sizeof(float);

    CUDA_CHECK(cudaMalloc(&scene.d_means, means_bytes));
    CUDA_CHECK(cudaMalloc(&scene.d_scales, scales_bytes));
    CUDA_CHECK(cudaMalloc(&scene.d_rotations, rot_bytes));
    CUDA_CHECK(cudaMalloc(&scene.d_colors, col_bytes));
    CUDA_CHECK(cudaMalloc(&scene.d_opacities, opacity_bytes));
    CUDA_CHECK(cudaMalloc(&scene.d_cov3d, cov3d_bytes));

    CUDA_CHECK(cudaMemcpy(scene.d_means, h_means, means_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(scene.d_scales, h_scales, scales_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(scene.d_rotations, h_rotations, rot_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(scene.d_colors, h_colors, col_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(scene.d_opacities, h_opacity, opacity_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(scene.d_cov3d, h_cov3d, cov3d_bytes, cudaMemcpyHostToDevice));
    scene.N = N;
    printf("uploaded scene buffer");
}

void forward(const SceneBuffers& scene,
             ScratchBuffers& scratch,
             const float* h_view,
             const float fx,
             const float fy,
             const float cx,
             const float cy,
             uint8_t* frame_buf) {
    // const size_t view_mat_size = 16 * sizeof(float);  // only need 12
    // float* d_view_mat;
    // CUDA_CHECK(cudaMalloc(&d_view_mat, view_mat_size));
    CUDA_CHECK(
        cudaMemset(scratch.d_framebuf, 0, (size_t)scratch.width * scratch.height * sizeof(uchar4)));
    CUDA_CHECK(cudaMemcpy(scratch.d_viewmat, h_view, 16 * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((scene.N + block.x - 1) / block.x);
    dim3 tile_grid((scratch.width + TILE - 1) / TILE, (scratch.height + TILE - 1) / TILE, 1);

    project_gaussians<<<grid, block>>>(scene.N,
                                       static_cast<const float*>(scene.d_means),
                                       static_cast<const float*>(scratch.d_viewmat),
                                       static_cast<const float*>(scene.d_cov3d),
                                       static_cast<const float*>(scene.d_opacities),
                                       fx,
                                       fy,
                                       cx,
                                       cy,
                                       static_cast<float*>(scratch.d_proj_xy),
                                       static_cast<float*>(scratch.d_depths),
                                       static_cast<int*>(scratch.d_tiles_touched),
                                       static_cast<float*>(scratch.d_radii),
                                       static_cast<float4*>(scratch.d_conic_opacities),
                                       static_cast<uchar4*>(scratch.d_framebuf),  // debug/test
                                       scratch.width,
                                       scratch.height,
                                       tile_grid);
    CUDA_CHECK(cudaGetLastError());  // bad launch config; execution errors surface at the next sync

    // prefix sum
    // tiles_touched [3,3,1,0,2] -> point_offset [3, 6, 7, 7, 9]
    int* d_in = static_cast<int*>(scratch.d_tiles_touched);
    int* d_out = static_cast<int*>(scratch.d_point_offsets);
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        scratch.d_cub_temp, scratch.d_cub_temp_bytes, d_in, d_out, scene.N));

    // L = last offset = total (tile, gaussian) pairs. Blocking copy, so it also waits for the work above.
    int L = 0;
    CUDA_CHECK(cudaMemcpy(&L,
                          static_cast<int*>(scratch.d_point_offsets) + scene.N - 1,
                          sizeof(int),
                          cudaMemcpyDeviceToHost));
    // printf("total points to rasterize: %d : gaussian count : %d\n ", L, scene.N);

    // nothing on screen: skip binning (a <<<0, 256>>> launch would be an invalid configuration)
    if (L == 0) {
        CUDA_CHECK(cudaMemcpy(frame_buf,
                              scratch.d_framebuf,
                              (size_t)scratch.width * scratch.height * sizeof(uchar4),
                              cudaMemcpyDeviceToHost));
        return;
    }

    // grow only: buffers live across frames, realloc only when this frame needs more than we have
    if (L > scratch.L_max) {
        cudaFree(scratch.d_keys_unsorted);
        cudaFree(scratch.d_vals_unsorted);
        cudaFree(scratch.d_keys_sorted);
        cudaFree(scratch.d_point_list);

        const int cap = L + L / 2;  // 50% headroom so small camera moves don't realloc
        CUDA_CHECK(cudaMalloc(&scratch.d_keys_unsorted, cap * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&scratch.d_vals_unsorted, cap * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&scratch.d_keys_sorted, cap * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&scratch.d_point_list, cap * sizeof(uint32_t)));
        scratch.L_max = cap;
    }

    // keygeneration (unsorted)
    // eg : keys[t1|3.1, t2|2.0, t3|4.2....]
    // values [G0, G0, G1, G2, G2 ....]
    // using same kernel lauch parameters as projection
    duplicate_and_generate_keys<<<grid, block>>>(scene.N,
                                                 static_cast<const float*>(scratch.d_proj_xy),
                                                 static_cast<const float*>(scratch.d_radii),
                                                 static_cast<const float*>(scratch.d_depths),
                                                 static_cast<const int*>(scratch.d_tiles_touched),
                                                 static_cast<const int*>(scratch.d_point_offsets),
                                                 static_cast<uint64_t*>(scratch.d_keys_unsorted),
                                                 static_cast<uint32_t*>(scratch.d_vals_unsorted),
                                                 tile_grid);
    CUDA_CHECK(cudaGetLastError());
    // faster, tell CUB to ignore bits that are not needed
    // 58x38 = 1900 tiles which is 21bits others are always zero
    int num_tiles = tile_grid.x * tile_grid.y;
    int tile_bits = 0;
    while ((1u << tile_bits) < (unsigned)num_tiles) tile_bits++;
    int end_bit = 32 + tile_bits;  // sort bits [0, 32 + tile_bits]

    // query size
    size_t sort_bytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, sort_bytes,                                          // nullptr - tell me the size
        static_cast<const uint64_t*>(scratch.d_keys_unsorted),
        static_cast<uint64_t*>(scratch.d_keys_sorted),
        static_cast<const uint32_t*>(scratch.d_vals_unsorted),
        static_cast<uint32_t*>(scratch.d_point_list),
        L, 0, end_bit));

    if (sort_bytes > scratch.d_sort_temp_bytes) {                // grow only, never shrink
        cudaFree(scratch.d_sort_temp);
        CUDA_CHECK(cudaMalloc(&scratch.d_sort_temp, sort_bytes));
        scratch.d_sort_temp_bytes = sort_bytes;
    }
    // do the sort
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        scratch.d_sort_temp, scratch.d_sort_temp_bytes,         // ← real buffer = "do it"
        static_cast<const uint64_t*>(scratch.d_keys_unsorted),  // keys in
        static_cast<uint64_t*>(scratch.d_keys_sorted),          // keys out
        static_cast<const uint32_t*>(scratch.d_vals_unsorted),  // values in  (Gaussian IDs)
        static_cast<uint32_t*>(scratch.d_point_list),           // values out (Gaussian IDs, sorted)
        L,                                                      // number of pairs
        0, end_bit));                                           // which key bits to sort on

    // find each tile's slice of the list - Tile ranges
    // clear every frame: tiles with no entries are never written by the kernel, they must read [0, 0)
    CUDA_CHECK(cudaMemset(scratch.d_tile_ranges, 0, num_tiles * sizeof(uint2)));
    
    identify_tile_ranges<<<(L + 255) / 256, 256>>>(
        L,
        static_cast<const uint64_t*>(scratch.d_keys_sorted),
        static_cast<uint2*>(scratch.d_tile_ranges));
    CUDA_CHECK(cudaGetLastError());

    // render: block = tile, thread = pixel; writes every on-screen pixel of the framebuffer
    render_tiles<<<tile_grid, dim3(TILE, TILE)>>>(
        static_cast<const uint2*>(scratch.d_tile_ranges),
        static_cast<const uint32_t*>(scratch.d_point_list),
        static_cast<const float2*>(scratch.d_proj_xy),
        static_cast<const float4*>(scratch.d_conic_opacities),
        static_cast<const float*>(scene.d_colors),
        scratch.width,
        scratch.height,
        make_float3(0.0f, 0.0f, 0.0f),  // black background
        static_cast<uchar4*>(scratch.d_framebuf));
    CUDA_CHECK(cudaGetLastError());

    // debug
    // std::vector<uint64_t> k(std::min(L, 20));
    // cudaMemcpy(k.data(), scratch.d_keys_sorted, k.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    // for (auto key : k) {
    //     uint32_t tile = key >> 32;
    //     uint32_t dbits = key & 0xFFFFFFFF;
    //     float depth; memcpy(&depth, &dbits, 4);
    //     printf("tile %u  depth %.3f\n", tile, depth);
    // }



    // blocking copy: also where errors from inside the kernels above would surface
    CUDA_CHECK(cudaMemcpy(frame_buf,
                          scratch.d_framebuf,
                          (size_t)scratch.width * scratch.height * sizeof(uchar4),
                          cudaMemcpyDeviceToHost));


}

void alloc_scratch(ScratchBuffers& scratch, int N, int W, int H) {
    scratch.width = W;
    scratch.height = H;

    CUDA_CHECK(cudaMalloc(&scratch.d_depths, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&scratch.d_proj_xy, N * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&scratch.d_viewmat, 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&scratch.d_tiles_touched, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&scratch.d_point_offsets, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&scratch.d_radii, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&scratch.d_conic_opacities, N * sizeof(float4)));

    size_t temp_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(nullptr, temp_bytes, (int*)nullptr, (int*)nullptr, N));
    CUDA_CHECK(cudaMalloc(&scratch.d_cub_temp, temp_bytes));
    scratch.d_cub_temp_bytes = temp_bytes;

    // one [start, end) per tile; depends only on the screen size, so it can be allocated once here
    const int num_tiles = ((W + TILE - 1) / TILE) * ((H + TILE - 1) / TILE);
    CUDA_CHECK(cudaMalloc(&scratch.d_tile_ranges, num_tiles * sizeof(uint2)));

    CUDA_CHECK(cudaMalloc(&scratch.d_framebuf, (size_t)W * H * sizeof(uchar4)));
}

void free_scratch(ScratchBuffers& scratch) {
    cudaFree(scratch.d_proj_xy);
    cudaFree(scratch.d_depths);
    cudaFree(scratch.d_viewmat);
    cudaFree(scratch.d_tiles_touched);
    cudaFree(scratch.d_point_offsets);
    cudaFree(scratch.d_radii);
    cudaFree(scratch.d_conic_opacities);
    cudaFree(scratch.d_framebuf);
    cudaFree(scratch.d_cub_temp);

    cudaFree(scratch.d_tile_ranges);

    cudaFree(scratch.d_keys_unsorted);
    cudaFree(scratch.d_vals_unsorted);
    cudaFree(scratch.d_keys_sorted);
    cudaFree(scratch.d_point_list);
    cudaFree(scratch.d_sort_temp);
    scratch.d_keys_unsorted = scratch.d_vals_unsorted = nullptr;
    scratch.d_keys_sorted = scratch.d_point_list = scratch.d_sort_temp = nullptr;
    scratch.L_max = 0;
    scratch.d_sort_temp_bytes = 0;

    scratch.d_tile_ranges = nullptr;

    scratch.d_cub_temp = nullptr;
    scratch.d_proj_xy = nullptr;
    scratch.d_depths = nullptr;
    scratch.d_viewmat = nullptr;
    scratch.d_tiles_touched = nullptr;
    scratch.d_point_offsets = nullptr;
    scratch.d_radii = nullptr;
    scratch.d_conic_opacities = nullptr;
    scratch.d_framebuf = nullptr;
}

void free_scene(SceneBuffers& scene) {
    cudaFree(scene.d_means);
    cudaFree(scene.d_scales);
    cudaFree(scene.d_rotations);
    cudaFree(scene.d_colors);
    cudaFree(scene.d_opacities);
    cudaFree(scene.d_cov3d);
    scene.d_cov3d = nullptr;
    scene.d_means = nullptr;
    scene.d_scales = nullptr;
    scene.d_rotations = nullptr;
    scene.d_colors = nullptr;
    scene.d_opacities = nullptr;
    scene.N = 0;
}

};  // namespace rasterizer