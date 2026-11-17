#pragma once
#include "glm/ext/matrix_float3x3.hpp"
#include "glm/matrix.hpp"
#include <cuda.h>  // inly because of glm, fix?
#include <cuda_runtime.h>
#include <vector_types.h>

#include <glm/glm.hpp>

namespace gmath {

// view is a 4x4 column-major matrix (Eigen default from vm.data())
// computes V * (p.x, p.y, p.z, 1)^T and returns xyz/w
__device__ inline float3 world_to_camera(float4 p, const float* __restrict__ view) {
    float x = view[0] * p.x + view[4] * p.y + view[8] * p.z + view[12];
    float y = view[1] * p.x + view[5] * p.y + view[9] * p.z + view[13];
    float z = view[2] * p.x + view[6] * p.y + view[10] * p.z + view[14];
    float w = view[3] * p.x + view[7] * p.y + view[11] * p.z + view[15];
    float inv_w = 1.0f / (w + 1e-7f);
    return make_float3(x * inv_w, y * inv_w, z * inv_w);
}

// project the world-space 3D covariance to a 2D screen covariance: J * W * Σ * W^T * J^T
// returns the symmetric 2x2 as {a, b, c} = [[a, b], [b, c]]
__device__ inline float3 project_cov2d(const float* __restrict__ cov3d,
                                       float3 cam,
                                       const float* __restrict__ view,
                                       float fx,
                                       float fy,
                                       float lim_x,
                                       float lim_y) {
    // compute jacobian
    float inv_z = 1.0f / cam.z;
    float inv_z2 = inv_z * inv_z;
    // linearize as if the Gaussian sat at most 1.3x outside the view edge (shape only)
    const float tx = fminf(lim_x, fmaxf(-lim_x, cam.x * inv_z)) * cam.z;
    const float ty = fminf(lim_y, fmaxf(-lim_y, cam.y * inv_z)) * cam.z;

    // padding row was padding(before this transpose)
    glm::mat3 J = glm::mat3(fx * inv_z,
                            0.0f,
                            0.0f,  // col 0
                            0.0f,
                            fy * inv_z,
                            0.0f,  // col 1
                            -fx * tx * inv_z2,
                            -fy * ty * inv_z2,
                            0.0f  // col 2
    );

    // reconstruct metrices before transform
    // view is column-major 4x4, top-left 3x3 is the rotation
    glm::mat3 W = glm::mat3(view[0],
                            view[1],
                            view[2],  // col 0
                            view[4],
                            view[5],
                            view[6],  // col 1
                            view[8],
                            view[9],
                            view[10]  // col 2
    );

    // reconstruct symmetric 3x3 from 6 stored values
    glm::mat3 sigma = glm::mat3(cov3d[0],
                                cov3d[1],
                                cov3d[2],  // col 0: [c00, c01, c02]
                                cov3d[1],
                                cov3d[3],
                                cov3d[4],  // col 1: [c01, c11, c12]
                                cov3d[2],
                                cov3d[4],
                                cov3d[5]  // col 2: [c02, c12, c22]
    );
    // transform cov3d world to camera

    glm::mat3 sigma_cam = W * sigma * glm::transpose(W);

    // J * E3d * J^T

    // J above is the real Jacobian (glm takes columns), so J Σ J^T, not J^T Σ J
    glm::mat3 cov2d_mat = J * sigma_cam * glm::transpose(J);

    /*
    [a, b, 0]
    [b, c, 0]
    [0, 0, 0]
    */
    // return a, b, c (symmetric matrix)
    return {float(cov2d_mat[0][0]), float(cov2d_mat[1][0]), float(cov2d_mat[1][1])};
}
}  // namespace gmath