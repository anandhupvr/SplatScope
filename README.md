# SplatScope

A 3D Gaussian Splatting viewer written in C++ and CUDA.

<p align="center">
  <img src="assets/orbit.gif" alt="SplatScope orbiting the Mip-NeRF 360 counter scene" width="640">
</p>

SplatScope renders scenes trained with [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
(Kerbl et al., SIGGRAPH 2023).
It is a learning project contains:  Projection -> Prefix sum -> Duplicate -> Radix sort -> Tile ranges -> Blend


