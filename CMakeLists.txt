cmake_minimum_required(VERSION 3.30)

project(SplatScope VERSION 0.1.0 LANGUAGES CUDA C CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type (Debug|Release|RelWithDebInfo)" FORCE)
endif()

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# find_package(glm CONFIG REQUIRED)
set(OpenGL_GL_PREFERENCE GLVND)
find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(EIGEN3 REQUIRED eigen3)

add_subdirectory(external/tinply)

add_library(glad STATIC external/glad/src/glad.c)
target_include_directories(glad PUBLIC external/glad/include)

set(SOURCES
    src/main.cpp
    src/viewer.cpp
    src/display.cpp
    src/scene/ply_loader.cpp
    src/scene/scene.cpp
    src/window/window.cpp
    src/renderer/framebuffer.cpp
    src/renderer/Shader.cpp
    src/renderer/cpu_renderer.cpp
    src/renderer/gpu_renderer.cpp
    src/splat/splat.cpp
    src/camera/camera.cpp
    src/camera/controls.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES}
    src/renderer/kernels/gaussian_pipeline.cu)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    SHADER_DIR="${CMAKE_SOURCE_DIR}/src/shaders"
    GLM_FORCE_CUDA
    GLM_FORCE_PURE
    GLM_FORCE_INLINE
    # GLM_COMPILER=0
)

target_include_directories(${PROJECT_NAME}
    BEFORE PRIVATE
    external/glm
    src
    ${EIGEN3_INCLUDE_DIRS}
    external/tinply/source
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
    glfw
    OpenGL::GL
    glad
    CUDA::cudart
    # glm::glm
    ${EIGEN3_LIBRARIES}
    tinyply
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)