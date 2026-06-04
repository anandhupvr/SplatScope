cmake_minimum_required(VERSION 3.30)

project(SplatScope VERSION 0.1.0 LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type (Debug|Release|RelWithDebInfo)" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# find_package(glm CONFIG REQUIRED)
set(OpenGL_GL_PREFERENCE GLVND)
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
    src/splat/splat.cpp
    src/camera/camera.cpp
    src/camera/controls.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

target_compile_definitions(${PROJECT_NAME} PRIVATE
    SHADER_DIR="${CMAKE_SOURCE_DIR}/src/shaders"
)

target_include_directories(${PROJECT_NAME}
    PRIVATE
    src
    ${EIGEN3_INCLUDE_DIRS}
    external/tinply/source
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
    glfw
    OpenGL::GL
    glad
    ${EIGEN3_LIBRARIES}
    tinyply
    # glm::glm
)