cmake_minimum_required(VERSION 3.30)

project(SplatScope VERSION 0.1.0 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type (Debug|Release|RelWithDebInfo)" FORCE)
endif()


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


find_package(glm CONFIG REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(EIGEN3 REQUIRED eigen3)

add_subdirectory(external/tinply)

set(SOURCES
    src/main.cpp
    src/scene/ply_loader.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME}
    PRIVATE
    src
    ${EIGEN3_INCLUDE_DIRS}
    external/tinply/source
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
    ${EIGEN3_LIBRARIES}
    tinyply
    glm::glm
)