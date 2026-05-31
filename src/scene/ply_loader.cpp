#include "scene/ply_loader.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "example-utils.hpp"  // later replace with own
#include "tinyply.h"

#include "scene/gaussian_cloud.h"

namespace {

inline float sigmoid(float x) {
    return 1.0 / (1.0 + std::exp(-x));
}

std::vector<Eigen::Vector3f> extract_vec3_data(const tinyply::PlyData& data) {
    if (data.t != tinyply::Type::FLOAT32) {
        throw std::runtime_error("Expected FLOAT32 vec3 attribute in PLY");
    }
    std::vector<Eigen::Vector3f> result;

    result.reserve(data.count);

    const float* raw =
        reinterpret_cast<const float*>(const_cast<tinyply::Buffer&>(data.buffer).get());
    for (size_t i = 0; i < data.count; i++) {
        result.emplace_back(raw[i * 3], raw[i * 3 + 1], raw[i * 3 + 2]);
    }

    return result;
}

std::vector<float> extract_float_data(const tinyply::PlyData& data) {
    std::vector<float> result;

    result.reserve(data.count);

    if (data.t == tinyply::Type::FLOAT32) {
        const float* raw =
            reinterpret_cast<const float*>(const_cast<tinyply::Buffer&>(data.buffer).get());
        result.assign(raw, raw + data.count);
    } else if (data.t == tinyply::Type::FLOAT64) {
        const double* raw =
            reinterpret_cast<const double*>(const_cast<tinyply::Buffer&>(data.buffer).get());
        for (size_t i = 0; i < data.count; i++) {
            result.push_back(static_cast<float>(raw[i]));
        }
    } else {
        throw std::runtime_error("Expected FLOAT32 OR FLOAT64 scaler attribute in PLY");
    }

    return result;
}

std::vector<Eigen::Quaternionf> extract_quaternion_data(const tinyply::PlyData& data) {
    if (data.t != tinyply::Type::FLOAT32) {
        throw std::runtime_error("Expected FLOAT32 quat attribute in PLY");
    }

    std::vector<Eigen::Quaternionf> result;

    result.reserve(data.count);

    const float* raw =
        reinterpret_cast<const float*>(const_cast<tinyply::Buffer&>(data.buffer).get());
    for (size_t i = 0; i < data.count; i++) {
        result.emplace_back(raw[i * 4],      // w
                            raw[i * 4 + 1],  // x
                            raw[i * 4 + 2],  // y
                            raw[i * 4 + 3]   // z
        );
        result.back().normalize();
    }

    return result;
}

void apply_3d_gs_decoding(GaussianCloud& cloud) {
    // scales are in log space , must activate with exp
    for (auto& s : cloud.scale) {
        s = s.array().exp();
    }

    // opacities store in pre-sigmod so, apply sigmoid
    for (auto& o : cloud.opacity) {
        o = sigmoid(o);
    }

    // rotations are already normalize in `extract_quaternion_data`
}

inline bool is_valid_at(const GaussianCloud& cloud, std::size_t i) {
    return cloud.mean[i].allFinite() && cloud.scale[i].allFinite() && cloud.color[i].allFinite() &&
           cloud.rotation[i].coeffs().allFinite() && std::isfinite(cloud.opacity[i]) &&
           cloud.scale[i].minCoeff() > 0.0f && cloud.rotation[i].norm() > 1e-6f;
}
GaussianCloud filter_valid(const GaussianCloud& src, std::size_t& skipped_out) {
    GaussianCloud out;
    const std::size_t n = src.size();
    out.mean.reserve(n);
    out.scale.reserve(n);
    out.color.reserve(n);
    out.opacity.reserve(n);
    out.rotation.reserve(n);

    skipped_out = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (!is_valid_at(src, i)) {
            ++skipped_out;
            continue;
        }
        out.mean.push_back(src.mean[i]);
        out.scale.push_back(src.scale[i]);
        out.color.push_back(src.color[i]);
        out.opacity.push_back(src.opacity[i]);
        out.rotation.push_back(src.rotation[i]);
    }
    return out;
}
}  // namespace

namespace scene {
GaussianCloud load(const std::string& file_path) {
    const bool preload_into_memory = true;

    std::unique_ptr<std::istream> file_stream;
    std::vector<uint8_t> byte_buffer;

    if (preload_into_memory) {
        byte_buffer = read_file_binary(file_path);
        file_stream.reset(new memory_stream((char*)byte_buffer.data(), byte_buffer.size()));
    } else {
        file_stream.reset(new std::ifstream(file_path, std::ios::binary));
    }

    if (!file_stream || file_stream->fail())
        throw std::runtime_error("file_stream failed to open " + file_path);

    tinyply::PlyFile file;
    file.parse_header(*file_stream);

    auto poisitions = file.request_properties_from_element("vertex", {"x", "y", "z"});
    auto colors = file.request_properties_from_element("vertex", {"f_dc_0", "f_dc_1", "f_dc_2"});
    auto opacities = file.request_properties_from_element("vertex", {"opacity"});
    auto scales = file.request_properties_from_element("vertex", {"scale_0", "scale_1", "scale_2"});
    auto rotations =
        file.request_properties_from_element("vertex", {"rot_0", "rot_1", "rot_2", "rot_3"});

    file.read(*file_stream);

    GaussianCloud raw_gaussian;
    raw_gaussian.mean = extract_vec3_data(*poisitions);
    raw_gaussian.color = extract_vec3_data(*colors);
    raw_gaussian.opacity = extract_float_data(*opacities);
    raw_gaussian.scale = extract_vec3_data(*scales);
    raw_gaussian.rotation = extract_quaternion_data(*rotations);

    auto raw_size = raw_gaussian.size();
    apply_3d_gs_decoding(raw_gaussian);

    size_t skipped = 0;

    auto gaussian_cloud = filter_valid(raw_gaussian, skipped);

    std::cout << "Loaded " << gaussian_cloud.size() << " gaussians (skipped " << skipped << ")\n";

    return gaussian_cloud;
}

}  // namespace scene
