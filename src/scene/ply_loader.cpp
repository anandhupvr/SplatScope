#include "scene/ply_loader.h"

#include <iostream>
#include <stdexcept>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "example-utils.hpp"
#include "scene/gaussian_cloud.h"
#include "tinyply.h"

using namespace tinyply;

struct TempGaussian {
    Eigen::Vector3f position;
    Eigen::Vector3f scale;
    float opacity;
    Eigen::Quaternionf rotation;
    Eigen::Vector3f color;
};

double sigmoid(float x) {
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

    file_stream->seekg(0, std::ios::end);
    file_stream->seekg(0, std::ios::beg);

    PlyFile file;
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

    size_t skipped = 0;

    // TODO
    // handle nan values
    GaussianCloud decoded_cloud;
    for (int i = 0; i < raw_size; i++) {
        // validate each gaussian
        // if not valid skip it
    }

    apply_3d_gs_decoding(decoded_cloud);
    std::cout << "Loaded " << decoded_cloud.mean.size() << " gaussians from " << file_path << "\n";

    return decoded_cloud;
}

}  // namespace scene
