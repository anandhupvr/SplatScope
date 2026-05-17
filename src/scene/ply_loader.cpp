#include "scene/ply_loader.h"
#include <stdexcept>
#include <string_view>
#include <vector>
#include "Eigen/Core"
#include "Eigen/Geometry"
#include "scene/gaussian_cloud.h"
#include "tinyply.h"
#include "example-utils.hpp"
#include <iostream>

using namespace tinyply;

std::vector<Eigen::Vector3f> extractVec3Data(const std::shared_ptr<PlyData>& plyData) {
    std::vector<Eigen::Vector3f> result;

    if (!plyData || plyData->count == 0) return result;

    result.reserve(plyData->count);

    if (plyData->t == tinyply::Type::FLOAT32) {
        const float* data = reinterpret_cast<const float*>(plyData->buffer.get());
        for (size_t i = 0; i < plyData->count; i++) {
            result.emplace_back(data[i * 3], data[i * 3 + 1], data[i * 3 + 2]);
        }
    }

    return result;
}

std::vector<float> extractFloatData(const std::shared_ptr<PlyData>& plyData) {
        std::vector<float> result;
        if (!plyData || plyData->count == 0) return result;
        
        result.reserve(plyData->count);
        
        if (plyData->t == tinyply::Type::FLOAT32) {
            const float* data = reinterpret_cast<const float*>(plyData->buffer.get());
            result.assign(data, data + plyData->count);
        } else if (plyData->t == tinyply::Type::FLOAT64) {
            const double* data = reinterpret_cast<const double*>(plyData->buffer.get());
            for (size_t i = 0; i < plyData->count; i++) {
                result.push_back(static_cast<float>(data[i]));
            }
        }
        
        return result;
}


std::vector<Eigen::Quaternionf> extractQuaternionData(const std::shared_ptr<PlyData>& plyData) {
    std::vector<Eigen::Quaternionf> result;

    if (!plyData || plyData->count == 0) return result;

    result.reserve(plyData->count);

    if (plyData->t == tinyply::Type::FLOAT32) {
        const float* data = reinterpret_cast<const float*>(plyData->buffer.get());
        for (size_t i = 0; i < plyData->count; i++) {
            result.emplace_back(
                data[i * 4],     // x
                data[i * 4 + 1], // y
                data[i * 4 + 2], // z
                data[i * 4 + 3]  // w
            );
        }
    }

    return result;
}

GaussianCloud load(const std::string& file_path) {

    const bool preload_into_memory = true;


    
    std::unique_ptr<std::istream> file_stream;
    std::vector<uint8_t> byte_buffer;

    try
    {
        // For most files < 1gb, pre-loading the entire file upfront and wrapping it into a 
        // stream is a net win for parsing speed, about 40% faster. 
        if (preload_into_memory)
        {
            byte_buffer = read_file_binary(file_path);
            file_stream.reset(new memory_stream((char*)byte_buffer.data(), byte_buffer.size()));
        }
        else
        {
            file_stream.reset(new std::ifstream(file_path, std::ios::binary));
        }

        if (!file_stream || file_stream->fail()) throw std::runtime_error("file_stream failed to open " + file_path);

        file_stream->seekg(0, std::ios::end);
        const float size_mb = file_stream->tellg() * float(1e-6);
        file_stream->seekg(0, std::ios::beg);

        PlyFile file;
        file.parse_header(*file_stream);

        // std::cout << "\t[ply_header] Type: " << (file.is_binary_file() ? "binary" : "ascii") << std::endl;
        // for (const auto & c : file.get_comments()) std::cout << "\t[ply_header] Comment: " << c << std::endl;
        // for (const auto & c : file.get_info()) std::cout << "\t[ply_header] Info: " << c << std::endl;

        GaussianCloud guassian;
        for (const auto & e : file.get_elements()) {
            if (e.name == "vertex") {
                std::cout << "\n Vertex Element (" << e.size << " points)" << std::endl;
            }
        }

        std::vector<float> positions_x, positions_y, positions_z;
        std::vector<float> colors_0, colors_1, colors_2;
        std::vector<float> opacitites;
        std::vector<float> scales_0, scales_1, scales_2;
        std::vector<float> rotations_0, rotations_1, rotations_2, rotations_3;

        std::vector<std::shared_ptr<PlyData>> plyData;
        
        try {
            plyData.push_back(file.request_properties_from_element("vertex", {"x", "y", "z"}));
            plyData.push_back(file.request_properties_from_element("vertex", {"f_dc_0", "f_dc_1", "f_dc_2"}));
            plyData.push_back(file.request_properties_from_element("vertex", {"opacity"}));
            plyData.push_back(file.request_properties_from_element("vertex", {"scale_0", "scale_1", "scale_2"}));
            plyData.push_back(file.request_properties_from_element("vertex", {"rot_0", "rot_1", "rot_2", "rot_3"}));
            
        } catch (const std::exception& e) {
            std::cerr << "Warning: " << e.what() << std::endl;
        }

        file.read(*file_stream);

        if (plyData.size() >= 0) {
            guassian.mean = extractVec3Data(plyData[0]);
            guassian.color = extractVec3Data(plyData[1]);
            guassian.opacity = extractFloatData(plyData[2]);
            guassian.scale = extractVec3Data(plyData[3]);
            guassian.rotation = extractQuaternionData(plyData[4]);
        }

        return guassian;
    }
    catch (const std::exception & e)
    {
        std::cerr << "Caught tinyply exception: " << e.what() << std::endl;
        throw std::runtime_error("PLY missing required attribute: " + std::string(e.what()));
    }
    

}
