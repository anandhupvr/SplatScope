#pragma once

#include <string>
#include "scene/gaussian_cloud.h"

namespace scene {
    GaussianCloud load(const std::string& file_path);
}// namespace scene