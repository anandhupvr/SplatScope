#include "camera/controls.h"
// namespace gs::camera::controls {

namespace {                                  // sensitivities — tune these to taste
constexpr float kOrbitSensitivity = 0.005f;  // radians per pixel
constexpr float kPanSensitivity = 0.01f;     // world units per pixel
constexpr float kZoomSensitivity = 0.1f;     // per scroll unit
}  // namespace

void update_camera(Camera& cam, const InputState& in) {
    // Left drag → orbit
    if (in.mouse_left_down) {
        cam.orbit(static_cast<float>(in.mouse_dx) * kOrbitSensitivity,
                  static_cast<float>(in.mouse_dy) * kOrbitSensitivity);
    }

    // Right drag → pan
    // if (in.mouse_right_down) {
    //     cam.pan({in.mouse_dx * kPanSensitivity, in.mouse_dy * kPanSensitivity});
    // }
    // Scroll → zoom
    if (in.scroll_dy != 0.0) {
        // Negative scroll usually means "zoom in" — multiplicative
        float factor = std::exp(static_cast<float>(-in.scroll_dy) * kZoomSensitivity);
        cam.zoom(factor);
    }
}

// }  // namespace gs::camera::controls