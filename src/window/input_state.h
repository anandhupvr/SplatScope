#pragma once

// namespace gs::window {

struct InputState {
    // Mouse position (pixels)
    double mouse_x = 0.0;
    double mouse_y = 0.0;

    // Per-frame mouse delta (pixels moved since last frame)
    double mouse_dx = 0.0;
    double mouse_dy = 0.0;

    // Buttons currently held
    bool mouse_left_down = false;
    bool mouse_right_down = false;
    bool mouse_middle_down = false;

    // Scroll wheel: accumulated scroll units since last frame
    double scroll_dy = 0.0;

    // Modifier keys
    bool key_shift = false;
    bool key_ctrl = false;

    // Window framebuffer size (in pixels)
    int fb_width = 0;
    int fb_height = 0;
};

// }  // namespace gs::window