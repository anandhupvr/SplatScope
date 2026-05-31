#pragma once
#include <cstdint>
#include <vector>
class FrameBuffer {
   public:
    FrameBuffer(int width, int height);
    ~FrameBuffer();

    void clear(int r, int g, int b);
    void set_pixel(int x, int y, int r, int g, int b);
    const uint8_t* data() const;

    const int width() const {
        return width_;
    }
    const int height() const {
        return height_;
    }

   private:
    int width_, height_;
    std::vector<uint8_t> pixels_;
};