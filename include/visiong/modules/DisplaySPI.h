// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_MODULES_DISPLAYSPI_H
#define VISIONG_MODULES_DISPLAYSPI_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>

class ImageBuffer;

struct DisplaySPIConfig {
    int width = 240;
    int height = 320;
    int rotation_degrees = 90;
    int x_offset = 0;
    int y_offset = 0;
    std::string backend = "auto";
    uint32_t speed_hz = 50000000;
    uint32_t source_clock_hz = 200000000;
    uint8_t spi_mode = 0;
    uint8_t bits_per_word = 8;
    std::string dc_pin = "GPIO1_C3";
    std::string reset_pin = "GPIO1_C2";
    std::string backlight_pin;
    bool bgr = false;
    bool invert = false;
    size_t transfer_chunk_size = 4096;
    bool multi_buffering = true;
    size_t buffer_count = 3;
};

class DisplaySPIDevice {
public:
    virtual ~DisplaySPIDevice() = default;

    virtual void release() = 0;
    virtual bool is_initialized() const = 0;

    virtual bool display(const ImageBuffer& img_buf) = 0;
    virtual bool display(ImageBuffer&& img_buf) = 0;
    virtual bool display(const ImageBuffer& img_buf, const std::tuple<int, int, int, int>& roi) = 0;
    virtual bool display(ImageBuffer&& img_buf, const std::tuple<int, int, int, int>& roi) = 0;
    virtual void clear(uint16_t color_rgb565 = 0) = 0;
    virtual void configure_geometry(int width, int height, int rotation_degrees) = 0;

    virtual int get_screen_width() const = 0;
    virtual int get_screen_height() const = 0;
    int screen_width() const { return get_screen_width(); }
    int screen_height() const { return get_screen_height(); }
};

class DisplaySPI : public DisplaySPIDevice {
public:
    DisplaySPI(const std::string& chip_model,
               const std::string& spi_bus_path = "/dev/spidev0.0",
               const DisplaySPIConfig& config = DisplaySPIConfig{});
    ~DisplaySPI() override;

    DisplaySPI(const DisplaySPI&) = delete;
    DisplaySPI& operator=(const DisplaySPI&) = delete;

    void release() override;
    bool is_initialized() const override;

    bool display(const ImageBuffer& img_buf) override;
    bool display(ImageBuffer&& img_buf) override;
    bool display(const ImageBuffer& img_buf, const std::tuple<int, int, int, int>& roi) override;
    bool display(ImageBuffer&& img_buf, const std::tuple<int, int, int, int>& roi) override;
    bool display_area(const ImageBuffer& img_buf, int x, int y);
    bool display_area(const ImageBuffer& img_buf, int x, int y, const std::tuple<int, int, int, int>& roi);
    bool display_area(ImageBuffer&& img_buf, int x, int y);
    bool display_area(ImageBuffer&& img_buf, int x, int y, const std::tuple<int, int, int, int>& roi);
    void clear(uint16_t color_rgb565 = 0) override;
    bool draw_rgb565(int x,
                     int y,
                     int w,
                     int h,
                     const void* data,
                     size_t size_bytes,
                     size_t stride_bytes = 0,
                     bool source_is_native_endian = true);
    bool draw_pixel(int x, int y, uint16_t color_rgb565);
    bool draw_line(int x0, int y0, int x1, int y1, uint16_t color_rgb565, int thickness = 1);
    bool draw_rectangle(int x, int y, int w, int h, uint16_t color_rgb565, int thickness = 1, bool fill = false);
    bool draw_circle(int cx, int cy, int radius, uint16_t color_rgb565, int thickness = 1, bool fill = false);
    bool draw_cross(int cx, int cy, uint16_t color_rgb565, int size = 5, int thickness = 1);
    void configure_geometry(int width, int height, int rotation_degrees) override;

    int get_screen_width() const override;
    int get_screen_height() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

std::unique_ptr<DisplaySPIDevice> create_display_spi_device(const std::string& chip_model,
                                                             const std::string& spi_bus_path,
                                                             const DisplaySPIConfig& config = DisplaySPIConfig{});

#endif  // VISIONG_MODULES_DISPLAYSPI_H
