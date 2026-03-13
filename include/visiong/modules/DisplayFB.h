// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_MODULES_DISPLAYFB_H
#define VISIONG_MODULES_DISPLAYFB_H

#include <memory>
#include <tuple>

class ImageBuffer;

class DisplayFB {
public:
    enum class Mode {
        LOW_REFRESH,
        HIGH_REFRESH,
    };

    explicit DisplayFB(Mode mode = Mode::HIGH_REFRESH);
    ~DisplayFB();

    DisplayFB(const DisplayFB&) = delete;
    DisplayFB& operator=(const DisplayFB&) = delete;

    void release();
    bool is_initialized() const;

    bool display(const ImageBuffer& img_buf);
    bool display(ImageBuffer&& img_buf);
    bool display(const ImageBuffer& img_buf, const std::tuple<int, int, int, int>& roi);
    bool display(ImageBuffer&& img_buf, const std::tuple<int, int, int, int>& roi);

    int get_screen_width() const;
    int get_screen_height() const;
    int screen_width() const { return get_screen_width(); }
    int screen_height() const { return get_screen_height(); }

private:
    void display_thread_func();

    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

#endif  // VISIONG_MODULES_DISPLAYFB_H
