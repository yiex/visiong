// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_CORE_RGAHELPER_H
#define VISIONG_CORE_RGAHELPER_H

#include <cstddef>

#include "im2d_version.h"
#include "im2d_type.h"

class RgaDmaBuffer {
public:
    RgaDmaBuffer(int width,
                 int height,
                 int format,
                 int wstride = 0,
                 int hstride = 0,
                 size_t size = 0);
    RgaDmaBuffer(int fd,
                 void* vir_addr,
                 size_t size,
                 int width,
                 int height,
                 int format,
                 int wstride,
                 int hstride);
    ~RgaDmaBuffer();

    RgaDmaBuffer(const RgaDmaBuffer&) = delete;
    RgaDmaBuffer& operator=(const RgaDmaBuffer&) = delete;

    rga_buffer_t get_buffer() const;

    void* get_vir_addr() const { return m_vir_addr; }
    size_t get_size() const { return m_size; }
    int get_wstride() const { return m_wstride; }
    int get_hstride() const { return m_hstride; }
    int get_fd() const { return m_fd; }
    int get_width() const { return m_width; }
    int get_height() const { return m_height; }
    int get_mpi_format() const { return m_mpi_format; }

private:
    int m_width;
    int m_height;
    int m_wstride;
    int m_hstride;
    int m_mpi_format;
    size_t m_size;
    int m_fd;
    rga_buffer_handle_t m_handle;
    void* m_vir_addr;
    bool m_is_owner;
};

#endif  // VISIONG_CORE_RGAHELPER_H

