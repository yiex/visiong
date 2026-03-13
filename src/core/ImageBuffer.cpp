// SPDX-License-Identifier: LGPL-3.0-or-later
#include "visiong/core/ImageBuffer.h"

#include "visiong/core/BufferStateMachine.h"
#include "visiong/core/RgaHelper.h"

#include "rk_mpi_mb.h"

#include <utility>

namespace {

// Construct from RK MPI MB_BLK (zero-copy); release via shared_ptr deleter. / Construct from RK MPI MB_BLK (零拷贝); release via shared_ptr deleter.
struct MbBlkReleaser {
    MB_BLK blk;
    void operator()(void*) const {
        if (blk != MB_INVALID_HANDLE)
            RK_MPI_MB_ReleaseMB(blk);
    }
};

}  // namespace

ImageBuffer::ImageBuffer()
    : m_mb_blk_handle_sptr(nullptr), m_dma_buf_sptr(nullptr), m_vir_addr(nullptr), m_size(0), m_user_data(),
      m_is_zero_copy(false), m_external_keep_alive(nullptr), m_mb_blk(MB_INVALID_HANDLE), m_dma_fd(-1),
      m_cached_bgr(nullptr), m_cached_gray(nullptr), width(0), height(0), w_stride(0), h_stride(0),
      format(RK_FMT_BUTT) {}

ImageBuffer::ImageBuffer(int w, int h, PIXEL_FORMAT_E fmt, std::vector<unsigned char>&& data)
    : m_mb_blk_handle_sptr(nullptr), m_dma_buf_sptr(nullptr), m_vir_addr(nullptr), m_size(data.size()),
      m_user_data(std::move(data)), m_is_zero_copy(false), m_external_keep_alive(nullptr),
      m_mb_blk(MB_INVALID_HANDLE), m_dma_fd(-1), m_cached_bgr(nullptr), m_cached_gray(nullptr), width(w),
      height(h), w_stride(w), h_stride(h), format(fmt) {}

ImageBuffer::ImageBuffer(int w, int h, PIXEL_FORMAT_E fmt, void* ptr, size_t size,
                         std::shared_ptr<void> keep_alive)
    : m_mb_blk_handle_sptr(nullptr), m_dma_buf_sptr(nullptr), m_vir_addr(ptr), m_size(size), m_user_data(),
      m_is_zero_copy(true), m_external_keep_alive(std::move(keep_alive)), m_mb_blk(MB_INVALID_HANDLE),
      m_dma_fd(-1), m_cached_bgr(nullptr), m_cached_gray(nullptr), width(w), height(h), w_stride(w),
      h_stride(h), format(fmt) {}

ImageBuffer::ImageBuffer(int w, int h, PIXEL_FORMAT_E fmt, MB_BLK mb_blk)
    : m_mb_blk_handle_sptr(nullptr), m_dma_buf_sptr(nullptr), m_vir_addr(nullptr), m_size(0), m_user_data(),
      m_is_zero_copy(true), m_external_keep_alive(nullptr), m_mb_blk(mb_blk), m_dma_fd(-1),
      m_cached_bgr(nullptr), m_cached_gray(nullptr), width(w), height(h), w_stride(w), h_stride(h),
      format(fmt) {
    if (mb_blk != MB_INVALID_HANDLE) {
        m_vir_addr = RK_MPI_MB_Handle2VirAddr(mb_blk);
        m_size = RK_MPI_MB_GetSize(mb_blk);
        m_mb_blk_handle_sptr = std::shared_ptr<void>(nullptr, MbBlkReleaser{mb_blk});
        const int fd = RK_MPI_MB_Handle2Fd(mb_blk);
        if (fd >= 0) {
            m_dma_fd = fd;
            visiong::bufstate::adopt_dma(*this,
                                         visiong::bufstate::BufferOwner::External,
                                         visiong::bufstate::CacheDomain::Unknown);
        }
    }
}

ImageBuffer::ImageBuffer(int w, int h, PIXEL_FORMAT_E fmt, std::shared_ptr<RgaDmaBuffer> dma_buf)
    : m_mb_blk_handle_sptr(nullptr), m_dma_buf_sptr(std::move(dma_buf)), m_vir_addr(nullptr), m_size(0),
      m_user_data(), m_is_zero_copy(true), m_external_keep_alive(nullptr), m_mb_blk(MB_INVALID_HANDLE),
      m_dma_fd(-1), m_cached_bgr(nullptr), m_cached_gray(nullptr), width(w), height(h), w_stride(0),
      h_stride(0), format(fmt) {
    if (m_dma_buf_sptr) {
        m_vir_addr = m_dma_buf_sptr->get_vir_addr();
        m_size = m_dma_buf_sptr->get_size();
        m_dma_fd = m_dma_buf_sptr->get_fd();
        w_stride = m_dma_buf_sptr->get_wstride();
        h_stride = m_dma_buf_sptr->get_hstride();
        if (m_dma_fd >= 0) {
            visiong::bufstate::adopt_dma(*this,
                                         visiong::bufstate::BufferOwner::External,
                                         visiong::bufstate::CacheDomain::Unknown);
        }
    }
}

ImageBuffer::~ImageBuffer() {}

ImageBuffer::ImageBuffer(const ImageBuffer& other)
    : m_mb_blk_handle_sptr(other.m_mb_blk_handle_sptr), m_dma_buf_sptr(other.m_dma_buf_sptr),
      m_vir_addr(other.m_vir_addr), m_size(other.m_size), m_user_data(other.m_user_data),
      m_is_zero_copy(other.m_is_zero_copy), m_external_keep_alive(other.m_external_keep_alive),
      m_mb_blk(other.m_mb_blk), m_dma_fd(other.m_dma_fd), m_cached_bgr(nullptr), m_cached_gray(nullptr),
      width(other.width), height(other.height), w_stride(other.w_stride), h_stride(other.h_stride),
      format(other.format) {}

ImageBuffer& ImageBuffer::operator=(const ImageBuffer& other) {
    if (this != &other) {
        m_mb_blk_handle_sptr = other.m_mb_blk_handle_sptr;
        m_dma_buf_sptr = other.m_dma_buf_sptr;
        m_vir_addr = other.m_vir_addr;
        m_size = other.m_size;
        m_user_data = other.m_user_data;
        m_is_zero_copy = other.m_is_zero_copy;
        m_external_keep_alive = other.m_external_keep_alive;
        m_mb_blk = other.m_mb_blk;
        m_dma_fd = other.m_dma_fd;
        width = other.width;
        height = other.height;
        w_stride = other.w_stride;
        h_stride = other.h_stride;
        format = other.format;
        m_cached_bgr = nullptr;
        m_cached_gray = nullptr;
    }
    return *this;
}

ImageBuffer::ImageBuffer(ImageBuffer&& other) noexcept
    : m_mb_blk_handle_sptr(std::move(other.m_mb_blk_handle_sptr)),
      m_dma_buf_sptr(std::move(other.m_dma_buf_sptr)), m_vir_addr(other.m_vir_addr), m_size(other.m_size),
      m_user_data(std::move(other.m_user_data)), m_is_zero_copy(other.m_is_zero_copy),
      m_external_keep_alive(std::move(other.m_external_keep_alive)), m_mb_blk(other.m_mb_blk),
      m_dma_fd(other.m_dma_fd), m_cached_bgr(std::move(other.m_cached_bgr)),
      m_cached_gray(std::move(other.m_cached_gray)), width(other.width), height(other.height),
      w_stride(other.w_stride), h_stride(other.h_stride), format(other.format) {
    other.width = 0;
    other.height = 0;
    other.m_vir_addr = nullptr;
    other.m_size = 0;
    other.m_external_keep_alive.reset();
    other.m_mb_blk = MB_INVALID_HANDLE;
    other.m_dma_fd = -1;
}

ImageBuffer& ImageBuffer::operator=(ImageBuffer&& other) noexcept {
    if (this != &other) {
        m_mb_blk_handle_sptr = std::move(other.m_mb_blk_handle_sptr);
        m_dma_buf_sptr = std::move(other.m_dma_buf_sptr);
        m_vir_addr = other.m_vir_addr;
        m_size = other.m_size;
        m_user_data = std::move(other.m_user_data);
        m_is_zero_copy = other.m_is_zero_copy;
        m_external_keep_alive = std::move(other.m_external_keep_alive);
        m_mb_blk = other.m_mb_blk;
        m_dma_fd = other.m_dma_fd;
        width = other.width;
        height = other.height;
        w_stride = other.w_stride;
        h_stride = other.h_stride;
        format = other.format;
        m_cached_bgr = std::move(other.m_cached_bgr);
        m_cached_gray = std::move(other.m_cached_gray);
        other.width = 0;
        other.height = 0;
        other.m_vir_addr = nullptr;
        other.m_size = 0;
        other.m_external_keep_alive.reset();
        other.m_mb_blk = MB_INVALID_HANDLE;
        other.m_dma_fd = -1;
    }
    return *this;
}

const void* ImageBuffer::get_data() const {
    return m_is_zero_copy ? m_vir_addr : m_user_data.data();
}
void* ImageBuffer::get_data() {
    return m_is_zero_copy ? m_vir_addr : m_user_data.data();
}
size_t ImageBuffer::get_size() const {
    return m_size;
}
bool ImageBuffer::is_valid() const {
    if (width <= 0 || height <= 0) {
        return false;
    }
    if (m_size <= 0) {
        return false;
    }
    if (m_is_zero_copy) {
        return m_vir_addr != nullptr;
    } else {
        return !m_user_data.empty() && m_user_data.data() != nullptr;
    }
}

