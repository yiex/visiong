// SPDX-License-Identifier: LGPL-3.0-or-later
#include "common/internal/dma_alloc.h"

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace {

struct dma_heap_allocation_data {
    std::uint64_t len;
    std::uint32_t fd;
    std::uint32_t fd_flags;
    std::uint64_t heap_flags;
};

struct dma_buf_sync {
    std::uint64_t flags;
};

#ifndef DMA_HEAP_IOCTL_ALLOC
#define DMA_HEAP_IOC_MAGIC 'H'
#define DMA_HEAP_IOCTL_ALLOC _IOWR(DMA_HEAP_IOC_MAGIC, 0x0, struct dma_heap_allocation_data)
#endif

#ifndef DMA_BUF_IOCTL_SYNC
#define DMA_BUF_BASE 'b'
#define DMA_BUF_IOCTL_SYNC _IOW(DMA_BUF_BASE, 0, struct dma_buf_sync)
#endif

constexpr std::uint64_t kDmaBufSyncRead = 1u << 0;
constexpr std::uint64_t kDmaBufSyncWrite = 2u << 0;
constexpr std::uint64_t kDmaBufSyncRw = kDmaBufSyncRead | kDmaBufSyncWrite;
constexpr std::uint64_t kDmaBufSyncStart = 0u << 2;
constexpr std::uint64_t kDmaBufSyncEnd = 1u << 2;

int sync_dma_buf_fd(int fd, std::uint64_t flags) {
    dma_buf_sync sync{};
    sync.flags = flags;
    return ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync);
}

}  // namespace

int dma_sync_device_to_cpu(int fd) {
    return sync_dma_buf_fd(fd, kDmaBufSyncStart | kDmaBufSyncRw);
}

int dma_sync_cpu_to_device(int fd) {
    return sync_dma_buf_fd(fd, kDmaBufSyncEnd | kDmaBufSyncRw);
}

int dma_buf_alloc(const char* path, size_t size, int* fd, void** va) {
    if (!path || !fd || !va || size == 0) {
        return -EINVAL;
    }

    const int heap_fd = open(path, O_RDWR | O_CLOEXEC);
    if (heap_fd < 0) {
        return -errno;
    }

    dma_heap_allocation_data alloc{};
    alloc.len = static_cast<std::uint64_t>(size);
    alloc.fd_flags = O_CLOEXEC | O_RDWR;

    if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &alloc) < 0) {
        const int err = errno;
        close(heap_fd);
        return -err;
    }

    close(heap_fd);

    const int prot = (fcntl(static_cast<int>(alloc.fd), F_GETFL) & O_RDWR) ? (PROT_READ | PROT_WRITE) : PROT_READ;
    void* mapped = mmap(nullptr, size, prot, MAP_SHARED, static_cast<int>(alloc.fd), 0);
    if (mapped == MAP_FAILED) {
        const int err = errno;
        close(static_cast<int>(alloc.fd));
        return -err;
    }

    *fd = static_cast<int>(alloc.fd);
    *va = mapped;
    return 0;
}

void dma_buf_free(size_t size, int* fd, void* va) {
    if (va && size > 0) {
        munmap(va, size);
    }
    if (fd && *fd >= 0) {
        close(*fd);
        *fd = -1;
    }
}
