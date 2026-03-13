// SPDX-License-Identifier: LGPL-3.0-or-later
#include "modules/internal/ive_memory.h"

#include <cerrno>
#include <cstring>
#include <sys/ioctl.h>

#include "rk_mpi_mb.h"
#include "rk_mpi_mmz.h"
#include "rk_mpi_sys.h"

namespace visiong::ive_internal {
namespace {

struct dma_buf_sync {
    unsigned long long flags;
};

#ifndef DMA_BUF_IOCTL_SYNC
#define DMA_BUF_BASE 'b'
#define DMA_BUF_IOCTL_SYNC _IOW(DMA_BUF_BASE, 0, struct dma_buf_sync)
#endif

constexpr unsigned long long kDmaBufSyncRead = 1ull << 0;
constexpr unsigned long long kDmaBufSyncWrite = 2ull << 0;
constexpr unsigned long long kDmaBufSyncRw = kDmaBufSyncRead | kDmaBufSyncWrite;
constexpr unsigned long long kDmaBufSyncStart = 0ull << 2;
constexpr unsigned long long kDmaBufSyncEnd = 1ull << 2;

RK_U32 compute_image_allocation_size(const IVE_IMAGE_S& image) {
    switch (image.enType) {
        case IVE_IMAGE_TYPE_U8C1:
        case IVE_IMAGE_TYPE_S8C1:
            return image.au32Stride[0] * image.u32Height;
        case IVE_IMAGE_TYPE_YUV420SP:
            return image.au32Stride[0] * image.u32Height * 3 / 2;
        case IVE_IMAGE_TYPE_YUV422SP:
            return image.au32Stride[0] * image.u32Height * 2;
        case IVE_IMAGE_TYPE_S16C1:
        case IVE_IMAGE_TYPE_U16C1:
            return image.au32Stride[0] * image.u32Height * sizeof(RK_U16);
        case IVE_IMAGE_TYPE_U8C3_PACKAGE:
            return image.au32Stride[0] * image.u32Height * 3;
        case IVE_IMAGE_TYPE_S32C1:
        case IVE_IMAGE_TYPE_U32C1:
            return image.au32Stride[0] * image.u32Height * sizeof(RK_U32);
        case IVE_IMAGE_TYPE_S64C1:
        case IVE_IMAGE_TYPE_U64C1:
            return image.au32Stride[0] * image.u32Height * sizeof(RK_U64);
        default:
            return 0;
    }
}

int sync_mmz_fd(void* vir_addr, unsigned long long flags) {
    MB_BLK handle = RK_MPI_MB_VirAddr2Handle(vir_addr);
    if (handle == MB_INVALID_HANDLE) {
        return RK_FAILURE;
    }

    const int fd = RK_MPI_MMZ_Handle2Fd(handle);
    if (fd < 0) {
        return RK_FAILURE;
    }

    dma_buf_sync sync{};
    sync.flags = flags;
    return ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync);
}

void bind_image_planes(IVE_IMAGE_S* image) {
    if (!image) {
        return;
    }

    switch (image->enType) {
        case IVE_IMAGE_TYPE_YUV420SP:
        case IVE_IMAGE_TYPE_YUV422SP:
            image->au32Stride[1] = image->au32Stride[0];
            image->au64PhyAddr[1] = image->au64PhyAddr[0] + static_cast<RK_U64>(image->au32Stride[0]) * image->u32Height;
            image->au64VirAddr[1] = image->au64VirAddr[0] + static_cast<RK_U64>(image->au32Stride[0]) * image->u32Height;
            break;
        case IVE_IMAGE_TYPE_U8C3_PACKAGE:
            image->au32Stride[1] = image->au32Stride[0];
            image->au32Stride[2] = image->au32Stride[0];
            image->au64PhyAddr[1] = image->au64PhyAddr[0] + 1;
            image->au64PhyAddr[2] = image->au64PhyAddr[1] + 1;
            image->au64VirAddr[1] = image->au64VirAddr[0] + 1;
            image->au64VirAddr[2] = image->au64VirAddr[1] + 1;
            break;
        default:
            break;
    }
}

}  // namespace

RK_U16 calc_stride(RK_U32 width, RK_U8 align) {
    return static_cast<RK_U16>(width + (align - width % align) % align);
}

RK_S32 mmz_alloc(RK_U64* phy_addr, void** vir_addr, RK_U32 size) {
    if (!phy_addr || !vir_addr || size == 0) {
        return RK_FAILURE;
    }

    MB_BLK handle = MB_INVALID_HANDLE;
    const RK_S32 ret = RK_MPI_MMZ_Alloc(&handle, size, RK_MMZ_ALLOC_TYPE_CMA | RK_MMZ_ALLOC_UNCACHEABLE);
    if (ret != RK_SUCCESS) {
        return ret;
    }

    *phy_addr = RK_MPI_MB_Handle2PhysAddr(handle);
    *vir_addr = RK_MPI_MB_Handle2VirAddr(handle);
    return RK_SUCCESS;
}

RK_S32 mmz_flush_start(void* vir_addr) {
    return sync_mmz_fd(vir_addr, kDmaBufSyncStart | kDmaBufSyncRw) == 0 ? RK_SUCCESS : RK_FAILURE;
}

RK_S32 mmz_flush_end(void* vir_addr) {
    return sync_mmz_fd(vir_addr, kDmaBufSyncEnd | kDmaBufSyncRw) == 0 ? RK_SUCCESS : RK_FAILURE;
}

RK_S32 mmz_free(void* vir_addr) {
    if (!vir_addr) {
        return RK_SUCCESS;
    }

    MB_BLK handle = RK_MPI_MB_VirAddr2Handle(vir_addr);
    if (handle == MB_INVALID_HANDLE) {
        return RK_FAILURE;
    }
    RK_MPI_MMZ_Free(handle);
    return RK_SUCCESS;
}

RK_S32 create_image(IVE_IMAGE_S* image, IVE_IMAGE_TYPE_E type, RK_U32 width, RK_U32 height) {
    if (!image) {
        return RK_FAILURE;
    }

    std::memset(image, 0, sizeof(*image));
    image->enType = type;
    image->u32Width = width;
    image->u32Height = height;
    image->au32Stride[0] = calc_stride(width);

    const RK_U32 size = compute_image_allocation_size(*image);
    if (size == 0) {
        return RK_FAILURE;
    }

    void* vir_addr = nullptr;
    const RK_S32 ret = mmz_alloc(&image->au64PhyAddr[0], &vir_addr, size);
    if (ret != RK_SUCCESS) {
        return ret;
    }

    image->au64VirAddr[0] = reinterpret_cast<RK_U64>(vir_addr);
    bind_image_planes(image);
    return RK_SUCCESS;
}

RK_S32 create_mem_info(IVE_MEM_INFO_S* mem_info, RK_U32 size) {
    if (!mem_info || size == 0) {
        return RK_FAILURE;
    }

    std::memset(mem_info, 0, sizeof(*mem_info));
    mem_info->u32Size = size;
    void* vir_addr = nullptr;
    const RK_S32 ret = mmz_alloc(&mem_info->u64PhyAddr, &vir_addr, size);
    if (ret != RK_SUCCESS) {
        return ret;
    }
    mem_info->u64VirAddr = reinterpret_cast<RK_U64>(vir_addr);
    return RK_SUCCESS;
}

}  // namespace visiong::ive_internal
