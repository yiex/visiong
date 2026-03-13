// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_MODULES_INTERNAL_IVE_MEMORY_H
#define VISIONG_MODULES_INTERNAL_IVE_MEMORY_H

#include "rk_mpi_ive.h"
#include "rk_mpi_mb.h"
#include "rk_mpi_sys.h"

#include <cstdint>

namespace visiong::ive_internal {

constexpr RK_U8 kImageAlign = 16;

RK_U16 calc_stride(RK_U32 width, RK_U8 align = kImageAlign);
RK_S32 mmz_alloc(RK_U64* phy_addr, void** vir_addr, RK_U32 size);
RK_S32 mmz_flush_start(void* vir_addr);
RK_S32 mmz_flush_end(void* vir_addr);
RK_S32 mmz_free(void* vir_addr);
RK_S32 create_image(IVE_IMAGE_S* image, IVE_IMAGE_TYPE_E type, RK_U32 width, RK_U32 height);
RK_S32 create_mem_info(IVE_MEM_INFO_S* mem_info, RK_U32 size);

inline void free_mmz(RK_U64& phy_addr, RK_U64& vir_addr) {
    if (vir_addr == 0) {
        phy_addr = 0;
        return;
    }
    mmz_free(reinterpret_cast<void*>(static_cast<std::uintptr_t>(vir_addr)));
    phy_addr = 0;
    vir_addr = 0;
}

}  // namespace visiong::ive_internal

#endif  // VISIONG_MODULES_INTERNAL_IVE_MEMORY_H
