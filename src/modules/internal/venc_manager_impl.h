// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_MODULES_INTERNAL_VENC_MANAGER_IMPL_H
#define VISIONG_MODULES_INTERNAL_VENC_MANAGER_IMPL_H

#include <mutex>

#include "rk_comm_venc.h"
#include "rk_mpi_mb.h"
#include "visiong/modules/VencManager.h"

struct VencManagerImpl {
    static constexpr int kVencChannelId = 0;

    std::mutex encode_mutex;
    mutable std::mutex mutex;
    bool is_initialized = false;
    VencConfig current_config;
    int user_count = 0;
    int vir_width = 0;
    int vir_height = 0;
    MB_POOL input_pool = MB_INVALID_POOLID;
};

#endif  // VISIONG_MODULES_INTERNAL_VENC_MANAGER_IMPL_H

