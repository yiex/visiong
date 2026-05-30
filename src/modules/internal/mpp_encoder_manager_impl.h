// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_MODULES_INTERNAL_MPP_ENCODER_MANAGER_IMPL_H
#define VISIONG_MODULES_INTERNAL_MPP_ENCODER_MANAGER_IMPL_H

#include <array>
#include <memory>
#include <mutex>

#include "modules/internal/mpp_encoder_backend.h"
#include "visiong/modules/MppEncoderManager.h"

struct MppEncoderManagerImpl {
    static constexpr int kDefaultMppChannelId = 0;
    static constexpr int kMaxMppChannels = 8;

    struct ChannelState {
        std::mutex encode_mutex;
        bool is_initialized = false;
        bool dedicated = false;
        MppConfig current_config;
        int user_count = 0;
        std::unique_ptr<visiong::mpp::MppEncoderBackend> mpp_backend;
    };

    mutable std::mutex mutex;
    std::array<ChannelState, kMaxMppChannels> channels;

    static constexpr int kMppChannelId = kDefaultMppChannelId;
};

#endif  // VISIONG_MODULES_INTERNAL_MPP_ENCODER_MANAGER_IMPL_H
