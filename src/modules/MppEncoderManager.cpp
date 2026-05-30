// SPDX-License-Identifier: LGPL-3.0-or-later

#include "visiong/modules/MppEncoderManager.h"

#include "core/internal/logger.h"
#include "modules/internal/mpp_encoder_manager_impl.h"

#include <algorithm>
#include <utility>

namespace {

bool is_valid_mpp_channel(int channel_id) {
    return channel_id >= 0 && channel_id < MppEncoderManagerImpl::kMaxMppChannels;
}

int normalize_mpp_channel(int channel_id) {
    return is_valid_mpp_channel(channel_id) ? channel_id : MppEncoderManagerImpl::kDefaultMppChannelId;
}

void reset_channel_state(MppEncoderManagerImpl::ChannelState& channel) {
    if (channel.mpp_backend) {
        channel.mpp_backend->reset();
    }
    channel.is_initialized = false;
}

}  // namespace

MppEncoderManager::MppEncoderManager() : m_impl(std::make_unique<MppEncoderManagerImpl>()) {}

MppEncoderManager::~MppEncoderManager() {
    for (int channel_id = 0; channel_id < MppEncoderManagerImpl::kMaxMppChannels; ++channel_id) {
        auto& channel = m_impl->channels[channel_id];
        std::lock_guard<std::mutex> encode_lock(channel.encode_mutex);
        std::lock_guard<std::mutex> lock(m_impl->mutex);
        reset_channel_state(channel);
        channel.current_config = MppConfig();
        channel.user_count = 0;
        channel.dedicated = false;
    }
}

void MppEncoderManager::acquireUser(int channel_id) {
    channel_id = normalize_mpp_channel(channel_id);
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    auto& channel = m_impl->channels[channel_id];
    if (channel.user_count < 0) {
        channel.user_count = 0;
    }
    ++channel.user_count;
}

void MppEncoderManager::releaseUser(int channel_id) {
    channel_id = normalize_mpp_channel(channel_id);
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    auto& channel = m_impl->channels[channel_id];
    if (channel.user_count <= 0) {
        channel.user_count = 0;
        return;
    }
    --channel.user_count;
}

int MppEncoderManager::acquireDedicatedChannel(int preferred_channel) {
    std::lock_guard<std::mutex> lock(m_impl->mutex);

    auto can_reserve = [&](int channel_id) {
        if (!is_valid_mpp_channel(channel_id)) {
            return false;
        }
        const auto& channel = m_impl->channels[channel_id];
        return !channel.dedicated && channel.user_count == 0;
    };

    int selected = -1;
    if (preferred_channel >= 0) {
        if (can_reserve(preferred_channel)) {
            selected = preferred_channel;
        }
    } else {
        for (int channel_id = 0; channel_id < MppEncoderManagerImpl::kMaxMppChannels; ++channel_id) {
            if (can_reserve(channel_id)) {
                selected = channel_id;
                break;
            }
        }
    }

    if (selected < 0) {
        VISIONG_LOG_WARN("MppEncoderManager", "No free dedicated MPP channel is available.");
        return -1;
    }

    auto& channel = m_impl->channels[selected];
    reset_channel_state(channel);
    channel.current_config = MppConfig();
    channel.user_count = 1;
    channel.dedicated = true;
    VISIONG_LOG_DEBUG("MppEncoderManager", "Reserved dedicated MPP channel " << selected << ".");
    return selected;
}

void MppEncoderManager::releaseDedicatedChannel(int channel_id) {
    if (!is_valid_mpp_channel(channel_id)) {
        return;
    }
    auto& channel = m_impl->channels[channel_id];
    std::lock_guard<std::mutex> encode_lock(channel.encode_mutex);
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    channel.dedicated = false;
    if (channel.user_count > 0) {
        --channel.user_count;
    }
    if (channel.user_count <= 0) {
        channel.user_count = 0;
        const bool was_initialized = channel.is_initialized;
        reset_channel_state(channel);
        channel.current_config = MppConfig();
        if (was_initialized) {
            VISIONG_LOG_DEBUG("MppEncoderManager", "MPP channel " << channel_id << " released.");
        }
    }
}

bool MppEncoderManager::requestIDR(bool instant) {
    return requestIDRForChannel(MppEncoderManagerImpl::kDefaultMppChannelId, instant);
}

bool MppEncoderManager::requestIDRForChannel(int channel_id, bool instant) {
    (void)instant;
    if (!is_valid_mpp_channel(channel_id)) {
        return false;
    }
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    const auto& channel = m_impl->channels[channel_id];
    if (!channel.is_initialized || !channel.mpp_backend) {
        return false;
    }
    if (static_cast<MppCodec>(channel.current_config.codec) == MppCodec::JPEG) {
        return false;
    }
    return channel.mpp_backend->requestIDR();
}

bool MppEncoderManager::isInitialized() const {
    return isInitialized(MppEncoderManagerImpl::kDefaultMppChannelId);
}

bool MppEncoderManager::isInitialized(int channel_id) const {
    channel_id = normalize_mpp_channel(channel_id);
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    return m_impl->channels[channel_id].is_initialized;
}

int MppEncoderManager::getWidth() const {
    return getWidth(MppEncoderManagerImpl::kDefaultMppChannelId);
}

int MppEncoderManager::getWidth(int channel_id) const {
    channel_id = normalize_mpp_channel(channel_id);
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    return m_impl->channels[channel_id].current_config.width;
}

int MppEncoderManager::getHeight() const {
    return getHeight(MppEncoderManagerImpl::kDefaultMppChannelId);
}

int MppEncoderManager::getHeight(int channel_id) const {
    channel_id = normalize_mpp_channel(channel_id);
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    return m_impl->channels[channel_id].current_config.height;
}

PIXEL_FORMAT_E MppEncoderManager::getFormat() const {
    return getFormat(MppEncoderManagerImpl::kDefaultMppChannelId);
}

PIXEL_FORMAT_E MppEncoderManager::getFormat(int channel_id) const {
    channel_id = normalize_mpp_channel(channel_id);
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    return m_impl->channels[channel_id].current_config.format;
}

MppCodec MppEncoderManager::getCodec() const {
    return getCodec(MppEncoderManagerImpl::kDefaultMppChannelId);
}

MppCodec MppEncoderManager::getCodec(int channel_id) const {
    channel_id = normalize_mpp_channel(channel_id);
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    return static_cast<MppCodec>(m_impl->channels[channel_id].current_config.codec);
}

int MppEncoderManager::getFps() const {
    return getFps(MppEncoderManagerImpl::kDefaultMppChannelId);
}

int MppEncoderManager::getFps(int channel_id) const {
    channel_id = normalize_mpp_channel(channel_id);
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    return m_impl->channels[channel_id].current_config.fps;
}

MppRcMode MppEncoderManager::getRcMode() const {
    return getRcMode(MppEncoderManagerImpl::kDefaultMppChannelId);
}

MppRcMode MppEncoderManager::getRcMode(int channel_id) const {
    channel_id = normalize_mpp_channel(channel_id);
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    return static_cast<MppRcMode>(m_impl->channels[channel_id].current_config.rc_mode);
}

int MppEncoderManager::getQuality() const {
    return getQuality(MppEncoderManagerImpl::kDefaultMppChannelId);
}

int MppEncoderManager::getQuality(int channel_id) const {
    channel_id = normalize_mpp_channel(channel_id);
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    return m_impl->channels[channel_id].current_config.quality;
}

bool MppEncoderManager::canReconfigure() const {
    return canReconfigure(MppEncoderManagerImpl::kDefaultMppChannelId);
}

bool MppEncoderManager::canReconfigure(int channel_id) const {
    channel_id = normalize_mpp_channel(channel_id);
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    const auto& channel = m_impl->channels[channel_id];
    return !channel.dedicated && channel.user_count <= 1;
}

std::vector<unsigned char> MppEncoderManager::encodeToJpeg(const ImageBuffer& img, int quality) {
    MppEncodedPacket packet;
    if (!encodeToVideo(img, MppCodec::JPEG, quality, packet)) {
        return {};
    }
    return std::move(packet.data);
}

void MppEncoderManager::releaseMppIfUnused() {
    releaseMppIfUnused(MppEncoderManagerImpl::kDefaultMppChannelId);
}

void MppEncoderManager::releaseMppIfUnused(int channel_id) {
    if (!is_valid_mpp_channel(channel_id)) {
        return;
    }
    auto& channel = m_impl->channels[channel_id];
    std::lock_guard<std::mutex> encode_lock(channel.encode_mutex);
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    if (channel.user_count > 0) {
        return;
    }
    const bool was_initialized = channel.is_initialized;
    reset_channel_state(channel);
    channel.current_config = MppConfig();
    if (was_initialized) {
        VISIONG_LOG_DEBUG("MppEncoderManager", "MPP channel " << channel_id << " released by user.");
    }
}
