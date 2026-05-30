// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_MODULES_MPPENCODERMANAGER_H
#define VISIONG_MODULES_MPPENCODERMANAGER_H

#include <cstdint>
#include <memory>
#include <vector>

#include "rk_comm_video.h"
#include "visiong/common/pixel_format.h"

class ImageBuffer;
struct MppEncoderManagerImpl;

struct MppConfig {
    int width = 0;
    int height = 0;
    PIXEL_FORMAT_E format = RK_FMT_BUTT;
    int codec = 0;
    int quality = 75;
    int fps = 30;
    int rc_mode = 0;
};

enum class MppCodec { JPEG = 0, H264 = 1, H265 = 2 };
enum class MppRcMode { CBR = 0, VBR = 1 };

struct MppEncodedPacket {
    std::vector<unsigned char> data;
    std::vector<unsigned char> codec_data;
    bool is_keyframe = false;
    uint32_t stream_seq = 0;
    uint32_t pack_count = 0;
    uint32_t pack_capacity = 0;
    uint32_t packs_appended = 0;
};

class MppEncoderManager {
  public:
    static MppEncoderManager& getInstance() {
        static MppEncoderManager instance;
        return instance;
    }

    class ScopedUser {
      public:
        explicit ScopedUser(MppEncoderManager& mgr, int channel_id = 0) : m_mgr(mgr), m_channel_id(channel_id) {
            m_mgr.acquireUser(m_channel_id);
        }
        ~ScopedUser() { m_mgr.releaseUser(m_channel_id); }
        ScopedUser(const ScopedUser&) = delete;
        ScopedUser& operator=(const ScopedUser&) = delete;

      private:
        MppEncoderManager& m_mgr;
        int m_channel_id;
    };

    void acquireUser(int channel_id = 0);
    void releaseUser(int channel_id = 0);
    int acquireDedicatedChannel(int preferred_channel = -1);
    void releaseDedicatedChannel(int channel_id);
    bool requestIDR(bool instant = true);
    bool requestIDRForChannel(int channel_id, bool instant = true);

    bool isInitialized() const;
    bool isInitialized(int channel_id) const;
    int getWidth() const;
    int getWidth(int channel_id) const;
    int getHeight() const;
    int getHeight(int channel_id) const;
    PIXEL_FORMAT_E getFormat() const;
    PIXEL_FORMAT_E getFormat(int channel_id) const;
    MppCodec getCodec() const;
    MppCodec getCodec(int channel_id) const;
    int getFps() const;
    int getFps(int channel_id) const;
    MppRcMode getRcMode() const;
    MppRcMode getRcMode(int channel_id) const;
    int getQuality() const;
    int getQuality(int channel_id) const;
    bool canReconfigure() const;
    bool canReconfigure(int channel_id) const;

    std::vector<unsigned char> encodeToJpeg(const ImageBuffer& img, int quality);
    bool encodeToVideo(const ImageBuffer& img,
                       MppCodec codec,
                       int quality,
                       MppEncodedPacket& out_packet,
                       int fps = 0,
                       MppRcMode rc_mode = MppRcMode::CBR);
    bool encodeToVideoOnChannel(int channel_id,
                                const ImageBuffer& img,
                                MppCodec codec,
                                int quality,
                                MppEncodedPacket& out_packet,
                                int fps = 0,
                                MppRcMode rc_mode = MppRcMode::CBR);
    bool encodeFrameInfoOnChannel(int channel_id,
                                  const VIDEO_FRAME_INFO_S& frame,
                                  MppCodec codec,
                                  int quality,
                                  MppEncodedPacket& out_packet,
                                  int fps = 0,
                                  MppRcMode rc_mode = MppRcMode::CBR);

    void releaseMppIfUnused();
    void releaseMppIfUnused(int channel_id);

  private:
    MppEncoderManager();
    ~MppEncoderManager();
    MppEncoderManager(const MppEncoderManager&) = delete;
    MppEncoderManager& operator=(const MppEncoderManager&) = delete;

    std::unique_ptr<MppEncoderManagerImpl> m_impl;
};

#endif  // VISIONG_MODULES_MPPENCODERMANAGER_H
