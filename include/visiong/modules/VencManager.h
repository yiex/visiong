// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_MODULES_VENCMANAGER_H
#define VISIONG_MODULES_VENCMANAGER_H

#include <cstdint>
#include <memory>
#include <vector>

#include "visiong/common/pixel_format.h"

class ImageBuffer;
struct VencManagerImpl;

struct VencConfig {
    int width = 0;
    int height = 0;
    PIXEL_FORMAT_E format = RK_FMT_BUTT;
    int codec = 0;
    int quality = 75;
    int fps = 30;
    int rc_mode = 0;
};

enum class VencCodec { JPEG = 0, H264 = 1, H265 = 2 };
enum class VencRcMode { CBR = 0, VBR = 1 };

struct VencEncodedPacket {
    std::vector<unsigned char> data;
    std::vector<unsigned char> codec_data;
    bool is_keyframe = false;
    uint32_t stream_seq = 0;
    uint32_t pack_count = 0;
    uint32_t pack_capacity = 0;
    uint32_t packs_appended = 0;
};

class VencManager {
  public:
    static VencManager& getInstance() {
        static VencManager instance;
        return instance;
    }

    class ScopedUser {
      public:
        explicit ScopedUser(VencManager& mgr) : m_mgr(mgr) { m_mgr.acquireUser(); }
        ~ScopedUser() { m_mgr.releaseUser(); }
        ScopedUser(const ScopedUser&) = delete;
        ScopedUser& operator=(const ScopedUser&) = delete;

      private:
        VencManager& m_mgr;
    };

    void acquireUser();
    void releaseUser();
    bool requestIDR(bool instant = true);

    bool isInitialized() const;
    int getWidth() const;
    int getHeight() const;
    PIXEL_FORMAT_E getFormat() const;
    VencCodec getCodec() const;
    int getFps() const;
    VencRcMode getRcMode() const;
    int getQuality() const;

    std::vector<unsigned char> encodeToJpeg(const ImageBuffer& img, int quality);
    bool encodeToVideo(const ImageBuffer& img,
                       VencCodec codec,
                       int quality,
                       VencEncodedPacket& out_packet,
                       int fps = 0,
                       VencRcMode rc_mode = VencRcMode::CBR);

    void releaseVencIfUnused();

  private:
    VencManager();
    ~VencManager();
    VencManager(const VencManager&) = delete;
    VencManager& operator=(const VencManager&) = delete;

    bool ensureVencReady(const VencConfig& new_config);
    static int clamp_quality(int quality);
    static int quality_to_qp(int quality);

    std::unique_ptr<VencManagerImpl> m_impl;
};

#endif  // VISIONG_MODULES_VENCMANAGER_H
