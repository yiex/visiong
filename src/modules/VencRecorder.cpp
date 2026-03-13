// SPDX-License-Identifier: LGPL-3.0-or-later
#include "visiong/modules/VencRecorder.h"

#include "visiong/core/ImageBuffer.h"
#include "core/internal/rga_utils.h"
#include "visiong/modules/VencManager.h"
#include "modules/internal/venc_utils.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

// media-server (libmov/libflv) / media-server 相关组件（libmov/libflv）。
#include "mov-format.h"
#include "mp4-writer.h"
#include "mpeg4-avc.h"
#include "mpeg4-hevc.h"

namespace {

static int mov_file_read(void* fp, void* data, uint64_t bytes) {
    if (bytes == fread(data, 1, (size_t)bytes, (FILE*)fp))
        return 0;
    return ferror((FILE*)fp) ? -ferror((FILE*)fp) : -1;
}

static int mov_file_write(void* fp, const void* data, uint64_t bytes) {
    if (bytes == fwrite(data, 1, (size_t)bytes, (FILE*)fp))
        return 0;
    return ferror((FILE*)fp) ? -ferror((FILE*)fp) : -1;
}

static int mov_file_seek(void* fp, int64_t offset) {
    // offset >=0: absolute, offset <0: SEEK_END + offset / offset >=0 表示绝对位置，offset <0 表示 SEEK_END + offset。
#if defined(_WIN32) || defined(_WIN64)
    return _fseeki64((FILE*)fp, offset, offset >= 0 ? SEEK_SET : SEEK_END);
#else
    return fseeko((FILE*)fp, (off_t)offset, offset >= 0 ? SEEK_SET : SEEK_END);
#endif
}

static int64_t mov_file_tell(void* fp) {
#if defined(_WIN32) || defined(_WIN64)
    return (int64_t)_ftelli64((FILE*)fp);
#else
    return (int64_t)ftello((FILE*)fp);
#endif
}

static const struct mov_buffer_t* mov_stdio_buffer() {
    static struct mov_buffer_t s_io = {
        mov_file_read,
        mov_file_write,
        mov_file_seek,
        mov_file_tell,
    };
    return &s_io;
}

static VencCodec to_venc_codec(VencRecorder::Codec c) {
    return (c == VencRecorder::Codec::H265) ? VencCodec::H265 : VencCodec::H264;
}

static int to_mov_object(VencRecorder::Codec c) {
    return (c == VencRecorder::Codec::H265) ? MOV_OBJECT_HEVC : MOV_OBJECT_H264;
}

static bool venc_can_reuse_current(const VencManager& venc, VencCodec codec, int in_width, int in_height,
                                   int fps, VencRcMode rc_mode) {
    if (!venc.isInitialized())
        return true;

    if (venc.getCodec() != codec)
        return false;
    if (venc.getRcMode() != rc_mode)
        return false;
    if (venc.getFps() != fps)
        return false;

    // encodeToVideo(video) requires a YUV420SP encoder session in the current backend. / encodeToVideo(video) 需要当前后端已经存在 YUV420SP 编码器会话。
    if (venc.getFormat() != RK_FMT_YUV420SP)
        return false;

    // The encoder may align width and height internally, so allow sessions that are at least large enough. / 编码器内部可能会对宽高做对齐，因此允许使用尺寸至少足够大的会话。
    const int w_aligned = (in_width + 15) & ~15;
    const int h_aligned = (in_height + 1) & ~1;
    if (venc.getWidth() < w_aligned || venc.getHeight() < h_aligned)
        return false;

    return true;
}

} // namespace

struct VencRecorder::Impl {
    std::string filepath;
    Codec codec = Codec::H264;
    Container container = Container::MP4;
    int quality = 75;
    VencRcMode rc_mode = VencRcMode::CBR;
    int fps = 30;
    bool mp4_faststart = true;

    FILE* fp = nullptr;
    mp4_writer_t* mp4 = nullptr;
    int track = -1;
    bool started = false;
    int width = 0;
    int height = 0;

    int64_t frame_index = 0;

    mpeg4_avc_t avc;
    mpeg4_hevc_t hevc;

    std::vector<uint8_t> mp4_frame_buf;
    std::vector<uint8_t> extra_buf;

    bool open = false;
    std::mutex mtx;

    Impl() {
        memset(&avc, 0, sizeof(avc));
        memset(&hevc, 0, sizeof(hevc));
    }

    void ensure_open() {
        if (open)
            return;
        const char* mode = (container == Container::MP4) ? "wb+" : "wb";
        fp = fopen(filepath.c_str(), mode);
        if (!fp)
            throw std::runtime_error("VencRecorder: Failed to open file: " + filepath);

        if (container == Container::MP4) {
            int flags = mp4_faststart ? MOV_FLAG_FASTSTART : 0;
            mp4 = mp4_writer_create(0 /*is_fmp4*/, mov_stdio_buffer(), fp, flags);
            if (!mp4) {
                fclose(fp);
                fp = nullptr;
                throw std::runtime_error("VencRecorder: mp4_writer_create failed.");
            }
        }

        open = true;
    }

    void close_nolock() {
        if (!open)
            return;
        if (mp4) {
            mp4_writer_destroy(mp4);
            mp4 = nullptr;
        }
        if (fp) {
            fclose(fp);
            fp = nullptr;
        }
        track = -1;
        started = false;
        width = height = 0;
        frame_index = 0;
        open = false;
        memset(&avc, 0, sizeof(avc));
        memset(&hevc, 0, sizeof(hevc));
        mp4_frame_buf.clear();
        extra_buf.clear();
    }

    void write_mp4_frame(const ImageBuffer& img) {
        VencManager& venc = VencManager::getInstance();
        const VencCodec venc_codec = to_venc_codec(codec);

        if (!venc_can_reuse_current(venc, venc_codec, img.width, img.height, fps, rc_mode)) {
            std::stringstream err;
            err << "VencRecorder: VENC is busy with " << venc.getWidth() << "x" << venc.getHeight() << " ["
                << PixelFormatToString(venc.getFormat()) << "] "
                << (venc.getCodec() == VencCodec::H264
                        ? "H264"
                        : (venc.getCodec() == VencCodec::H265 ? "H265" : "JPEG"))
                << ", fps=" << venc.getFps()
                << ", rc=" << (venc.getRcMode() == VencRcMode::CBR ? "CBR" : "VBR") << ".";
            throw std::runtime_error(err.str());
        }

        VencEncodedPacket packet;
        if (!venc.encodeToVideo(img, venc_codec, quality, packet, fps, rc_mode) || packet.data.empty()) {
            throw std::runtime_error("VencRecorder: VENC encoding failed (empty packet).");
        }
        // Use the active VENC output size as the encoded resolution. / 使用当前激活的 VENC 输出尺寸作为编码分辨率。
        const int enc_w = venc.getWidth();
        const int enc_h = venc.getHeight();
        if (width == 0 && height == 0) {
            width = enc_w;
            height = enc_h;
        } else if (width != enc_w || height != enc_h) {
            std::stringstream err;
            err << "VencRecorder: Resolution changed during recording: " << width << "x" << height << " -> "
                << enc_w << "x" << enc_h;
            throw std::runtime_error(err.str());
        }

        int vcl = 0;
        int update = 0;
        const size_t cap = packet.data.size() + 64 * 1024; // Reserve enough space for AVCC/HVCC conversion output.
        if (mp4_frame_buf.size() < cap)
            mp4_frame_buf.resize(cap);

        int n = 0;
        if (codec == Codec::H264) {
            n = h264_annexbtomp4(&avc, packet.data.data(), packet.data.size(), mp4_frame_buf.data(),
                                 mp4_frame_buf.size(), &vcl, &update);
        } else {
            n = h265_annexbtomp4(&hevc, packet.data.data(), packet.data.size(), mp4_frame_buf.data(),
                                 mp4_frame_buf.size(), &vcl, &update);
        }
        if (n <= 0) {
            throw std::runtime_error("VencRecorder: annexb->mp4 conversion failed.");
        }

        if (track < 0) {
            // Wait until SPS/PPS or VPS metadata is available before creating the MP4 track. / 在创建 MP4 track 之前，先等待 SPS/PPS 或 VPS 元数据就绪。
            bool ready = false;
            if (codec == Codec::H264) {
                ready = (avc.nb_sps >= 1 && avc.nb_pps >= 1);
            } else {
                ready = (hevc.numOfArrays >= 1);
            }
            if (!ready) {
                return; // drop until we have codec config
            }

            if (extra_buf.size() < 64 * 1024)
                extra_buf.resize(64 * 1024);
            int extra_size = 0;
            if (codec == Codec::H264) {
                extra_size =
                    mpeg4_avc_decoder_configuration_record_save(&avc, extra_buf.data(), extra_buf.size());
            } else {
                extra_size =
                    mpeg4_hevc_decoder_configuration_record_save(&hevc, extra_buf.data(), extra_buf.size());
            }
            if (extra_size <= 0) {
                throw std::runtime_error("VencRecorder: failed to build MP4 decoder configuration record.");
            }

            track = mp4_writer_add_video(mp4, (uint8_t)to_mov_object(codec), width, height, extra_buf.data(),
                                         (size_t)extra_size);
            if (track < 0) {
                throw std::runtime_error("VencRecorder: mp4_writer_add_video failed.");
            }
        }
        // Start MP4 output from the first VCL frame. / 从第一帧 VCL 开始输出 MP4。
        if (!started) {
            if (vcl != 1)
                return;
            started = true;
        }

        const int64_t pts_ms = (frame_index * 1000) / fps;
        const int flags = (vcl == 1) ? MOV_AV_FLAG_KEYFREAME : 0;
        if (0 != mp4_writer_write(mp4, track, mp4_frame_buf.data(), (size_t)n, pts_ms, pts_ms, flags)) {
            throw std::runtime_error("VencRecorder: mp4_writer_write failed.");
        }
        frame_index++;
    }

    void write_annexb_frame(const ImageBuffer& img) {
        VencManager& venc = VencManager::getInstance();
        const VencCodec venc_codec = to_venc_codec(codec);

        if (venc.isInitialized()) {
            // Annex-B output must not reuse a conflicting hardware encoder session. / Annex-B 输出不能复用存在冲突的硬件编码器会话。
            if (venc.getCodec() != venc_codec || venc.getWidth() != img.width ||
                venc.getHeight() != img.height || venc.getFormat() != img.format) {
                std::stringstream err;
                err << "VencRecorder: VENC Conflict: Hardware is busy with " << venc.getWidth() << "x"
                    << venc.getHeight() << " [" << PixelFormatToString(venc.getFormat()) << "] "
                    << (venc.getCodec() == VencCodec::H264
                            ? "H264"
                            : (venc.getCodec() == VencCodec::H265 ? "H265" : "JPEG"))
                    << ".";
                throw std::runtime_error(err.str());
            }
        }

        VencEncodedPacket packet;
        if (!venc.encodeToVideo(img, venc_codec, quality, packet, fps, rc_mode) || packet.data.empty()) {
            throw std::runtime_error("VencRecorder: VENC encoding failed (empty packet).");
        }
        // Prepend SPS/PPS/VPS headers to the first Annex-B frame when available. / 如果可用，则把 SPS/PPS/VPS 头预置到第一帧 Annex-B 数据前。
        if (frame_index == 0 && !packet.codec_data.empty()) {
            fwrite(packet.codec_data.data(), 1, packet.codec_data.size(), fp);
        }
        fwrite(packet.data.data(), 1, packet.data.size(), fp);
        frame_index++;
    }

    void write_frame(const ImageBuffer& img) {
        ensure_open();
        if (container == Container::MP4) {
            write_mp4_frame(img);
        } else {
            write_annexb_frame(img);
        }
    }
};

VencRecorder::VencRecorder(const std::string& filepath, Codec codec, Container container, int quality,
                           const std::string& rc_mode, int fps, bool mp4_faststart)
    : m_impl(std::make_unique<Impl>()) {
    m_impl->filepath = filepath;
    m_impl->codec = codec;
    m_impl->container = container;
    m_impl->quality = visiong::venc::clamp_quality(quality);
    m_impl->fps = visiong::venc::clamp_record_fps(fps);
    m_impl->rc_mode =
        (visiong::venc::normalize_rc_mode(rc_mode) == "vbr") ? VencRcMode::VBR : VencRcMode::CBR;
    m_impl->mp4_faststart = mp4_faststart;
    // Construction only stores validated parameters; the output file is opened lazily on first write. / 构造阶段仅保存已校验参数；输出文件会在第一次写入时按需打开。
}

VencRecorder::~VencRecorder() {
    try {
        close();
    } catch (...) {
        // destructor must not throw / 析构函数不得抛异常。
    }
}

void VencRecorder::write(const ImageBuffer& img) {
    if (!m_impl)
        throw std::runtime_error("VencRecorder: invalid instance");
    VencManager::ScopedUser user(VencManager::getInstance());
    std::lock_guard<std::mutex> lock(m_impl->mtx);
    m_impl->write_frame(img);
}

void VencRecorder::close() {
    if (!m_impl)
        return;
    std::lock_guard<std::mutex> lock(m_impl->mtx);
    m_impl->close_nolock();
}

bool VencRecorder::is_open() const {
    return m_impl && m_impl->open;
}

std::string VencRecorder::path() const {
    return m_impl ? m_impl->filepath : std::string();
}

// -----------------------------------------------------------------------------
// Cached recorders for ImageBuffer.save_venc_h26x(container="mp4") / 为 ImageBuffer.save_venc_h26x(container="mp4") 缓存的录制器。
// -----------------------------------------------------------------------------

namespace {
std::mutex g_cache_mutex;
std::unordered_map<std::string, std::shared_ptr<VencRecorder>> g_mp4_cache;
} // namespace

void save_venc_mp4_frame(const std::string& filepath, VencRecorder::Codec codec, const ImageBuffer& img,
                         int quality, const std::string& rc_mode, int fps, bool mp4_faststart, bool append) {
    std::shared_ptr<VencRecorder> recorder_to_use;
    std::shared_ptr<VencRecorder> recorder_to_close;

    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);

        if (!append) {
            auto it = g_mp4_cache.find(filepath);
            if (it != g_mp4_cache.end()) {
                recorder_to_close = it->second;
                g_mp4_cache.erase(it);
            }
        }

        auto it = g_mp4_cache.find(filepath);
        if (it != g_mp4_cache.end()) {
            recorder_to_use = it->second;
        } else {
            auto rec = std::make_shared<VencRecorder>(filepath, codec, VencRecorder::Container::MP4,
                                                      quality, rc_mode, fps, mp4_faststart);
            recorder_to_use = rec;
            g_mp4_cache.emplace(filepath, rec);
        }
    }

    if (recorder_to_close) {
        recorder_to_close->close();
    }
    if (recorder_to_use) {
        recorder_to_use->write(img);
    }
}

void close_venc_recorder(const std::string& filepath) {
    std::shared_ptr<VencRecorder> recorder;
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        auto it = g_mp4_cache.find(filepath);
        if (it != g_mp4_cache.end()) {
            recorder = it->second;
            g_mp4_cache.erase(it);
        }
    }
    if (recorder) {
        recorder->close();
    }
}

void close_all_venc_recorders() {
    std::vector<std::shared_ptr<VencRecorder>> recorders;
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        recorders.reserve(g_mp4_cache.size());
        for (auto& kv : g_mp4_cache) {
            if (kv.second) {
                recorders.push_back(kv.second);
            }
        }
        g_mp4_cache.clear();
    }
    for (auto& recorder : recorders) {
        recorder->close();
    }
}

