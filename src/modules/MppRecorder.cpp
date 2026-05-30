// SPDX-License-Identifier: LGPL-3.0-or-later
#include "visiong/modules/MppRecorder.h"

#include "visiong/core/ImageBuffer.h"
#include "core/internal/rga_utils.h"
#include "visiong/modules/MppEncoderManager.h"
#include "modules/internal/mpp_utils.h"

#include <cerrno>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

// media-server components.
#include "mov-format.h"
#include "mp4-writer.h"
#include "mpeg4-avc.h"
#include "mpeg4-hevc.h"

namespace {

enum class RecorderTimestampMode {
    FixedFps,
    WallClock,
};

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
    // Non-negative offsets are absolute; negative offsets are relative to EOF.
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

static MppCodec to_mpp_codec(MppRecorder::Codec c) {
    return (c == MppRecorder::Codec::H265) ? MppCodec::H265 : MppCodec::H264;
}

static int to_mov_object(MppRecorder::Codec c) {
    return (c == MppRecorder::Codec::H265) ? MOV_OBJECT_HEVC : MOV_OBJECT_H264;
}

static std::string lower_ext(const std::string& filepath) {
    const size_t dot = filepath.rfind('.');
    if (dot == std::string::npos) {
        return {};
    }
    std::string ext = filepath.substr(dot + 1);
    for (char& ch : ext) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return ext;
}

static MppRecorder::Codec resolve_user_video_codec(const std::string& filepath, const std::string& codec) {
    std::string normalized;
    normalized.reserve(codec.size());
    for (unsigned char ch : codec) {
        normalized.push_back(static_cast<char>(std::tolower(ch)));
    }
    if (normalized.empty() || normalized == "auto") {
        const std::string ext = lower_ext(filepath);
        return (ext == "h265" || ext == "hevc") ? MppRecorder::Codec::H265 : MppRecorder::Codec::H264;
    }
    if (normalized == "h264" || normalized == "avc") {
        return MppRecorder::Codec::H264;
    }
    if (normalized == "h265" || normalized == "hevc") {
        return MppRecorder::Codec::H265;
    }
    throw std::invalid_argument("VideoRecorder: codec must be 'auto', 'h264', or 'h265'.");
}

static MppRecorder::Container resolve_user_video_container(const std::string& filepath) {
    return lower_ext(filepath) == "mp4" ? MppRecorder::Container::MP4 : MppRecorder::Container::ANNEXB;
}

static RecorderTimestampMode parse_timestamp_mode(const std::string& mode) {
    std::string normalized;
    normalized.reserve(mode.size());
    for (unsigned char ch : mode) {
        if (ch == '-' || ch == '_') {
            continue;
        }
        normalized.push_back(static_cast<char>(std::tolower(ch)));
    }

    if (normalized.empty() || normalized == "fixed" || normalized == "fixedfps" || normalized == "cfr") {
        return RecorderTimestampMode::FixedFps;
    }
    if (normalized == "wallclock" || normalized == "time" || normalized == "realtime" || normalized == "vfr") {
        return RecorderTimestampMode::WallClock;
    }
    throw std::invalid_argument("MppRecorder: timestamp_mode must be 'fixed' or 'wallclock'.");
}

static std::string describe_errno(int error_code) {
    if (error_code > 0) {
        return std::strerror(error_code);
    }
    return "unknown error";
}

static void fwrite_or_throw(FILE* fp, const void* data, size_t bytes, const std::string& filepath) {
    if (bytes == 0) {
        return;
    }
    if (fwrite(data, 1, bytes, fp) != bytes) {
        const int error_code = errno;
        throw std::runtime_error("MppRecorder: failed to write file: " + filepath + ": " +
                                 describe_errno(error_code));
    }
}

static bool mpp_can_reuse_current(const MppEncoderManager& mpp, int channel_id, MppCodec codec, int in_width,
                                   int in_height, int fps, MppRcMode rc_mode) {
    if (!mpp.isInitialized(channel_id))
        return true;

    if (mpp.getCodec(channel_id) != codec)
        return false;
    if (mpp.getRcMode(channel_id) != rc_mode)
        return false;
    if (mpp.getFps(channel_id) != fps)
        return false;

    if (mpp.getFormat(channel_id) != RK_FMT_YUV420SP)
        return false;

    // The backend may align width and height internally, so allow larger sessions.
    const int w_aligned = (in_width + 15) & ~15;
    const int h_aligned = (in_height + 1) & ~1;
    if (mpp.getWidth(channel_id) < w_aligned || mpp.getHeight(channel_id) < h_aligned)
        return false;

    return true;
}

} // namespace

struct MppRecorder::Impl {
    std::string filepath;
    Codec codec = Codec::H264;
    Container container = Container::MP4;
    int quality = 75;
    MppRcMode rc_mode = MppRcMode::CBR;
    int fps = 30;
    bool mp4_faststart = true;
    RecorderTimestampMode timestamp_mode = RecorderTimestampMode::FixedFps;
    int requested_mpp_channel = -1;
    int mpp_channel = -1;
    bool mpp_channel_acquired = false;

    FILE* fp = nullptr;
    mp4_writer_t* mp4 = nullptr;
    int track = -1;
    bool started = false;
    int width = 0;
    int height = 0;

    int64_t frame_index = 0;
    bool wallclock_started = false;
    std::chrono::steady_clock::time_point wallclock_start;
    int64_t last_pts_ms = -1;

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

    bool reserve_mpp_channel() {
        if (mpp_channel_acquired) {
            return true;
        }
        auto& mpp = MppEncoderManager::getInstance();
        mpp_channel = mpp.acquireDedicatedChannel(requested_mpp_channel);
        if (mpp_channel < 0) {
            return false;
        }
        mpp_channel_acquired = true;
        return true;
    }

    void release_mpp_channel() {
        if (mpp_channel_acquired && mpp_channel >= 0) {
            MppEncoderManager::getInstance().releaseDedicatedChannel(mpp_channel);
        }
        mpp_channel = -1;
        mpp_channel_acquired = false;
    }

    int64_t next_pts_ms(std::chrono::steady_clock::time_point frame_time) {
        int64_t pts_ms = 0;
        if (timestamp_mode == RecorderTimestampMode::WallClock) {
            if (!wallclock_started) {
                wallclock_start = frame_time;
                wallclock_started = true;
                pts_ms = 0;
            } else {
                pts_ms = std::chrono::duration_cast<std::chrono::milliseconds>(frame_time - wallclock_start).count();
            }
            if (pts_ms <= last_pts_ms) {
                pts_ms = last_pts_ms + 1;
            }
        } else {
            pts_ms = (frame_index * 1000) / fps;
        }
        last_pts_ms = pts_ms;
        return pts_ms;
    }

    void ensure_open() {
        if (open)
            return;
        const char* mode = (container == Container::MP4) ? "wb+" : "wb";
        fp = fopen(filepath.c_str(), mode);
        if (!fp)
            throw std::runtime_error("MppRecorder: Failed to open file: " + filepath);

        if (container == Container::MP4) {
            int flags = mp4_faststart ? MOV_FLAG_FASTSTART : 0;
            mp4 = mp4_writer_create(0 /*is_fmp4*/, mov_stdio_buffer(), fp, flags);
            if (!mp4) {
                fclose(fp);
                fp = nullptr;
                throw std::runtime_error("MppRecorder: mp4_writer_create failed.");
            }
        }

        open = true;
    }

    void close_nolock() {
        if (!open) {
            release_mpp_channel();
            return;
        }
        int close_error = 0;
        if (mp4) {
            mp4_writer_destroy(mp4);
            mp4 = nullptr;
        }
        if (fp) {
            if (ferror(fp)) {
                close_error = EIO;
            }
            if (fclose(fp) != 0 && close_error == 0) {
                close_error = errno;
            }
            fp = nullptr;
        }
        track = -1;
        started = false;
        width = height = 0;
        frame_index = 0;
        wallclock_started = false;
        last_pts_ms = -1;
        open = false;
        memset(&avc, 0, sizeof(avc));
        memset(&hevc, 0, sizeof(hevc));
        mp4_frame_buf.clear();
        extra_buf.clear();
        release_mpp_channel();
        if (close_error != 0) {
            throw std::runtime_error("MppRecorder: failed to close file: " + filepath + ": " +
                                     describe_errno(close_error));
        }
    }

    void write_mp4_frame(const ImageBuffer& img) {
        MppEncoderManager& mpp = MppEncoderManager::getInstance();
        const MppCodec mpp_codec = to_mpp_codec(codec);
        const auto frame_time = std::chrono::steady_clock::now();

        if (!mpp_can_reuse_current(mpp, mpp_channel, mpp_codec, img.width, img.height, fps, rc_mode)) {
            std::stringstream err;
            err << "MppRecorder: MPP is busy with " << mpp.getWidth(mpp_channel) << "x" << mpp.getHeight(mpp_channel) << " ["
                << PixelFormatToString(mpp.getFormat(mpp_channel)) << "] "
                << (mpp.getCodec(mpp_channel) == MppCodec::H264
                        ? "H264"
                        : (mpp.getCodec(mpp_channel) == MppCodec::H265 ? "H265" : "JPEG"))
                << ", fps=" << mpp.getFps(mpp_channel)
                << ", rc=" << (mpp.getRcMode(mpp_channel) == MppRcMode::CBR ? "CBR" : "VBR") << ".";
            throw std::runtime_error(err.str());
        }

        MppEncodedPacket packet;
        if (!mpp.encodeToVideoOnChannel(mpp_channel, img, mpp_codec, quality, packet, fps, rc_mode) ||
            packet.data.empty()) {
            throw std::runtime_error("MppRecorder: MPP encoding failed (empty packet).");
        }
        // Use the active MPP output size as the encoded resolution.
        const int enc_w = mpp.getWidth(mpp_channel);
        const int enc_h = mpp.getHeight(mpp_channel);
        if (width == 0 && height == 0) {
            width = enc_w;
            height = enc_h;
        } else if (width != enc_w || height != enc_h) {
            std::stringstream err;
            err << "MppRecorder: Resolution changed during recording: " << width << "x" << height << " -> "
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
            throw std::runtime_error("MppRecorder: annexb->mp4 conversion failed.");
        }

        if (track < 0) {
            // Wait until SPS/PPS or VPS metadata is available before creating the MP4 track.
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
                throw std::runtime_error("MppRecorder: failed to build MP4 decoder configuration record.");
            }

            track = mp4_writer_add_video(mp4, (uint8_t)to_mov_object(codec), width, height, extra_buf.data(),
                                         (size_t)extra_size);
            if (track < 0) {
                throw std::runtime_error("MppRecorder: mp4_writer_add_video failed.");
            }
        }
        // Start MP4 output from the first VCL frame.
        if (!started) {
            if (vcl != 1)
                return;
            started = true;
        }

        const int64_t pts_ms = next_pts_ms(frame_time);
        const int flags = (vcl == 1) ? MOV_AV_FLAG_KEYFREAME : 0;
        if (0 != mp4_writer_write(mp4, track, mp4_frame_buf.data(), (size_t)n, pts_ms, pts_ms, flags)) {
            throw std::runtime_error("MppRecorder: mp4_writer_write failed.");
        }
        frame_index++;
    }

    void write_annexb_frame(const ImageBuffer& img) {
        MppEncoderManager& mpp = MppEncoderManager::getInstance();
        const MppCodec mpp_codec = to_mpp_codec(codec);

        if (mpp.isInitialized(mpp_channel)) {
            // Annex-B output must not reuse a conflicting hardware encoder session.
            if (mpp.getCodec(mpp_channel) != mpp_codec || mpp.getWidth(mpp_channel) != img.width ||
                mpp.getHeight(mpp_channel) != img.height || mpp.getFormat(mpp_channel) != img.format) {
                std::stringstream err;
                err << "MppRecorder: MPP Conflict: Hardware is busy with " << mpp.getWidth(mpp_channel) << "x"
                    << mpp.getHeight(mpp_channel) << " [" << PixelFormatToString(mpp.getFormat(mpp_channel)) << "] "
                    << (mpp.getCodec(mpp_channel) == MppCodec::H264
                            ? "H264"
                            : (mpp.getCodec(mpp_channel) == MppCodec::H265 ? "H265" : "JPEG"))
                    << ".";
                throw std::runtime_error(err.str());
            }
        }

        MppEncodedPacket packet;
        if (!mpp.encodeToVideoOnChannel(mpp_channel, img, mpp_codec, quality, packet, fps, rc_mode) ||
            packet.data.empty()) {
            throw std::runtime_error("MppRecorder: MPP encoding failed (empty packet).");
        }
        // Prepend SPS/PPS/VPS headers to the first Annex-B frame when available.
        if (frame_index == 0 && !packet.codec_data.empty()) {
            fwrite_or_throw(fp, packet.codec_data.data(), packet.codec_data.size(), filepath);
        }
        fwrite_or_throw(fp, packet.data.data(), packet.data.size(), filepath);
        frame_index++;
    }

    void write_frame(const ImageBuffer& img) {
        if (!reserve_mpp_channel()) {
            throw std::runtime_error("MppRecorder: no free dedicated MPP channel is available.");
        }
        ensure_open();
        if (container == Container::MP4) {
            write_mp4_frame(img);
        } else {
            write_annexb_frame(img);
        }
    }
};

MppRecorder::MppRecorder(const std::string& filepath, Codec codec, Container container, int quality,
                           const std::string& rc_mode, int fps, bool mp4_faststart, int mpp_channel,
                           const std::string& timestamp_mode)
    : m_impl(std::make_unique<Impl>()) {
    m_impl->filepath = filepath;
    m_impl->codec = codec;
    m_impl->container = container;
    m_impl->quality = visiong::mpp::clamp_quality(quality);
    m_impl->fps = visiong::mpp::clamp_record_fps(fps);
    m_impl->rc_mode =
        (visiong::mpp::normalize_rc_mode(rc_mode) == "vbr") ? MppRcMode::VBR : MppRcMode::CBR;
    m_impl->mp4_faststart = mp4_faststart;
    m_impl->requested_mpp_channel = mpp_channel;
    m_impl->timestamp_mode = parse_timestamp_mode(timestamp_mode);
    if (!m_impl->reserve_mpp_channel()) {
        throw std::runtime_error("MppRecorder: no free dedicated MPP channel is available.");
    }
    // Construction only stores validated parameters; the output file is opened lazily on first write.
}

MppRecorder::~MppRecorder() {
    try {
        close();
    } catch (...) {
        // Destructors must not throw.
    }
}

void MppRecorder::write(const ImageBuffer& img) {
    if (!m_impl)
        throw std::runtime_error("MppRecorder: invalid instance");
    std::lock_guard<std::mutex> lock(m_impl->mtx);
    m_impl->write_frame(img);
}

void MppRecorder::close() {
    if (!m_impl)
        return;
    std::lock_guard<std::mutex> lock(m_impl->mtx);
    m_impl->close_nolock();
}

bool MppRecorder::is_open() const {
    return m_impl && m_impl->open;
}

std::string MppRecorder::path() const {
    return m_impl ? m_impl->filepath : std::string();
}

int MppRecorder::get_mpp_channel() const {
    return m_impl ? m_impl->mpp_channel : -1;
}

VideoRecorder::VideoRecorder(const std::string& filepath, int quality, int fps, const std::string& codec)
    : m_path(filepath) {
    m_recorder = std::make_unique<MppRecorder>(
        filepath,
        resolve_user_video_codec(filepath, codec),
        resolve_user_video_container(filepath),
        quality,
        "cbr",
        fps,
        true);
}

VideoRecorder::~VideoRecorder() {
    try {
        close();
    } catch (...) {
    }
}

void VideoRecorder::write(const ImageBuffer& img) {
    if (!m_recorder) {
        throw std::runtime_error("VideoRecorder: recorder is closed.");
    }
    m_recorder->write(img);
}

void VideoRecorder::close() {
    if (m_recorder) {
        m_recorder->close();
        m_recorder.reset();
    }
}

bool VideoRecorder::is_open() const {
    return m_recorder != nullptr;
}

bool VideoRecorder::is_started() const {
    return m_recorder && m_recorder->is_open();
}

std::string VideoRecorder::path() const {
    return m_path;
}

// -----------------------------------------------------------------------------
// Cached recorders for ImageBuffer.save_video() and advanced save_mpp_* MP4 paths.
// -----------------------------------------------------------------------------

namespace {
std::mutex g_cache_mutex;
std::unordered_map<std::string, std::shared_ptr<MppRecorder>> g_mp4_cache;
} // namespace

void save_mpp_mp4_frame(const std::string& filepath, MppRecorder::Codec codec, const ImageBuffer& img,
                         int quality, const std::string& rc_mode, int fps, bool mp4_faststart, bool append) {
    std::shared_ptr<MppRecorder> recorder_to_use;
    std::shared_ptr<MppRecorder> recorder_to_close;

    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);

        if (!append) {
            auto it = g_mp4_cache.find(filepath);
            if (it != g_mp4_cache.end()) {
                recorder_to_close = it->second;
                g_mp4_cache.erase(it);
            }
        }

    }

    if (recorder_to_close) {
        recorder_to_close->close();
    }
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        auto it = g_mp4_cache.find(filepath);
        if (it != g_mp4_cache.end()) {
            recorder_to_use = it->second;
        } else {
            auto rec = std::make_shared<MppRecorder>(filepath, codec, MppRecorder::Container::MP4,
                                                      quality, rc_mode, fps, mp4_faststart);
            recorder_to_use = rec;
            g_mp4_cache.emplace(filepath, rec);
        }
    }

    if (recorder_to_use) {
        recorder_to_use->write(img);
    }
}

void close_mpp_recorder(const std::string& filepath) {
    std::shared_ptr<MppRecorder> recorder;
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

void close_all_mpp_recorders() {
    std::vector<std::shared_ptr<MppRecorder>> recorders;
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

void close_video(const std::string& filepath) {
    close_mpp_recorder(filepath);
}

void close_all_videos() {
    close_all_mpp_recorders();
}
