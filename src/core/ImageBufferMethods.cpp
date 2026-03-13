// SPDX-License-Identifier: LGPL-3.0-or-later
#include "stb_image.h"
#include <iomanip>
#include <ctime>
#include "visiong/core/ImageBuffer.h"
#include "visiong/core/BufferStateMachine.h"
#include "visiong/core/RgaHelper.h"
#include "core/internal/rga_utils.h"
#include "im2d.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <vector>
#include <cmath>
#include <numeric>
#include <map>
#include "quirc.h"
#include <chrono>
#include <random>
#include "visiong/modules/VencManager.h"
#include "visiong/modules/VencRecorder.h"
#include <fstream>
#include "stb_image_write.h"
#include "visiong/common/build_config.h"
#if VISIONG_WITH_IVE
#include "visiong/modules/IVE.h"
#endif
#include "visiong/common/pixel_format.h"
#include "common/internal/string_utils.h"
#include "modules/internal/venc_utils.h"
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#ifndef CV_PI
#define CV_PI 3.1415926535897932384626433832795
#endif

// ============================================================================
ImageBuffer ImageBuffer::load(const std::string& filepath) {
    int w, h, channels;
    unsigned char* data = stbi_load(filepath.c_str(), &w, &h, &channels, 0);

    if (!data) {
        throw std::runtime_error("Failed to load image: " + std::string(stbi_failure_reason()));
    }

    PIXEL_FORMAT_E format;
    switch (channels) {
        case 1: format = visiong::kGray8Format; break;
        case 3: format = RK_FMT_RGB888; break;
        case 4: format = RK_FMT_RGBA8888; break;
        default:
            stbi_image_free(data);
            throw std::runtime_error("Unsupported number of channels: " + std::to_string(channels));
    }

    size_t data_size = static_cast<size_t>(w) * h * channels;
    std::vector<unsigned char> img_data(data, data + data_size);
    stbi_image_free(data);

    ImageBuffer temp_raw_img(w, h, format, std::move(img_data));

    // Align to hardware-friendly dimensions for downstream RGA paths.
    // 将尺寸对齐到更适合下游 RGA 路径的硬件边界。
    const int aligned_w = (w + 15) & ~15;
    const int aligned_h = (h + 1) & ~1;

    if (w == aligned_w && h == aligned_h) {
        return temp_raw_img;
    }

    try {
        return temp_raw_img.resize(aligned_w, aligned_h);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to align loaded image with RGA: " + std::string(e.what()));
    }
}

void ImageBuffer::save_hsv_bin(const std::string& filepath) const {
    if (!is_valid()) throw std::runtime_error("save_hsv_bin: Invalid ImageBuffer");
    if (filepath.empty()) throw std::runtime_error("save_hsv_bin: filepath must not be empty.");

    ImageBuffer hsv_buf;
    bool used_ive_path = false;

#if VISIONG_WITH_IVE
    if (format == RK_FMT_YUV420SP || format == RK_FMT_YUV420SP_VU) {
        auto& ive = IVE::get_instance();
        hsv_buf = ive.yuv_to_hsv(*this, true);
        used_ive_path = true;
    }
#endif

    if (!hsv_buf.is_valid()) {
        // Fall back to OpenCV conversion when IVE is disabled or the source is not YUV.
        // 当 IVE 被关闭或输入并非 YUV 时，回退到 OpenCV 颜色空间转换。
        const ImageBuffer& bgr = this->get_bgr_version();
        visiong::bufstate::prepare_cpu_read(bgr);

        cv::Mat bgr_mat(
            bgr.height,
            bgr.width,
            CV_8UC3,
            const_cast<void*>(bgr.get_data()),
            static_cast<size_t>(bgr.w_stride) * 3);
        cv::Mat hsv_mat;
        cv::cvtColor(bgr_mat, hsv_mat, cv::COLOR_BGR2HSV_FULL);

        const size_t hsv_size = hsv_mat.total() * hsv_mat.elemSize();
        std::vector<unsigned char> hsv_data(hsv_mat.data, hsv_mat.data + hsv_size);
        hsv_buf = ImageBuffer(hsv_mat.cols, hsv_mat.rows, RK_FMT_RGB888, std::move(hsv_data));
    }

    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs) throw std::runtime_error("save_hsv_bin: Failed to create " + filepath);
    visiong::bufstate::prepare_cpu_read(hsv_buf);
    ofs.write(reinterpret_cast<const char*>(hsv_buf.get_data()), hsv_buf.get_size());
    ofs.close();

    std::cout << "[visiong] HSV raw data saved to: " << filepath
              << (used_ive_path ? " (IVE path)" : " (OpenCV fallback)") << std::endl;
}

void ImageBuffer::save_venc_jpg(const std::string& filepath, int quality) const {
    if (!is_valid()) throw std::runtime_error("save_venc_jpg: Invalid ImageBuffer");
    const int normalized_quality = visiong::venc::clamp_quality(quality);

    auto& venc = VencManager::getInstance();
    VencManager::ScopedUser user(venc);

    if (venc.isInitialized()) {
        if (venc.getWidth() != this->width ||
            venc.getHeight() != this->height || 
            venc.getFormat() != this->format) {
            
            std::stringstream err;
            err << "VENC Conflict: Hardware is busy with " 
                << venc.getWidth() << "x" << venc.getHeight() << " [" << PixelFormatToString(venc.getFormat()) << "]. "
                << "Cannot save image with " << this->width << "x" << this->height << " [" << PixelFormatToString(this->format) << "].";
            throw std::runtime_error(err.str());
        }
    } 
    std::vector<unsigned char> jpg_data = venc.encodeToJpeg(*this, normalized_quality);

    if (jpg_data.empty()) {
        throw std::runtime_error("save_venc_jpg: VENC hardware encoding failed (returned empty data).");
    }

    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs) throw std::runtime_error("save_venc_jpg: Failed to open file for writing: " + filepath);
    ofs.write(reinterpret_cast<const char*>(jpg_data.data()), jpg_data.size());
    ofs.close();
}

namespace {
std::string get_ext_lower(const std::string& filepath) {
    size_t dot = filepath.rfind('.');
    if (dot == std::string::npos) return {};
    return visiong::to_lower_copy(filepath.substr(dot + 1));
}

enum class SaveContainer {
    ANNEXB = 0,
    MP4 = 1
};

SaveContainer resolve_container(const std::string& container, const std::string& filepath) {
    const std::string c = visiong::to_lower_copy(container);
    if (c.empty() || c == "auto") {
        return (get_ext_lower(filepath) == "mp4") ? SaveContainer::MP4 : SaveContainer::ANNEXB;
    }
    if (c == "mp4") return SaveContainer::MP4;
    if (c == "annexb" || c == "raw" || c == "h264" || c == "h265") return SaveContainer::ANNEXB;
    throw std::invalid_argument("save_venc_h26x: container must be 'auto', 'annexb', or 'mp4'.");
}

VencRecorder::Codec to_recorder_codec(VencCodec codec) {
    return (codec == VencCodec::H265) ? VencRecorder::Codec::H265 : VencRecorder::Codec::H264;
}

bool is_file_empty(const std::string& filepath) {
    std::ifstream ifs(filepath, std::ios::binary | std::ios::ate);
    if (!ifs) return true;
    std::streampos size = ifs.tellg();
    return size <= 0;
}

void save_venc_video_impl(const ImageBuffer& img, const std::string& filepath, VencCodec codec,
                          int quality, const std::string& rc_mode, int fps, bool append,
                          const std::string& container, bool mp4_faststart) {
    if (!img.is_valid()) throw std::runtime_error("save_venc_h26x: Invalid ImageBuffer");
    if (filepath.empty()) throw std::runtime_error("save_venc_h26x: filepath must not be empty.");
    const int normalized_quality = visiong::venc::clamp_quality(quality);
    const int normalized_fps = visiong::venc::clamp_record_fps(fps);
    const std::string normalized_rc_mode = visiong::venc::normalize_rc_mode(rc_mode);
    const VencRcMode rc = (normalized_rc_mode == "vbr") ? VencRcMode::VBR : VencRcMode::CBR;

    const SaveContainer out_container = resolve_container(container, filepath);
    if (out_container == SaveContainer::MP4) {
        save_venc_mp4_frame(filepath, to_recorder_codec(codec), img, normalized_quality,
                            normalized_rc_mode, normalized_fps, mp4_faststart, append);
        return;
    }

    auto& venc = VencManager::getInstance();
    VencManager::ScopedUser user(venc);

    if (venc.isInitialized()) {
        const bool shape_or_codec_conflict =
            (venc.getWidth() != img.width || venc.getHeight() != img.height || venc.getFormat() != img.format ||
             venc.getCodec() != codec);
        const bool encoder_param_conflict =
            (venc.getQuality() != normalized_quality || venc.getFps() != normalized_fps || venc.getRcMode() != rc);
        if (shape_or_codec_conflict || encoder_param_conflict) {
            std::stringstream err;
            err << "VENC Conflict: Hardware is busy with "
                << venc.getWidth() << "x" << venc.getHeight() << " ["
                << PixelFormatToString(venc.getFormat()) << "] "
                << (venc.getCodec() == VencCodec::H264 ? "H264" : (venc.getCodec() == VencCodec::H265 ? "H265" : "JPEG"))
                << " q=" << venc.getQuality()
                << " fps=" << venc.getFps()
                << " rc=" << (venc.getRcMode() == VencRcMode::VBR ? "VBR" : "CBR")
                << ". Cannot save video with "
                << img.width << "x" << img.height << " ["
                << PixelFormatToString(img.format) << "] "
                << (codec == VencCodec::H264 ? "H264" : "H265")
                << " q=" << normalized_quality
                << " fps=" << normalized_fps
                << " rc=" << (rc == VencRcMode::VBR ? "VBR" : "CBR")
                << ".";
            throw std::runtime_error(err.str());
        }
    }

    VencEncodedPacket packet;
    if (!venc.encodeToVideo(img, codec, normalized_quality, packet, normalized_fps, rc)) {
        throw std::runtime_error("save_venc_h26x: VENC hardware encoding failed.");
    }
    if (packet.data.empty()) {
        throw std::runtime_error("save_venc_h26x: VENC returned empty data.");
    }

    const bool need_prefix = (!append) || is_file_empty(filepath);
    std::ios::openmode mode = std::ios::binary | (append ? std::ios::app : std::ios::trunc);
    std::ofstream ofs(filepath, mode);
    if (!ofs) throw std::runtime_error("save_venc_h26x: Failed to open file for writing: " + filepath);

    if (need_prefix && !packet.codec_data.empty()) {
        ofs.write(reinterpret_cast<const char*>(packet.codec_data.data()), packet.codec_data.size());
    }
    ofs.write(reinterpret_cast<const char*>(packet.data.data()), packet.data.size());
    ofs.close();
}
} // namespace

void ImageBuffer::save_venc_h264(const std::string& filepath, int quality, const std::string& rc_mode,
                                 int fps, bool append, const std::string& container, bool mp4_faststart) const {
    save_venc_video_impl(*this, filepath, VencCodec::H264, quality, rc_mode, fps, append, container, mp4_faststart);
}

void ImageBuffer::save_venc_h265(const std::string& filepath, int quality, const std::string& rc_mode,
                                 int fps, bool append, const std::string& container, bool mp4_faststart) const {
    save_venc_video_impl(*this, filepath, VencCodec::H265, quality, rc_mode, fps, append, container, mp4_faststart);
}

void ImageBuffer::save(const std::string& filepath, int quality) const {
    if (!is_valid()) {
        throw std::runtime_error("save: Cannot save an invalid ImageBuffer.");
    }
    if (quality < 1 || quality > 100) {
        throw std::invalid_argument("save: JPEG/PNG quality must be between 1 and 100.");
    }

    std::string ext;
    size_t dot_pos = filepath.rfind('.');
    if (dot_pos != std::string::npos) {
        ext = filepath.substr(dot_pos + 1);
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c){ return std::tolower(c); });
    }

    if (ext == "jpg" || ext == "jpeg" || ext == "bmp") {
        const ImageBuffer& bgr_img = this->get_bgr_version();
        if (!bgr_img.is_valid()) {
            throw std::runtime_error("save: Failed to get a BGR version of the image for saving.");
        }

        std::vector<unsigned char> rgb_data;
        convert_bgr_to_compact_rgb(bgr_img, rgb_data);

        if (ext == "jpg" || ext == "jpeg") {
            if (stbi_write_jpg(filepath.c_str(), bgr_img.width, bgr_img.height, 3, rgb_data.data(), quality) == 0) {
                throw std::runtime_error("save: Failed to write JPEG file.");
            }
        } else { // ext == "bmp"
            if (stbi_write_bmp(filepath.c_str(), bgr_img.width, bgr_img.height, 3, rgb_data.data()) == 0) {
                throw std::runtime_error("save: Failed to write BMP file.");
            }
        }

    } else if (ext == "png") {
        ImageBuffer rgba_img_owner;
        const ImageBuffer* rgba_img = this;

        if (this->format != RK_FMT_RGBA8888) {
            try {
                rgba_img_owner = this->to_format(RK_FMT_RGBA8888);
                rgba_img = &rgba_img_owner;
            } catch (const std::exception& e) {
                throw std::runtime_error("save: Failed to convert image to RGBA8888 for PNG: " + std::string(e.what()));
            }
        }
        
        if (!rgba_img->is_valid()) {
            throw std::runtime_error("save: Failed to get a valid RGBA image for saving as PNG.");
        }

        int stride_in_bytes = rgba_img->w_stride * 4;
        visiong::bufstate::prepare_cpu_read(*rgba_img);
        if (stbi_write_png(filepath.c_str(), rgba_img->width, rgba_img->height, 4, rgba_img->get_data(), stride_in_bytes) == 0) {
            throw std::runtime_error("save: Failed to write PNG file.");
        }

    } else {
        throw std::invalid_argument("save: Unsupported file extension '" + ext + "'. Supported formats are: jpg, jpeg, png, bmp.");
    }
}

// Copy DMA-backed image data into compact CPU-owned storage.
// 将 DMA 后端图像复制到紧凑的 CPU 自有存储中。
ImageBuffer& ImageBuffer::copy_from_dma(const RgaDmaBuffer& dma_buf) {
    width = dma_buf.get_width();
    height = dma_buf.get_height();
    format = static_cast<PIXEL_FORMAT_E>(dma_buf.get_mpi_format());
    w_stride = width;
    h_stride = height;

    m_is_zero_copy = false;
    m_mb_blk_handle_sptr = nullptr;
    m_dma_buf_sptr = nullptr;

    const int bpp = get_bpp_for_format(format);
    if (bpp == 0) {
        throw std::runtime_error("copy_from_dma: Unknown bpp for format.");
    }

    const size_t compact_width_bytes = static_cast<size_t>(width) * bpp / 8;
    m_size = compact_width_bytes * height;
    m_user_data.resize(m_size);

    visiong::bufstate::prepare_cpu_read(dma_buf);
    copy_data_from_stride(
        m_user_data.data(),
        dma_buf.get_vir_addr(),
        compact_width_bytes,
        height,
        dma_buf.get_wstride() * bpp / 8);

    return *this;
}


ImageBuffer ImageBuffer::from_cv_mat(const cv::Mat& mat, PIXEL_FORMAT_E format) {
    if (mat.empty()) {
        throw std::runtime_error("from_cv_mat: Input cv::Mat is empty.");
    }
    if (!((mat.type() == CV_8UC3 && (format == RK_FMT_BGR888 || format == RK_FMT_RGB888)) ||
          (mat.type() == CV_8UC4 && (format == RK_FMT_BGRA8888 || format == RK_FMT_RGBA8888)) ||
          (mat.type() == CV_8UC1 && format == visiong::kGray8Format))) {
        throw std::runtime_error("from_cv_mat: Mat type and format mismatch.");
    }
    return ImageBuffer(mat.cols, mat.rows, format, std::vector<unsigned char>(mat.data, mat.data + mat.total() * mat.elemSize()));
}

// Returns a non-owning BGR cv::Mat view over ImageBuffer storage.
// 返回一个不拥有数据所有权的 BGR cv::Mat 视图。
static cv::Mat image_buffer_to_bgr_mat_view(const ImageBuffer& img_buf) {
    if (!img_buf.is_valid()) return cv::Mat();
    const ImageBuffer& bgr_version = img_buf.get_bgr_version();
    visiong::bufstate::prepare_cpu_read(bgr_version);
    
    cv::Mat view(bgr_version.height, bgr_version.w_stride, CV_8UC3, const_cast<void*>(bgr_version.get_data()));
    
    if (bgr_version.w_stride != bgr_version.width) {
        return view(cv::Rect(0, 0, bgr_version.width, bgr_version.height));
    }
    return view;
}

// Cached grayscale view accessor.
// 灰度缓存视图访问器。
const ImageBuffer& ImageBuffer::get_bgr_version() const {
    if (this->format == RK_FMT_BGR888) {
        return *this;
    }
    if (!m_cached_bgr) {
        m_cached_bgr = std::make_unique<ImageBuffer>(this->to_format(RK_FMT_BGR888));
    }
    return *m_cached_bgr;
}

const ImageBuffer& ImageBuffer::get_gray_version() const {
    if (this->format == visiong::kGray8Format) {
        return *this;
    }
    if (!m_cached_gray) {
        m_cached_gray = std::make_unique<ImageBuffer>(this->to_grayscale());
    }
    return *m_cached_gray;
}


ImageBuffer ImageBuffer::rotate(int angle_degrees) const {
    if (!is_valid()) throw std::runtime_error("rotate: invalid source image");

    int transform_flag = 0;
    switch (angle_degrees) {
        case 90:  transform_flag = IM_HAL_TRANSFORM_ROT_90; break;
        case 180: transform_flag = IM_HAL_TRANSFORM_ROT_180; break;
        case 270: transform_flag = IM_HAL_TRANSFORM_ROT_270; break;
        default:  throw std::invalid_argument("rotate: angle must be 90, 180, or 270.");
    }

    int new_width = (angle_degrees == 180) ? this->width : this->height;
    int new_height = (angle_degrees == 180) ? this->height : this->width;
    im_rect src_rect = {0, 0, this->width, this->height};
    im_rect dst_rect = {0, 0, new_width, new_height};

    // Zero-copy path: wrap source DMA and rotate directly via RGA.
    // 零拷贝路径：封装源 DMA 并直接通过 RGA 旋转。
    if (this->is_zero_copy() && this->get_dma_fd() >= 0) {
        RgaDmaBuffer src_wrapper(this->get_dma_fd(), const_cast<void*>(this->get_data()), this->get_size(),
                                this->width, this->height, static_cast<int>(this->format), this->w_stride, this->h_stride);
        visiong::bufstate::prepare_device_read(src_wrapper, visiong::bufstate::BufferOwner::RGA);
        auto dst_sptr = std::make_shared<RgaDmaBuffer>(new_width, new_height, static_cast<int>(this->format));
        visiong::bufstate::prepare_device_write(*dst_sptr,
                                                visiong::bufstate::BufferOwner::RGA,
                                                visiong::bufstate::AccessIntent::Overwrite);
        if (improcess(src_wrapper.get_buffer(), dst_sptr->get_buffer(), {}, src_rect, dst_rect, {}, transform_flag) != IM_STATUS_SUCCESS) {
            throw std::runtime_error("RGA rotate (using improcess) failed");
        }
        ImageBuffer out(new_width, new_height, this->format, std::move(dst_sptr));
        visiong::bufstate::mark_device_write(out, visiong::bufstate::BufferOwner::RGA);
        return out;
    }

    RgaDmaBuffer src_dma(this->width, this->height, static_cast<int>(this->format));
    int bpp = get_bpp_for_format(this->format);
    copy_data_with_stride(src_dma.get_vir_addr(), src_dma.get_wstride() * bpp / 8, this->get_data(), this->w_stride * bpp / 8, this->height, this->width * bpp / 8);
    visiong::bufstate::mark_cpu_write(src_dma);
    visiong::bufstate::prepare_device_read(src_dma, visiong::bufstate::BufferOwner::RGA);

    RgaDmaBuffer dst_dma(new_width, new_height, this->format);
    visiong::bufstate::prepare_device_write(dst_dma,
                                            visiong::bufstate::BufferOwner::RGA,
                                            visiong::bufstate::AccessIntent::Overwrite);
    if (improcess(src_dma.get_buffer(), dst_dma.get_buffer(), {}, src_rect, dst_rect, {}, transform_flag) != IM_STATUS_SUCCESS) {
        throw std::runtime_error("RGA rotate (using improcess) failed");
    }
    visiong::bufstate::mark_device_write(dst_dma, visiong::bufstate::BufferOwner::RGA);
    visiong::bufstate::prepare_cpu_read(dst_dma);

    std::vector<unsigned char> new_data(static_cast<size_t>(new_width) * new_height * bpp / 8);
    copy_data_from_stride(new_data.data(), dst_dma.get_vir_addr(),
                          static_cast<size_t>(new_width) * bpp / 8, new_height,
                          dst_dma.get_wstride() * bpp / 8);
    return ImageBuffer(new_width, new_height, this->format, std::move(new_data));
}

ImageBuffer ImageBuffer::flip(bool horizontal, bool vertical) const {
    if (!is_valid()) throw std::runtime_error("flip: invalid source image");
    if (!horizontal && !vertical) return this->copy();

    im_rect rect = {0, 0, this->width, this->height};

    // Zero-copy path: run flip fully in DMA via RGA.
    // 零拷贝路径：通过 RGA 在 DMA 上完成翻转。
    if (this->is_zero_copy() && this->get_dma_fd() >= 0) {
        auto do_one_flip = [this, &rect](int transform_flag) -> ImageBuffer {
            RgaDmaBuffer src_wrapper(this->get_dma_fd(), const_cast<void*>(this->get_data()), this->get_size(),
                                    this->width, this->height, static_cast<int>(this->format), this->w_stride, this->h_stride);
            visiong::bufstate::prepare_device_read(src_wrapper, visiong::bufstate::BufferOwner::RGA);
            auto dst_sptr = std::make_shared<RgaDmaBuffer>(this->width, this->height, static_cast<int>(this->format));
            visiong::bufstate::prepare_device_write(*dst_sptr,
                                                    visiong::bufstate::BufferOwner::RGA,
                                                    visiong::bufstate::AccessIntent::Overwrite);
            if (improcess(src_wrapper.get_buffer(), dst_sptr->get_buffer(), {}, rect, rect, {}, transform_flag) != IM_STATUS_SUCCESS) {
                throw std::runtime_error("RGA flip failed");
            }
            ImageBuffer out(this->width, this->height, this->format, std::move(dst_sptr));
            visiong::bufstate::mark_device_write(out, visiong::bufstate::BufferOwner::RGA);
            return out;
        };
        if (horizontal && !vertical) return do_one_flip(IM_HAL_TRANSFORM_FLIP_H);
        if (!horizontal && vertical) return do_one_flip(IM_HAL_TRANSFORM_FLIP_V);
        // both: H then V
        // 双向翻转：先水平再垂直。
        ImageBuffer step1 = do_one_flip(IM_HAL_TRANSFORM_FLIP_H);
        RgaDmaBuffer src2(step1.get_dma_fd(), const_cast<void*>(step1.get_data()), step1.get_size(),
                         step1.width, step1.height, static_cast<int>(step1.format), step1.w_stride, step1.h_stride);
        visiong::bufstate::prepare_device_read(src2, visiong::bufstate::BufferOwner::RGA);
        auto dst2_sptr = std::make_shared<RgaDmaBuffer>(step1.width, step1.height, static_cast<int>(step1.format));
        visiong::bufstate::prepare_device_write(*dst2_sptr,
                                                visiong::bufstate::BufferOwner::RGA,
                                                visiong::bufstate::AccessIntent::Overwrite);
        if (improcess(src2.get_buffer(), dst2_sptr->get_buffer(), {}, rect, rect, {}, IM_HAL_TRANSFORM_FLIP_V) != IM_STATUS_SUCCESS) {
            throw std::runtime_error("RGA vertical flip failed");
        }
        ImageBuffer out(step1.width, step1.height, step1.format, std::move(dst2_sptr));
        visiong::bufstate::mark_device_write(out, visiong::bufstate::BufferOwner::RGA);
        return out;
    }

    ImageBuffer current_step_img = this->copy();

    if (horizontal) {
        RgaDmaBuffer src_dma(current_step_img.width, current_step_img.height, static_cast<int>(current_step_img.format));
        int bpp = get_bpp_for_format(current_step_img.format);
        copy_data_with_stride(src_dma.get_vir_addr(), src_dma.get_wstride() * bpp / 8, current_step_img.get_data(), current_step_img.w_stride * bpp / 8, current_step_img.height, current_step_img.width * bpp / 8);
        visiong::bufstate::mark_cpu_write(src_dma);
        visiong::bufstate::prepare_device_read(src_dma, visiong::bufstate::BufferOwner::RGA);

        RgaDmaBuffer dst_dma(current_step_img.width, current_step_img.height, current_step_img.format);
        visiong::bufstate::prepare_device_write(dst_dma,
                                                visiong::bufstate::BufferOwner::RGA,
                                                visiong::bufstate::AccessIntent::Overwrite);
        if (improcess(src_dma.get_buffer(), dst_dma.get_buffer(), {}, rect, rect, {}, IM_HAL_TRANSFORM_FLIP_H) != IM_STATUS_SUCCESS) {
            throw std::runtime_error("RGA horizontal flip failed");
        }
        visiong::bufstate::mark_device_write(dst_dma, visiong::bufstate::BufferOwner::RGA);
        visiong::bufstate::prepare_cpu_read(dst_dma);
        std::vector<unsigned char> new_data(static_cast<size_t>(dst_dma.get_width()) * dst_dma.get_height() * bpp / 8);
        copy_data_from_stride(new_data.data(), dst_dma.get_vir_addr(), (size_t)dst_dma.get_width() * bpp / 8, dst_dma.get_height(), dst_dma.get_wstride() * bpp / 8);
        current_step_img = ImageBuffer(dst_dma.get_width(), dst_dma.get_height(), current_step_img.format, std::move(new_data));
    }

    if (vertical) {
        RgaDmaBuffer src_dma(current_step_img.width, current_step_img.height, static_cast<int>(current_step_img.format));
        int bpp = get_bpp_for_format(current_step_img.format);
        copy_data_with_stride(src_dma.get_vir_addr(), src_dma.get_wstride() * bpp / 8, current_step_img.get_data(), current_step_img.w_stride * bpp / 8, current_step_img.height, current_step_img.width * bpp / 8);
        visiong::bufstate::mark_cpu_write(src_dma);
        visiong::bufstate::prepare_device_read(src_dma, visiong::bufstate::BufferOwner::RGA);

        RgaDmaBuffer dst_dma(current_step_img.width, current_step_img.height, current_step_img.format);
        visiong::bufstate::prepare_device_write(dst_dma,
                                                visiong::bufstate::BufferOwner::RGA,
                                                visiong::bufstate::AccessIntent::Overwrite);
        if (improcess(src_dma.get_buffer(), dst_dma.get_buffer(), {}, rect, rect, {}, IM_HAL_TRANSFORM_FLIP_V) != IM_STATUS_SUCCESS) {
            throw std::runtime_error("RGA vertical flip failed");
        }
        visiong::bufstate::mark_device_write(dst_dma, visiong::bufstate::BufferOwner::RGA);
        visiong::bufstate::prepare_cpu_read(dst_dma);
        std::vector<unsigned char> new_data(static_cast<size_t>(dst_dma.get_width()) * dst_dma.get_height() * bpp / 8);
        copy_data_from_stride(new_data.data(), dst_dma.get_vir_addr(), (size_t)dst_dma.get_width() * bpp / 8, dst_dma.get_height(), dst_dma.get_wstride() * bpp / 8);
        current_step_img = ImageBuffer(dst_dma.get_width(), dst_dma.get_height(), current_step_img.format, std::move(new_data));
    }

    return current_step_img;
}

ImageBuffer ImageBuffer::copy() const {
    if (!is_valid()) {
        return ImageBuffer();
    }

    ImageBuffer cloned;
    cloned.width = width;
    cloned.height = height;
    cloned.w_stride = w_stride;
    cloned.h_stride = h_stride;
    cloned.format = format;

    cloned.m_is_zero_copy = false;
    cloned.m_mb_blk_handle_sptr = nullptr;
    cloned.m_dma_buf_sptr = nullptr;
    cloned.m_external_keep_alive = nullptr;
    cloned.m_mb_blk = MB_INVALID_HANDLE;
    cloned.m_dma_fd = -1;
    cloned.m_vir_addr = nullptr;
    cloned.m_size = m_size;

    if (m_size > 0) {
        cloned.m_user_data.resize(m_size);
        if (this->is_zero_copy() && this->get_dma_fd() >= 0) {
            visiong::bufstate::prepare_cpu_read(*this);
        }
        const void* src = get_data();
        if (src == nullptr) {
            throw std::runtime_error("copy: source buffer data pointer is null.");
        }
        std::memcpy(cloned.m_user_data.data(), src, m_size);
    }

    return cloned;
}

// ============================================================================
// ============================================================================
// Analysis and shape-detection methods are implemented in ImageBufferAnalysis.cpp.
// 分析与形状检测方法实现位于 ImageBufferAnalysis.cpp。

// warp_perspective implementation.
// warp_perspective 的实现。
ImageBuffer ImageBuffer::warp_perspective(const Polygon& quad, int out_width, int out_height) const {
    if (!is_valid()) {
        throw std::runtime_error("warp_perspective: cannot operate on an invalid ImageBuffer.");
    }
    if (quad.size() != 4) {
        throw std::invalid_argument("warp_perspective: input quadrilateral must have exactly 4 points.");
    }
    if (out_width <= 0 || out_height <= 0) {
        throw std::invalid_argument("warp_perspective: output width and height must be positive.");
    }

    cv::Mat src_mat = image_buffer_to_bgr_mat_view(*this);
    if (src_mat.empty()) {
        throw std::runtime_error("warp_perspective: failed to get a valid cv::Mat view from the ImageBuffer.");
    }

    std::vector<cv::Point2f> src_points;
    for (const auto& p_tuple : quad) {
        src_points.emplace_back(static_cast<float>(std::get<0>(p_tuple)), static_cast<float>(std::get<1>(p_tuple)));
    }

    std::vector<cv::Point2f> dst_points = {
        {0.0f, 0.0f},
        {static_cast<float>(out_width - 1), 0.0f},
        {static_cast<float>(out_width - 1), static_cast<float>(out_height - 1)},
        {0.0f, static_cast<float>(out_height - 1)}
    };

    cv::Mat transform_matrix = cv::getPerspectiveTransform(src_points, dst_points);

    cv::Mat dst_mat;
    cv::warpPerspective(src_mat, dst_mat, transform_matrix, cv::Size(out_width, out_height));

    return ImageBuffer::from_cv_mat(dst_mat, RK_FMT_BGR888);
}


// Drawing, binarization, and blending methods are implemented in ImageBufferDrawing.cpp.
// 绘制、二值化与混合方法实现位于 ImageBufferDrawing.cpp。

