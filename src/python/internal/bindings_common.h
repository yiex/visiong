// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "visiong/core/Camera.h"
#include "core/internal/rga_utils.h"
#include "visiong/modules/DisplayFB.h"
#include "visiong/modules/DisplayHTTP.h"
#include "visiong/modules/DisplayHTTPFLV.h"
#include "visiong/modules/DisplayRTSP.h"
#include "visiong/modules/DisplayUDP.h"
#include "visiong/modules/Touch.h"
#include "visiong/modules/VencRecorder.h"
#include "im2d.hpp"
#include "visiong/common/build_config.h"
#include "visiong/common/pixel_format.h"
#include "common/internal/string_utils.h"

#if VISIONG_WITH_GUI
#include "visiong/modules/GUIManager.h"
#include "nuklear.h"
#endif

#if VISIONG_WITH_IVE
#include "visiong/modules/IVE.h"
#endif

#if VISIONG_WITH_NPU
#include "visiong/npu/NPU.h"
#include "visiong/npu/LowLevelNPU.h"
#include "visiong/npu/PPOCR.h"
#include <visiong/npu/NanoTrack.h>
#endif

namespace py = pybind11;
using namespace pybind11::literals;

inline constexpr PIXEL_FORMAT_E kGray8 = visiong::kGray8Format;

std::string get_image_buffer_compact_bytes(const ImageBuffer& img);
py::bytes get_image_buffer_bytes(const ImageBuffer& img);
py::array make_image_buffer_numpy_view(ImageBuffer& img);

DisplayFB::Mode parse_displayfb_mode(const std::string& mode_str);
#if VISIONG_WITH_NPU
ModelType parse_model_type(const std::string& model_type_str);
#endif
DisplayRTSP::Codec parse_rtsp_codec(const std::string& codec_str);
DisplayRTSP::RcMode parse_rtsp_rc_mode(const std::string& rc_mode_str);
VencRecorder::Codec parse_venc_codec(const std::string& codec_str);
VencRecorder::Container parse_venc_container(const std::string& container_str);
DisplayHTTPFLV::Codec parse_httpflv_codec(const std::string& codec_str);
DisplayHTTPFLV::RcMode parse_httpflv_rc_mode(const std::string& rc_mode_str);

void bind_core_types(py::module_& m);
void bind_image_buffer(py::module_& m);
void bind_camera(py::module_& m);
#if VISIONG_WITH_IVE
void bind_ive(py::module_& m);
#endif
void bind_io_devices(py::module_& m);
void bind_pinmux(py::module_& m);
#if VISIONG_WITH_NPU
void bind_npu(py::module_& m);
#endif
#if VISIONG_WITH_GUI
void bind_gui(py::module_& m);
#endif
void bind_streaming(py::module_& m);

