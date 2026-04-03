// SPDX-License-Identifier: LGPL-3.0-or-later
#include "internal/bindings_common.h"
#include "visiong/audio/KwsFrontend.h"

#include <cstring>

namespace {

py::array_t<float> make_frontend_output_array(const std::vector<float>& values,
                                              const std::vector<int64_t>& shape64) {
    std::vector<ssize_t> shape;
    shape.reserve(shape64.size());
    for (int64_t dim : shape64) {
        shape.push_back(static_cast<ssize_t>(dim));
    }
    py::array_t<float> out(shape);
    if (!values.empty()) {
        std::memcpy(out.mutable_data(), values.data(), values.size() * sizeof(float));
    }
    return out;
}

visiong::audio::KwsLogMelConfig build_kws_config(int sample_rate,
                                                 int clip_samples,
                                                 int window_size_ms,
                                                 int window_stride_ms,
                                                 int fft_size,
                                                 int num_mel_bins,
                                                 float lower_edge_hertz,
                                                 float upper_edge_hertz,
                                                 float epsilon,
                                                 bool normalize) {
    visiong::audio::KwsLogMelConfig config;
    config.sample_rate = sample_rate;
    config.clip_samples = clip_samples;
    config.window_size_ms = window_size_ms;
    config.window_stride_ms = window_stride_ms;
    config.fft_size = fft_size;
    config.num_mel_bins = num_mel_bins;
    config.lower_edge_hertz = lower_edge_hertz;
    config.upper_edge_hertz = upper_edge_hertz;
    config.epsilon = epsilon;
    config.normalize = normalize;
    return config;
}

std::vector<int64_t> parse_output_shape(const visiong::audio::KwsLogMelFrontend& frontend,
                                        const std::string& output_format) {
    const std::string format = visiong::to_lower_copy(output_format);
    if (format == "hw" || format == "frames_bins") {
        return frontend.output_shape_hw();
    }
    if (format == "nchw" || format == "tensor") {
        return frontend.output_shape_nchw();
    }
    throw std::invalid_argument("KwsLogMelFrontend output_format must be 'hw' or 'nchw'.");
}

}  // namespace

void bind_audio(py::module_& m) {
    using visiong::audio::KwsLogMelFrontend;

    py::class_<KwsLogMelFrontend>(m, "KwsLogMelFrontend",
                                  "Native log-mel frontend for keyword spotting and other short audio classifiers.")
        .def(py::init([](int sample_rate,
                         int clip_samples,
                         int window_size_ms,
                         int window_stride_ms,
                         int fft_size,
                         int num_mel_bins,
                         float lower_edge_hertz,
                         float upper_edge_hertz,
                         float epsilon,
                         bool normalize) {
                 return std::make_unique<KwsLogMelFrontend>(
                     build_kws_config(sample_rate,
                                      clip_samples,
                                      window_size_ms,
                                      window_stride_ms,
                                      fft_size,
                                      num_mel_bins,
                                      lower_edge_hertz,
                                      upper_edge_hertz,
                                      epsilon,
                                      normalize));
             }),
             "sample_rate"_a = 16000,
             "clip_samples"_a = 16000,
             "window_size_ms"_a = 30,
             "window_stride_ms"_a = 20,
             "fft_size"_a = 512,
             "num_mel_bins"_a = 40,
             "lower_edge_hertz"_a = 20.0f,
             "upper_edge_hertz"_a = 4000.0f,
             "epsilon"_a = 1e-6f,
             "normalize"_a = true,
             "Creates a native KWS log-mel frontend. Default settings match the RV1106 keyword spotting tutorial.")
        .def_property_readonly("sample_rate", &KwsLogMelFrontend::sample_rate)
        .def_property_readonly("clip_samples", &KwsLogMelFrontend::clip_samples)
        .def_property_readonly("frame_length", &KwsLogMelFrontend::frame_length)
        .def_property_readonly("frame_step", &KwsLogMelFrontend::frame_step)
        .def_property_readonly("fft_size", &KwsLogMelFrontend::fft_size)
        .def_property_readonly("num_frames", &KwsLogMelFrontend::num_frames)
        .def_property_readonly("num_mel_bins", &KwsLogMelFrontend::num_mel_bins)
        .def("output_shape",
             [](const KwsLogMelFrontend& self, const std::string& output_format) {
                 return parse_output_shape(self, output_format);
             },
             "output_format"_a = "nchw")
        .def("compute_from_pcm16",
             [](const KwsLogMelFrontend& self,
                py::array_t<int16_t, py::array::c_style | py::array::forcecast> audio,
                const std::string& output_format) {
                 const py::buffer_info info = audio.request();
                 std::vector<float> values;
                 {
                     py::gil_scoped_release release;
                     values = self.compute_from_pcm16(static_cast<const int16_t*>(info.ptr),
                                                      static_cast<size_t>(info.size));
                 }
                 return make_frontend_output_array(values, parse_output_shape(self, output_format));
             },
             "audio"_a,
             "output_format"_a = "nchw",
             "Computes normalized log-mel features from a PCM16 numpy array.")
        .def("compute_from_float",
             [](const KwsLogMelFrontend& self,
                py::array_t<float, py::array::c_style | py::array::forcecast> audio,
                const std::string& output_format) {
                 const py::buffer_info info = audio.request();
                 std::vector<float> values;
                 {
                     py::gil_scoped_release release;
                     values = self.compute_from_float(static_cast<const float*>(info.ptr),
                                                      static_cast<size_t>(info.size));
                 }
                 return make_frontend_output_array(values, parse_output_shape(self, output_format));
             },
             "audio"_a,
             "output_format"_a = "nchw",
             "Computes normalized log-mel features from a float32 numpy array.");
}
