// SPDX-License-Identifier: LGPL-3.0-or-later
#include "internal/bindings_common.h"
#include "visiong/core/BufferStateMachine.h"
#include "python/internal/license_guard.h"
#include "visiong/common/LegacyKey.h"

#include <string>

PYBIND11_MODULE(_visiong, m) {
    py::print(visiong::python::community_banner());

    m.def("get_unique_id", &visiong::python::get_unique_id,
          "Reads the unique 6-byte chip ID from the RV1106 OTP area and returns it as a 12-character hex string.");

    m.def("decrypt_legacy_value",
          &visiong::legacy::decrypt_legacy_value,
          py::arg("key"),
          "Decrypts the stored legacy value with a 64-character hex key.");

    m.def("dma_state_metrics",
          [](bool reset) {
              const auto metrics = visiong::bufstate::get_metrics();
              py::dict out;
              out["state_machine_enabled"] = visiong::bufstate::is_state_machine_enabled();
              out["trace_enabled"] = visiong::bufstate::is_trace_enabled();
              out["cpu_to_device_sync_count"] = py::int_(metrics.cpu_to_device_sync_count);
              out["device_to_cpu_sync_count"] = py::int_(metrics.device_to_cpu_sync_count);
              out["skipped_sync_count"] = py::int_(metrics.skipped_sync_count);
              out["fence_pending_count"] = py::int_(metrics.fence_pending_count);
              out["fence_resolved_count"] = py::int_(metrics.fence_resolved_count);
              out["trace_line_count"] = py::int_(metrics.trace_line_count);
              out["tracked_buffer_count"] = py::int_(metrics.tracked_buffer_count);
              if (reset) {
                  visiong::bufstate::reset_metrics();
              }
              return out;
          },
          py::arg("reset") = false,
          "Returns dma-buf state-machine counters as a dict. Set reset=True to clear counters after reading.");

    m.def("dma_state_reset_metrics",
          &visiong::bufstate::reset_metrics,
          "Resets dma-buf state-machine counters.");

    m.def("dma_state_dump_metrics",
          [](const std::string& output_path, bool reset_after_dump) {
              const char* path = output_path.empty() ? nullptr : output_path.c_str();
              return visiong::bufstate::dump_metrics(path, reset_after_dump);
          },
          py::arg("output_path") = "",
          py::arg("reset_after_dump") = false,
          "Returns dma-buf state-machine counters as JSON and optionally writes them to output_path.");

    std::string module_doc = "Python bindings for Rockchip camera, display, and image processing.";
#if VISIONG_WITH_IVE
    module_doc += " IVE enabled.";
#endif
#if VISIONG_WITH_NPU
    module_doc += " NPU enabled.";
#endif
#if VISIONG_WITH_GUI
    module_doc += " GUI enabled.";
#endif
    m.doc() = module_doc.c_str();

    bind_core_types(m);
    bind_audio(m);
    bind_image_buffer(m);
    bind_camera(m);
#if VISIONG_WITH_IVE
    bind_ive(m);
#endif
    bind_io_devices(m);
    bind_pinmux(m);
#if VISIONG_WITH_NPU
    bind_npu(m);
#endif
#if VISIONG_WITH_GUI
    bind_gui(m);
#endif
    bind_streaming(m);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "1.0.4";
#endif
}
