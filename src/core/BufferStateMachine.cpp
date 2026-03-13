// SPDX-License-Identifier: LGPL-3.0-or-later
#include "visiong/core/BufferStateMachine.h"

#include "visiong/core/ImageBuffer.h"
#include "visiong/core/RgaHelper.h"
#include "common/internal/dma_alloc.h"
#include "rk_mpi_mb.h"
#include "rk_mpi_sys.h"
#include "core/internal/logger.h"

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <unordered_map>

namespace visiong {
namespace bufstate {
namespace {

struct BufferRecord {
    DmaBufferView view{};
    BufferOwner owner = BufferOwner::Unknown;
    CacheDomain cache = CacheDomain::Unknown;
    FenceState fence{};
};

struct MetricsAtomic {
    std::atomic<uint64_t> cpu_to_device_sync_count{0};
    std::atomic<uint64_t> device_to_cpu_sync_count{0};
    std::atomic<uint64_t> skipped_sync_count{0};
    std::atomic<uint64_t> fence_pending_count{0};
    std::atomic<uint64_t> fence_resolved_count{0};
    std::atomic<uint64_t> trace_line_count{0};
};

std::mutex g_state_mutex;
std::unordered_map<int, BufferRecord> g_dma_state;
MetricsAtomic g_metrics;
std::once_flag g_observer_once;

bool env_enabled(const char* name, bool default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return default_value;
    }
    return (*value != '0');
}

const char* env_text(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return nullptr;
    }
    return value;
}

bool metrics_summary_enabled() {
    return env_enabled("VISIONG_DMA_STATE_STATS", false);
}

const char* metrics_file_path() {
    return env_text("VISIONG_DMA_STATE_STATS_FILE");
}

const char* trace_file_path() {
    return env_text("VISIONG_DMA_STATE_TRACE_FILE");
}

void emit_stderr_line(const std::string& line) {
    std::fprintf(stderr, "%s\n", line.c_str());
    std::fflush(stderr);
}

void append_line_to_file(const char* path, const std::string& line) {
    if (path == nullptr || *path == '\0') {
        return;
    }
    std::ofstream ofs(path, std::ios::out | std::ios::app);
    if (!ofs) {
        emit_stderr_line(std::string("[WARN][BufferState] trace_write_failed path=") + path +
                         " error=" + std::strerror(errno));
        return;
    }
    ofs << line << '\n';
}

void dump_metrics_on_exit();

void ensure_process_observers() {
    std::call_once(g_observer_once, []() { std::atexit(dump_metrics_on_exit); });
}

bool state_machine_enabled() {
    ensure_process_observers();
#if VISIONG_ENABLE_DMA_BUF_STATE_MACHINE
    return env_enabled("VISIONG_DMA_STATE_MACHINE", true);
#else
    return false;
#endif
}

bool trace_enabled() {
    ensure_process_observers();
    return env_enabled("VISIONG_DMA_STATE_TRACE", false);
}

void bump(std::atomic<uint64_t>& counter) {
    counter.fetch_add(1, std::memory_order_relaxed);
}

void reset_atomic(std::atomic<uint64_t>& counter) {
    counter.store(0, std::memory_order_relaxed);
}

uintptr_t ptr_id(void* ptr) {
    return reinterpret_cast<uintptr_t>(ptr);
}

BufferRecord& ensure_record_locked(const DmaBufferView& view) {
    BufferRecord& rec = g_dma_state[view.dma_fd];
    const uintptr_t incoming_ptr = ptr_id(view.virt_addr);
    const uintptr_t current_ptr = ptr_id(rec.view.virt_addr);
    if (rec.view.dma_fd != view.dma_fd || (incoming_ptr != 0 && current_ptr != 0 && incoming_ptr != current_ptr) ||
        (view.size != 0 && rec.view.size != 0 && view.size != rec.view.size)) {
        rec = BufferRecord{};
        rec.view = view;
        return rec;
    }
    if (rec.view.virt_addr == nullptr && view.virt_addr != nullptr) {
        rec.view.virt_addr = view.virt_addr;
    }
    if (rec.view.size == 0 && view.size != 0) {
        rec.view.size = view.size;
    }
    if (rec.view.dma_fd != view.dma_fd) {
        rec.view.dma_fd = view.dma_fd;
    }
    return rec;
}

BufferRecord* find_record_locked(const DmaBufferView& view) {
    if (!view.valid()) {
        return nullptr;
    }
    auto it = g_dma_state.find(view.dma_fd);
    if (it == g_dma_state.end()) {
        return nullptr;
    }
    BufferRecord& rec = it->second;
    const uintptr_t incoming_ptr = ptr_id(view.virt_addr);
    const uintptr_t current_ptr = ptr_id(rec.view.virt_addr);
    if (incoming_ptr != 0 && current_ptr != 0 && incoming_ptr != current_ptr) {
        return nullptr;
    }
    if (view.size != 0 && rec.view.size != 0 && view.size != rec.view.size) {
        return nullptr;
    }
    return &rec;
}

void emit_trace_line(const std::string& line) {
    if (!trace_enabled()) {
        return;
    }
    bump(g_metrics.trace_line_count);
    emit_stderr_line(line);
    append_line_to_file(trace_file_path(), line);
}

void record_skipped_sync(bool skipped) {
    if (skipped) {
        bump(g_metrics.skipped_sync_count);
    }
}

void trace_transition(const char* action,
                      const DmaBufferView& view,
                      const BufferRecord& before,
                      const BufferRecord& after,
                      bool did_sync,
                      bool sync_candidate) {
    if (!trace_enabled()) {
        return;
    }
    std::ostringstream oss;
    oss << "[INFO][BufferState] " << action << " fd=" << view.dma_fd << " sync=" << (did_sync ? 1 : 0)
        << " skipped=" << ((sync_candidate && !did_sync) ? 1 : 0)
        << " owner=" << to_cstr(before.owner) << "->" << to_cstr(after.owner)
        << " cache=" << to_cstr(before.cache) << "->" << to_cstr(after.cache)
        << " fence_seq=" << after.fence.seq << " fence_pending=" << (after.fence.pending ? 1 : 0)
        << " producer=" << to_cstr(after.fence.producer);
    emit_trace_line(oss.str());
}

void resolve_pending_fence_locked(BufferRecord& rec) {
    if (!rec.fence.pending) {
        return;
    }
    // Current pipeline uses synchronous IM_SYNC / blocking MPI submission. / 当前 pipeline uses synchronous IM_SYNC / blocking MPI submission.
    // Keep fence metadata for future async backends, but treat pending work as complete here.
    rec.fence.pending = false;
    rec.fence.fence_fd = -1;
    bump(g_metrics.fence_resolved_count);
}

bool should_sync_device_to_cpu(const BufferRecord& rec) {
    return rec.cache == CacheDomain::DeviceDirty ||
           (rec.cache == CacheDomain::Unknown && rec.owner != BufferOwner::CPU);
}

bool should_sync_cpu_to_device(const BufferRecord& rec) {
    return rec.cache == CacheDomain::CPUDirty ||
           (rec.cache == CacheDomain::Unknown && rec.owner == BufferOwner::CPU);
}

void sync_to_cpu(const DmaBufferView& view, MB_BLK blk) {
    if (!view.valid()) {
        return;
    }
    bump(g_metrics.device_to_cpu_sync_count);
    if (blk != MB_INVALID_HANDLE) {
        RK_MPI_SYS_MmzFlushCache(blk, RK_TRUE);
    } else {
        dma_sync_device_to_cpu(view.dma_fd);
    }
}

void sync_to_device(const DmaBufferView& view, MB_BLK blk) {
    if (!view.valid()) {
        return;
    }
    bump(g_metrics.cpu_to_device_sync_count);
    if (blk != MB_INVALID_HANDLE) {
        RK_MPI_SYS_MmzFlushCache(blk, RK_FALSE);
    } else {
        dma_sync_cpu_to_device(view.dma_fd);
    }
}

BufferStateMetrics collect_metrics() {
    BufferStateMetrics metrics;
    metrics.cpu_to_device_sync_count = g_metrics.cpu_to_device_sync_count.load(std::memory_order_relaxed);
    metrics.device_to_cpu_sync_count = g_metrics.device_to_cpu_sync_count.load(std::memory_order_relaxed);
    metrics.skipped_sync_count = g_metrics.skipped_sync_count.load(std::memory_order_relaxed);
    metrics.fence_pending_count = g_metrics.fence_pending_count.load(std::memory_order_relaxed);
    metrics.fence_resolved_count = g_metrics.fence_resolved_count.load(std::memory_order_relaxed);
    metrics.trace_line_count = g_metrics.trace_line_count.load(std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        metrics.tracked_buffer_count = static_cast<uint64_t>(g_dma_state.size());
    }
    return metrics;
}

std::string metrics_to_json(const BufferStateMetrics& metrics) {
    std::ostringstream oss;
    oss << "{\n"
        << "  \"state_machine_enabled\": " << (is_state_machine_enabled() ? "true" : "false") << ",\n"
        << "  \"trace_enabled\": " << (is_trace_enabled() ? "true" : "false") << ",\n"
        << "  \"cpu_to_device_sync_count\": " << metrics.cpu_to_device_sync_count << ",\n"
        << "  \"device_to_cpu_sync_count\": " << metrics.device_to_cpu_sync_count << ",\n"
        << "  \"skipped_sync_count\": " << metrics.skipped_sync_count << ",\n"
        << "  \"fence_pending_count\": " << metrics.fence_pending_count << ",\n"
        << "  \"fence_resolved_count\": " << metrics.fence_resolved_count << ",\n"
        << "  \"trace_line_count\": " << metrics.trace_line_count << ",\n"
        << "  \"tracked_buffer_count\": " << metrics.tracked_buffer_count << "\n"
        << "}";
    return oss.str();
}

std::string metrics_to_summary_line(const BufferStateMetrics& metrics) {
    std::ostringstream oss;
    oss << "state_machine=" << (is_state_machine_enabled() ? 1 : 0)
        << " trace=" << (is_trace_enabled() ? 1 : 0)
        << " c2d=" << metrics.cpu_to_device_sync_count
        << " d2c=" << metrics.device_to_cpu_sync_count
        << " skipped=" << metrics.skipped_sync_count
        << " fence_pending=" << metrics.fence_pending_count
        << " fence_resolved=" << metrics.fence_resolved_count
        << " trace_lines=" << metrics.trace_line_count
        << " tracked=" << metrics.tracked_buffer_count;
    return oss.str();
}

bool write_text_file(const char* path, const std::string& text) {
    if (path == nullptr || *path == '\0') {
        return false;
    }
    std::ofstream ofs(path, std::ios::out | std::ios::trunc);
    if (!ofs) {
        return false;
    }
    ofs << text;
    ofs.flush();
    return static_cast<bool>(ofs);
}

std::string dump_metrics_impl(const char* output_path, bool reset_after_dump, bool emit_summary) {
    ensure_process_observers();
    const BufferStateMetrics metrics = collect_metrics();
    const std::string json = metrics_to_json(metrics);

    bool wrote_file = false;
    if (output_path != nullptr && *output_path != '\0') {
        wrote_file = write_text_file(output_path, json);
        if (!wrote_file) {
            emit_stderr_line(std::string("[WARN][BufferStateMetrics] write_failed path=") + output_path +
                             " error=" + std::strerror(errno));
        }
    }

    if (emit_summary && (metrics_summary_enabled() || wrote_file)) {
        std::ostringstream oss;
        oss << "[INFO][BufferStateMetrics] " << metrics_to_summary_line(metrics);
        if (output_path != nullptr && *output_path != '\0') {
            oss << " path=" << output_path << " wrote=" << (wrote_file ? 1 : 0);
        }
        emit_stderr_line(oss.str());
    }

    if (reset_after_dump) {
        reset_metrics();
    }
    return json;
}

void dump_metrics_on_exit() {
    dump_metrics_impl(metrics_file_path(), false, true);
}

void adopt_dma_impl(const DmaBufferView& view, BufferOwner owner, CacheDomain cache) {
    if (!view.valid()) {
        return;
    }
    if (!state_machine_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_state_mutex);
    BufferRecord& rec = ensure_record_locked(view);
    if (owner != BufferOwner::Unknown) {
        rec.owner = owner;
    }
    if (cache != CacheDomain::Unknown) {
        rec.cache = cache;
    }
}

void prepare_cpu_read_impl(const DmaBufferView& view, MB_BLK blk) {
    if (!view.valid()) {
        return;
    }

    if (!state_machine_enabled()) {
        sync_to_cpu(view, blk);
        return;
    }

    std::lock_guard<std::mutex> lock(g_state_mutex);
    BufferRecord& rec = ensure_record_locked(view);
    BufferRecord before = rec;
    resolve_pending_fence_locked(rec);
    const bool sync_candidate = true;
    const bool do_sync = should_sync_device_to_cpu(rec);
    if (do_sync) {
        sync_to_cpu(view, blk);
    } else {
        record_skipped_sync(sync_candidate);
    }
    rec.owner = BufferOwner::CPU;
    if (rec.cache != CacheDomain::CPUDirty) {
        rec.cache = CacheDomain::CPUClean;
    }
    trace_transition("prepare_cpu_read", view, before, rec, do_sync, sync_candidate);
}

void prepare_cpu_write_impl(const DmaBufferView& view, AccessIntent intent) {
    if (!view.valid()) {
        return;
    }

    if (!state_machine_enabled()) {
        if (intent == AccessIntent::ReadModifyWrite) {
            sync_to_cpu(view, MB_INVALID_HANDLE);
        }
        return;
    }

    std::lock_guard<std::mutex> lock(g_state_mutex);
    BufferRecord& rec = ensure_record_locked(view);
    BufferRecord before = rec;
    resolve_pending_fence_locked(rec);
    const bool sync_candidate = (intent == AccessIntent::ReadModifyWrite);
    const bool do_sync = sync_candidate && should_sync_device_to_cpu(rec);
    if (do_sync) {
        sync_to_cpu(view, MB_INVALID_HANDLE);
    } else {
        record_skipped_sync(sync_candidate);
    }
    rec.owner = BufferOwner::CPU;
    if (intent == AccessIntent::ReadOnly && rec.cache != CacheDomain::CPUDirty) {
        rec.cache = CacheDomain::CPUClean;
    }
    trace_transition("prepare_cpu_write", view, before, rec, do_sync, sync_candidate);
}

void mark_cpu_write_impl(const DmaBufferView& view, BufferOwner owner) {
    if (!view.valid()) {
        return;
    }
    if (!state_machine_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_state_mutex);
    BufferRecord& rec = ensure_record_locked(view);
    BufferRecord before = rec;
    rec.owner = owner;
    rec.cache = CacheDomain::CPUDirty;
    rec.fence.pending = false;
    rec.fence.fence_fd = -1;
    rec.fence.producer = owner;
    trace_transition("mark_cpu_write", view, before, rec, false, false);
}

void prepare_device_read_impl(const DmaBufferView& view, MB_BLK blk, BufferOwner consumer) {
    if (!view.valid()) {
        return;
    }

    if (!state_machine_enabled()) {
        sync_to_device(view, blk);
        return;
    }

    std::lock_guard<std::mutex> lock(g_state_mutex);
    BufferRecord& rec = ensure_record_locked(view);
    BufferRecord before = rec;
    resolve_pending_fence_locked(rec);
    const bool sync_candidate = true;
    const bool do_sync = should_sync_cpu_to_device(rec);
    if (do_sync) {
        sync_to_device(view, blk);
    } else {
        record_skipped_sync(sync_candidate);
    }
    rec.owner = consumer;
    if (do_sync || rec.cache == CacheDomain::CPUClean) {
        rec.cache = CacheDomain::DeviceClean;
    }
    trace_transition("prepare_device_read", view, before, rec, do_sync, sync_candidate);
}

void prepare_device_write_impl(const DmaBufferView& view,
                               MB_BLK blk,
                               BufferOwner producer,
                               AccessIntent intent) {
    if (!view.valid()) {
        return;
    }

    if (!state_machine_enabled()) {
        if (intent == AccessIntent::ReadModifyWrite) {
            sync_to_device(view, blk);
        }
        return;
    }

    std::lock_guard<std::mutex> lock(g_state_mutex);
    BufferRecord& rec = ensure_record_locked(view);
    BufferRecord before = rec;
    resolve_pending_fence_locked(rec);
    const bool sync_candidate = (intent == AccessIntent::ReadModifyWrite);
    const bool do_sync = sync_candidate && should_sync_cpu_to_device(rec);
    if (do_sync) {
        sync_to_device(view, blk);
    } else {
        record_skipped_sync(sync_candidate);
    }
    rec.owner = producer;
    if (do_sync || rec.cache == CacheDomain::CPUClean) {
        rec.cache = CacheDomain::DeviceClean;
    }
    trace_transition("prepare_device_write", view, before, rec, do_sync, sync_candidate);
}

void mark_device_write_impl(const DmaBufferView& view,
                            BufferOwner producer,
                            bool fence_pending,
                            int fence_fd) {
    if (!view.valid()) {
        return;
    }
    if (!state_machine_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_state_mutex);
    BufferRecord& rec = ensure_record_locked(view);
    BufferRecord before = rec;
    rec.owner = producer;
    rec.cache = CacheDomain::DeviceDirty;
    rec.fence.seq += 1;
    rec.fence.pending = fence_pending;
    rec.fence.fence_fd = fence_pending ? fence_fd : -1;
    rec.fence.producer = producer;
    if (fence_pending) {
        bump(g_metrics.fence_pending_count);
    }
    trace_transition("mark_device_write", view, before, rec, false, false);
}

}  // namespace

const char* to_cstr(BufferOwner owner) {
    switch (owner) {
        case BufferOwner::CPU:
            return "CPU";
        case BufferOwner::Camera:
            return "Camera";
        case BufferOwner::RGA:
            return "RGA";
        case BufferOwner::NPU:
            return "NPU";
        case BufferOwner::VENC:
            return "VENC";
        case BufferOwner::IVE:
            return "IVE";
        case BufferOwner::External:
            return "External";
        case BufferOwner::Unknown:
        default:
            return "Unknown";
    }
}

const char* to_cstr(CacheDomain cache) {
    switch (cache) {
        case CacheDomain::CPUClean:
            return "CPUClean";
        case CacheDomain::CPUDirty:
            return "CPUDirty";
        case CacheDomain::DeviceClean:
            return "DeviceClean";
        case CacheDomain::DeviceDirty:
            return "DeviceDirty";
        case CacheDomain::Unknown:
        default:
            return "Unknown";
    }
}

DmaBufferView make_dma_view(const ImageBuffer& image) {
    return make_dma_view(image.get_dma_fd(), const_cast<void*>(image.get_data()), image.get_size());
}

DmaBufferView make_dma_view(const RgaDmaBuffer& dma) {
    return make_dma_view(dma.get_fd(), dma.get_vir_addr(), dma.get_size());
}

DmaBufferView make_dma_view(int dma_fd, void* virt_addr, size_t size) {
    DmaBufferView view;
    view.dma_fd = dma_fd;
    view.virt_addr = virt_addr;
    view.size = size;
    return view;
}

BufferStateSnapshot inspect(const DmaBufferView& view) {
    BufferStateSnapshot snapshot;
    snapshot.view = view;
    if (!view.valid()) {
        return snapshot;
    }
    std::lock_guard<std::mutex> lock(g_state_mutex);
    BufferRecord* rec = find_record_locked(view);
    if (rec == nullptr) {
        return snapshot;
    }
    snapshot.tracked = true;
    snapshot.owner = rec->owner;
    snapshot.cache = rec->cache;
    snapshot.fence = rec->fence;
    snapshot.view = rec->view;
    return snapshot;
}

BufferStateSnapshot inspect(const ImageBuffer& image) {
    return inspect(make_dma_view(image));
}

BufferStateSnapshot inspect(const RgaDmaBuffer& dma) {
    return inspect(make_dma_view(dma));
}

BufferStateMetrics get_metrics() {
    return collect_metrics();
}

void reset_metrics() {
    reset_atomic(g_metrics.cpu_to_device_sync_count);
    reset_atomic(g_metrics.device_to_cpu_sync_count);
    reset_atomic(g_metrics.skipped_sync_count);
    reset_atomic(g_metrics.fence_pending_count);
    reset_atomic(g_metrics.fence_resolved_count);
    reset_atomic(g_metrics.trace_line_count);
}

std::string dump_metrics(const char* output_path, bool reset_after_dump) {
    return dump_metrics_impl(output_path, reset_after_dump, true);
}

bool is_state_machine_enabled() {
    return state_machine_enabled();
}

bool is_trace_enabled() {
    return trace_enabled();
}

void adopt_dma(const DmaBufferView& view, BufferOwner owner, CacheDomain cache) {
    adopt_dma_impl(view, owner, cache);
}

void adopt_dma(const ImageBuffer& image, BufferOwner owner, CacheDomain cache) {
    adopt_dma_impl(make_dma_view(image), owner, cache);
}

void adopt_dma(const RgaDmaBuffer& dma, BufferOwner owner, CacheDomain cache) {
    adopt_dma_impl(make_dma_view(dma), owner, cache);
}

void prepare_cpu_read(const DmaBufferView& view) {
    prepare_cpu_read_impl(view, MB_INVALID_HANDLE);
}

void prepare_cpu_read(const ImageBuffer& image) {
    prepare_cpu_read_impl(make_dma_view(image), MB_INVALID_HANDLE);
}

void prepare_cpu_read(const RgaDmaBuffer& dma) {
    prepare_cpu_read_impl(make_dma_view(dma), MB_INVALID_HANDLE);
}

void prepare_cpu_write(const DmaBufferView& view, AccessIntent intent) {
    prepare_cpu_write_impl(view, intent);
}

void prepare_cpu_write(const ImageBuffer& image, AccessIntent intent) {
    prepare_cpu_write_impl(make_dma_view(image), intent);
}

void prepare_cpu_write(const RgaDmaBuffer& dma, AccessIntent intent) {
    prepare_cpu_write_impl(make_dma_view(dma), intent);
}

void mark_cpu_write(const DmaBufferView& view, BufferOwner owner) {
    mark_cpu_write_impl(view, owner);
}

void mark_cpu_write(const ImageBuffer& image, BufferOwner owner) {
    mark_cpu_write_impl(make_dma_view(image), owner);
}

void mark_cpu_write(const RgaDmaBuffer& dma, BufferOwner owner) {
    mark_cpu_write_impl(make_dma_view(dma), owner);
}

void prepare_device_read(const DmaBufferView& view, BufferOwner consumer) {
    prepare_device_read_impl(view, MB_INVALID_HANDLE, consumer);
}

void prepare_device_read(const ImageBuffer& image, BufferOwner consumer) {
    prepare_device_read_impl(make_dma_view(image), MB_INVALID_HANDLE, consumer);
}

void prepare_device_read(const RgaDmaBuffer& dma, BufferOwner consumer) {
    prepare_device_read_impl(make_dma_view(dma), MB_INVALID_HANDLE, consumer);
}

void prepare_device_write(const DmaBufferView& view, BufferOwner producer, AccessIntent intent) {
    prepare_device_write_impl(view, MB_INVALID_HANDLE, producer, intent);
}

void prepare_device_write(const ImageBuffer& image, BufferOwner producer, AccessIntent intent) {
    prepare_device_write_impl(make_dma_view(image), MB_INVALID_HANDLE, producer, intent);
}

void prepare_device_write(const RgaDmaBuffer& dma, BufferOwner producer, AccessIntent intent) {
    prepare_device_write_impl(make_dma_view(dma), MB_INVALID_HANDLE, producer, intent);
}

void mark_device_write(const DmaBufferView& view, BufferOwner producer, bool fence_pending, int fence_fd) {
    mark_device_write_impl(view, producer, fence_pending, fence_fd);
}

void mark_device_write(const ImageBuffer& image, BufferOwner producer, bool fence_pending, int fence_fd) {
    mark_device_write_impl(make_dma_view(image), producer, fence_pending, fence_fd);
}

void mark_device_write(const RgaDmaBuffer& dma, BufferOwner producer, bool fence_pending, int fence_fd) {
    mark_device_write_impl(make_dma_view(dma), producer, fence_pending, fence_fd);
}

void prepare_mb_cpu_read(MB_BLK blk, int dma_fd) {
    if (blk == MB_INVALID_HANDLE && dma_fd < 0) {
        return;
    }
    void* virt_addr = (blk != MB_INVALID_HANDLE) ? RK_MPI_MB_Handle2VirAddr(blk) : nullptr;
    const size_t size = (blk != MB_INVALID_HANDLE) ? RK_MPI_MB_GetSize(blk) : 0;
    prepare_cpu_read_impl(make_dma_view(dma_fd, virt_addr, size), blk);
}

void prepare_mb_device_read(MB_BLK blk, int dma_fd, BufferOwner consumer) {
    if (blk == MB_INVALID_HANDLE && dma_fd < 0) {
        return;
    }
    void* virt_addr = (blk != MB_INVALID_HANDLE) ? RK_MPI_MB_Handle2VirAddr(blk) : nullptr;
    const size_t size = (blk != MB_INVALID_HANDLE) ? RK_MPI_MB_GetSize(blk) : 0;
    prepare_device_read_impl(make_dma_view(dma_fd, virt_addr, size), blk, consumer);
}

void mark_mb_device_write(MB_BLK blk, int dma_fd, BufferOwner producer) {
    if (blk == MB_INVALID_HANDLE && dma_fd < 0) {
        return;
    }
    void* virt_addr = (blk != MB_INVALID_HANDLE) ? RK_MPI_MB_Handle2VirAddr(blk) : nullptr;
    const size_t size = (blk != MB_INVALID_HANDLE) ? RK_MPI_MB_GetSize(blk) : 0;
    mark_device_write_impl(make_dma_view(dma_fd, virt_addr, size), producer, false, -1);
}

void mark_mb_cpu_write(MB_BLK blk, int dma_fd, BufferOwner owner) {
    if (blk == MB_INVALID_HANDLE && dma_fd < 0) {
        return;
    }
    void* virt_addr = (blk != MB_INVALID_HANDLE) ? RK_MPI_MB_Handle2VirAddr(blk) : nullptr;
    const size_t size = (blk != MB_INVALID_HANDLE) ? RK_MPI_MB_GetSize(blk) : 0;
    mark_cpu_write_impl(make_dma_view(dma_fd, virt_addr, size), owner);
}

}  // namespace bufstate
}  // namespace visiong

