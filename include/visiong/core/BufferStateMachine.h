// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_CORE_BUFFERSTATEMACHINE_H
#define VISIONG_CORE_BUFFERSTATEMACHINE_H

#include <cstddef>
#include <cstdint>
#include <string>

#include "rk_comm_mb.h"

struct ImageBuffer;
class RgaDmaBuffer;

namespace visiong {
namespace bufstate {

#ifndef VISIONG_ENABLE_DMA_BUF_STATE_MACHINE
#define VISIONG_ENABLE_DMA_BUF_STATE_MACHINE 1
#endif

enum class BufferOwner : uint8_t {
    Unknown = 0,
    CPU,
    Camera,
    RGA,
    NPU,
    VENC,
    IVE,
    External,
};

enum class CacheDomain : uint8_t {
    Unknown = 0,
    CPUClean,
    CPUDirty,
    DeviceClean,
    DeviceDirty,
};

enum class AccessIntent : uint8_t {
    ReadOnly = 0,
    Overwrite,
    ReadModifyWrite,
};

struct FenceState {
    uint64_t seq = 0;
    int fence_fd = -1;
    bool pending = false;
    BufferOwner producer = BufferOwner::Unknown;
};

struct DmaBufferView {
    int dma_fd = -1;
    void* virt_addr = nullptr;
    size_t size = 0;

    bool valid() const { return dma_fd >= 0; }
};

struct BufferStateSnapshot {
    bool tracked = false;
    DmaBufferView view{};
    BufferOwner owner = BufferOwner::Unknown;
    CacheDomain cache = CacheDomain::Unknown;
    FenceState fence{};
};

struct BufferStateMetrics {
    uint64_t cpu_to_device_sync_count = 0;
    uint64_t device_to_cpu_sync_count = 0;
    uint64_t skipped_sync_count = 0;
    uint64_t fence_pending_count = 0;
    uint64_t fence_resolved_count = 0;
    uint64_t trace_line_count = 0;
    uint64_t tracked_buffer_count = 0;
};

DmaBufferView make_dma_view(const ImageBuffer& image);
DmaBufferView make_dma_view(const RgaDmaBuffer& dma);
DmaBufferView make_dma_view(int dma_fd, void* virt_addr, size_t size);

BufferStateSnapshot inspect(const DmaBufferView& view);
BufferStateSnapshot inspect(const ImageBuffer& image);
BufferStateSnapshot inspect(const RgaDmaBuffer& dma);

BufferStateMetrics get_metrics();
void reset_metrics();
std::string dump_metrics(const char* output_path = nullptr, bool reset_after_dump = false);
bool is_state_machine_enabled();
bool is_trace_enabled();

void adopt_dma(const DmaBufferView& view, BufferOwner owner, CacheDomain cache);
void adopt_dma(const ImageBuffer& image, BufferOwner owner, CacheDomain cache);
void adopt_dma(const RgaDmaBuffer& dma, BufferOwner owner, CacheDomain cache);

void prepare_cpu_read(const DmaBufferView& view);
void prepare_cpu_read(const ImageBuffer& image);
void prepare_cpu_read(const RgaDmaBuffer& dma);

void prepare_cpu_write(const DmaBufferView& view, AccessIntent intent = AccessIntent::ReadModifyWrite);
void prepare_cpu_write(const ImageBuffer& image, AccessIntent intent = AccessIntent::ReadModifyWrite);
void prepare_cpu_write(const RgaDmaBuffer& dma, AccessIntent intent = AccessIntent::ReadModifyWrite);

void mark_cpu_write(const DmaBufferView& view, BufferOwner owner = BufferOwner::CPU);
void mark_cpu_write(const ImageBuffer& image, BufferOwner owner = BufferOwner::CPU);
void mark_cpu_write(const RgaDmaBuffer& dma, BufferOwner owner = BufferOwner::CPU);

void prepare_device_read(const DmaBufferView& view, BufferOwner consumer);
void prepare_device_read(const ImageBuffer& image, BufferOwner consumer);
void prepare_device_read(const RgaDmaBuffer& dma, BufferOwner consumer);

void prepare_device_write(const DmaBufferView& view,
                          BufferOwner producer,
                          AccessIntent intent = AccessIntent::Overwrite);
void prepare_device_write(const ImageBuffer& image,
                          BufferOwner producer,
                          AccessIntent intent = AccessIntent::Overwrite);
void prepare_device_write(const RgaDmaBuffer& dma,
                          BufferOwner producer,
                          AccessIntent intent = AccessIntent::Overwrite);

void mark_device_write(const DmaBufferView& view,
                       BufferOwner producer,
                       bool fence_pending = false,
                       int fence_fd = -1);
void mark_device_write(const ImageBuffer& image,
                       BufferOwner producer,
                       bool fence_pending = false,
                       int fence_fd = -1);
void mark_device_write(const RgaDmaBuffer& dma,
                       BufferOwner producer,
                       bool fence_pending = false,
                       int fence_fd = -1);

void prepare_mb_cpu_read(MB_BLK blk, int dma_fd);
void prepare_mb_device_read(MB_BLK blk, int dma_fd, BufferOwner consumer);
void mark_mb_device_write(MB_BLK blk, int dma_fd, BufferOwner producer);
void mark_mb_cpu_write(MB_BLK blk, int dma_fd, BufferOwner owner = BufferOwner::CPU);

const char* to_cstr(BufferOwner owner);
const char* to_cstr(CacheDomain cache);

}  // namespace bufstate
}  // namespace visiong

#endif  // VISIONG_CORE_BUFFERSTATEMACHINE_H

