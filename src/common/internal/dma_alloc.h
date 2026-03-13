// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_COMMON_INTERNAL_DMA_ALLOC_H
#define VISIONG_COMMON_INTERNAL_DMA_ALLOC_H

#include <cstddef>

inline constexpr char DMA_HEAP_UNCACHE_PATH[] = "/dev/dma_heap/system-uncached";
inline constexpr char DMA_HEAP_PATH[] = "/dev/dma_heap/system";
inline constexpr char DMA_HEAP_DMA32_UNCACHE_PATH[] = "/dev/dma_heap/system-uncached-dma32";
inline constexpr char DMA_HEAP_DMA32_PATH[] = "/dev/dma_heap/system-dma32";
inline constexpr char CMA_HEAP_UNCACHE_PATH[] = "/dev/dma_heap/cma-uncached";
inline constexpr char RV1106_CMA_HEAP_PATH[] = "/dev/rk_dma_heap/rk-dma-heap-cma";

int dma_sync_device_to_cpu(int fd);
int dma_sync_cpu_to_device(int fd);
int dma_buf_alloc(const char* path, size_t size, int* fd, void** va);
void dma_buf_free(size_t size, int* fd, void* va);

#endif  // VISIONG_COMMON_INTERNAL_DMA_ALLOC_H
