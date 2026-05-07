/* SPDX-License-Identifier: GPL-2.0-only WITH Linux-syscall-note */
#ifndef VISIONG_UAPI_VISIONG_HW_H
#define VISIONG_UAPI_VISIONG_HW_H

#include <linux/ioctl.h>
#include <linux/types.h>

#define VISIONG_HW_ABI_VERSION 1U
#define VISIONG_HW_DRIVER_VERSION 10U

#define VISIONG_HW_FEATURE_REG_ACCESS   (1U << 0)
#define VISIONG_HW_FEATURE_PIN_SESSION  (1U << 1)
#define VISIONG_HW_FEATURE_GPIO_IRQ     (1U << 2)
#define VISIONG_HW_FEATURE_DMA_BUFFER   (1U << 3)
#define VISIONG_HW_FEATURE_DMA_MEMCPY   (1U << 4)
#define VISIONG_HW_FEATURE_SPI_DISPLAY  (1U << 5)
#define VISIONG_HW_FEATURE_DMA_FILL     (1U << 6)
#define VISIONG_HW_FEATURE_SPI_REG      (1U << 7)

enum visiong_hw_reg_block {
	VISIONG_HW_REG_BLOCK_IOC = 0,
	VISIONG_HW_REG_BLOCK_PMUIOC = 1,
	VISIONG_HW_REG_BLOCK_CRU = 2,
	VISIONG_HW_REG_BLOCK_GPIO0 = 3,
	VISIONG_HW_REG_BLOCK_GPIO1 = 4,
	VISIONG_HW_REG_BLOCK_GPIO2 = 5,
	VISIONG_HW_REG_BLOCK_GPIO3 = 6,
	VISIONG_HW_REG_BLOCK_GPIO4 = 7,
	VISIONG_HW_REG_BLOCK_SPI0 = 8,
	VISIONG_HW_REG_BLOCK_SPI1 = 9,
	VISIONG_HW_REG_BLOCK_I2C0 = 10,
	VISIONG_HW_REG_BLOCK_I2C1 = 11,
	VISIONG_HW_REG_BLOCK_I2C2 = 12,
	VISIONG_HW_REG_BLOCK_I2C3 = 13,
	VISIONG_HW_REG_BLOCK_I2C4 = 14,
	VISIONG_HW_REG_BLOCK_UART0 = 15,
	VISIONG_HW_REG_BLOCK_UART1 = 16,
	VISIONG_HW_REG_BLOCK_UART2 = 17,
	VISIONG_HW_REG_BLOCK_UART3 = 18,
	VISIONG_HW_REG_BLOCK_UART4 = 19,
	VISIONG_HW_REG_BLOCK_UART5 = 20,
	VISIONG_HW_REG_BLOCK_PWM0_3 = 21,
	VISIONG_HW_REG_BLOCK_PWM4_7 = 22,
	VISIONG_HW_REG_BLOCK_PWM8_11 = 23,
	VISIONG_HW_REG_BLOCK_DMAC = 24,
	VISIONG_HW_REG_BLOCK_GICD = 25,
	VISIONG_HW_REG_BLOCK_COUNT = 26,
};

#define VISIONG_HW_REG_FLAG_HIWORD_UPDATE (1U << 0)

#define VISIONG_HW_DMA_ALLOC_WRITE_COMBINE (1U << 0)
#define VISIONG_HW_DMA_SYNC_BIDIRECTIONAL  0U
#define VISIONG_HW_DMA_SYNC_TO_DEVICE      1U
#define VISIONG_HW_DMA_SYNC_FROM_DEVICE    2U
#define VISIONG_HW_DMA_SYNC_START          (1U << 0)
#define VISIONG_HW_DMA_SYNC_END            (1U << 1)
#define VISIONG_HW_DMA_MEMCPY_STATUS_DONE  0U
#define VISIONG_HW_DMA_MEMCPY_STATUS_TIMEOUT 1U
#define VISIONG_HW_DMA_MEMCPY_STATUS_ERROR 2U
#define VISIONG_HW_DMA_MEMCPY_STATUS_PENDING 3U
#define VISIONG_HW_DMA_MEMCPY_ASYNC        (1U << 0)

#define VISIONG_HW_SPI_STATUS_DONE         0U
#define VISIONG_HW_SPI_STATUS_TIMEOUT      1U
#define VISIONG_HW_SPI_STATUS_ERROR        2U
#define VISIONG_HW_SPI_STATUS_PENDING      3U
#define VISIONG_HW_SPI_REG_TX_ONLY         (1U << 0)

#define VISIONG_HW_IRQ_EDGE_RISING         (1U << 0)
#define VISIONG_HW_IRQ_EDGE_FALLING        (1U << 1)
#define VISIONG_HW_IRQ_EDGE_BOTH           (VISIONG_HW_IRQ_EDGE_RISING | VISIONG_HW_IRQ_EDGE_FALLING)

#define VISIONG_HW_WAIT_STATUS_EVENT       0U
#define VISIONG_HW_WAIT_STATUS_TIMEOUT     1U

struct visiong_hw_caps {
	__u32 size;
	__u32 abi_version;
	__u32 driver_version;
	__u32 feature_flags;
	__u32 chip_id;
	__u32 max_dma_bytes;
	__u32 max_transfer_bytes;
	__u32 reserved[9];
};

struct visiong_hw_reg_access {
	__u32 size;
	__u32 block;
	__u32 offset;
	__u32 value;
	__u32 mask;
	__u32 flags;
	__u32 reserved[2];
};

struct visiong_hw_dma_alloc {
	__u32 size;
	__u32 bytes;
	__u32 flags;
	__s32 fd;
	__u32 reserved[4];
};

struct visiong_hw_dma_sync {
	__u32 size;
	__s32 fd;
	__u32 direction;
	__u32 flags;
	__u32 offset;
	__u32 bytes;
	__u32 reserved[4];
};

struct visiong_hw_dma_memcpy {
	__u32 size;
	__s32 dst_fd;
	__s32 src_fd;
	__u32 dst_offset;
	__u32 src_offset;
	__u32 bytes;
	__u32 flags;
	__u32 status;
	__u32 handle;
	__u32 reserved[3];
};

struct visiong_hw_dma_fill {
	__u32 size;
	__s32 fd;
	__u32 offset;
	__u32 bytes;
	__u32 value;
	__u32 flags;
	__u32 status;
	__u32 reserved[5];
};

struct visiong_hw_wait {
	__u32 size;
	__u32 handle;
	__s32 timeout_ms;
	__u32 status;
	__u32 timestamp_ns_lo;
	__u32 timestamp_ns_hi;
	__u32 reserved[4];
};

struct visiong_hw_irq_request {
	__u32 size;
	__u32 bank;
	__u32 pin;
	__u32 edge;
	__u32 flags;
	__u32 handle;
	__u32 reserved[4];
};

struct visiong_hw_spi_display_open {
	__u32 size;
	__u32 bus;
	__u32 chip_select;
	__u32 width;
	__u32 height;
	__u32 rotation;
	__u32 speed_hz;
	__u32 flags;
	__u32 handle;
	__u32 reserved[7];
};

struct visiong_hw_spi_display_submit {
	__u32 size;
	__u32 handle;
	__s32 dmabuf_fd;
	__u32 offset;
	__u32 bytes;
	__u32 stride;
	__u32 x;
	__u32 y;
	__u32 width;
	__u32 height;
	__u32 format;
	__u32 flags;
	__u32 job_handle;
	__u32 reserved[3];
};

struct visiong_hw_spi_reg_transfer {
	__u32 size;
	__u32 bus;
	__u32 chip_select;
	__u32 speed_hz;
	__u32 source_clock_hz;
	__u32 mode;
	__u32 bits_per_word;
	__u32 flags;
	__u64 tx_ptr;
	__u64 rx_ptr;
	__u32 tx_len;
	__u32 rx_len;
	__u32 status;
	__u32 transferred;
	__u32 dummy;
	__u32 reserved[3];
};

struct visiong_hw_spi_reg_release {
	__u32 size;
	__u32 bus;
	__u32 flags;
	__u32 reserved[5];
};

#define VISIONG_HW_IOC_MAGIC 'V'

#define VISIONG_HW_GET_CAPS \
	_IOR(VISIONG_HW_IOC_MAGIC, 0x00, struct visiong_hw_caps)

#define VISIONG_HW_REG_READ \
	_IOWR(VISIONG_HW_IOC_MAGIC, 0x10, struct visiong_hw_reg_access)
#define VISIONG_HW_REG_WRITE \
	_IOW(VISIONG_HW_IOC_MAGIC, 0x11, struct visiong_hw_reg_access)

#define VISIONG_HW_DMA_ALLOC \
	_IOWR(VISIONG_HW_IOC_MAGIC, 0x20, struct visiong_hw_dma_alloc)
#define VISIONG_HW_DMA_SYNC \
	_IOW(VISIONG_HW_IOC_MAGIC, 0x21, struct visiong_hw_dma_sync)
#define VISIONG_HW_DMA_MEMCPY \
	_IOWR(VISIONG_HW_IOC_MAGIC, 0x22, struct visiong_hw_dma_memcpy)
#define VISIONG_HW_DMA_WAIT \
	_IOWR(VISIONG_HW_IOC_MAGIC, 0x23, struct visiong_hw_wait)
#define VISIONG_HW_DMA_FILL \
	_IOWR(VISIONG_HW_IOC_MAGIC, 0x24, struct visiong_hw_dma_fill)

#define VISIONG_HW_IRQ_REQUEST \
	_IOWR(VISIONG_HW_IOC_MAGIC, 0x30, struct visiong_hw_irq_request)
#define VISIONG_HW_IRQ_WAIT \
	_IOWR(VISIONG_HW_IOC_MAGIC, 0x31, struct visiong_hw_wait)
#define VISIONG_HW_IRQ_RELEASE \
	_IOW(VISIONG_HW_IOC_MAGIC, 0x32, struct visiong_hw_wait)

#define VISIONG_HW_SPI_DISPLAY_OPEN \
	_IOWR(VISIONG_HW_IOC_MAGIC, 0x40, struct visiong_hw_spi_display_open)
#define VISIONG_HW_SPI_DISPLAY_SUBMIT \
	_IOWR(VISIONG_HW_IOC_MAGIC, 0x41, struct visiong_hw_spi_display_submit)
#define VISIONG_HW_SPI_DISPLAY_WAIT \
	_IOWR(VISIONG_HW_IOC_MAGIC, 0x42, struct visiong_hw_wait)
#define VISIONG_HW_SPI_DISPLAY_CLOSE \
	_IOW(VISIONG_HW_IOC_MAGIC, 0x43, struct visiong_hw_wait)
#define VISIONG_HW_SPI_REG_TRANSFER \
	_IOWR(VISIONG_HW_IOC_MAGIC, 0x44, struct visiong_hw_spi_reg_transfer)
#define VISIONG_HW_SPI_REG_RELEASE \
	_IOW(VISIONG_HW_IOC_MAGIC, 0x45, struct visiong_hw_spi_reg_release)

#endif /* VISIONG_UAPI_VISIONG_HW_H */
