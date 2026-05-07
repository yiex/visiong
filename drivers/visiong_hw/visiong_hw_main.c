// SPDX-License-Identifier: GPL-2.0-only
#include <linux/capability.h>
#include <linux/completion.h>
#include <linux/dma-buf.h>
#include <linux/dmaengine.h>
#include <linux/dma-mapping.h>
#include <linux/errno.h>
#include <linux/fs.h>
#include <linux/gpio.h>
#include <linux/io.h>
#include <linux/interrupt.h>
#include <linux/kernel.h>
#include <linux/ktime.h>
#include <linux/miscdevice.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/poll.h>
#include <linux/platform_device.h>
#include <linux/scatterlist.h>
#include <linux/sched.h>
#include <linux/slab.h>
#include <linux/spi/spi.h>
#include <linux/string.h>
#include <linux/syscalls.h>
#include <linux/uaccess.h>
#include <linux/wait.h>

#include <visiong/uapi/visiong_hw.h>

#define VISIONG_HW_DEVICE_NAME "visiong-hw"
#define VISIONG_HW_MAX_TRANSFER_BYTES (1024U * 1024U)
#define VISIONG_HW_MAX_DMA_BYTES (8U * 1024U * 1024U)
#define VISIONG_HW_MAX_IRQS_PER_SESSION 16U
#define VISIONG_HW_MAX_DMA_JOBS_PER_SESSION 8U
#define VISIONG_HW_MAX_SPI_DISPLAYS_PER_SESSION 4U
#define VISIONG_HW_MAX_SPI_JOBS_PER_DISPLAY 4U
#define VISIONG_HW_MAX_SPI_TRANSFER_BYTES (256U * 1024U)
#define VISIONG_HW_SPI_REG_DMA_MIN_BYTES 256U

struct visiong_hw_reg_region {
	const char *name;
	u32 block;
	phys_addr_t base;
	u32 size;
	bool writable;
	void __iomem *map;
};

struct visiong_hw_irq_slot {
	bool active;
	u32 handle;
	u32 bank;
	u32 pin;
	unsigned int gpio;
	unsigned int irq;
	u32 seq;
	u64 timestamp_ns;
	spinlock_t lock;
	wait_queue_head_t waitq;
};

struct visiong_hw_dma_map_ctx {
	struct dma_buf *dmabuf;
	struct dma_buf_attachment *attach;
	struct sg_table *sgt;
	dma_addr_t addr;
	size_t size;
};

struct visiong_hw_dma_job {
	bool active;
	bool submitted;
	bool waiting;
	u32 handle;
	u32 status;
	u64 timestamp_ns;
	dma_cookie_t cookie;
	struct completion done;
	struct visiong_hw_dma_map_ctx dst;
	struct visiong_hw_dma_map_ctx src;
};

struct visiong_hw_spi_job {
	bool active;
	bool submitted;
	bool waiting;
	bool completed;
	u32 handle;
	u32 status;
	u64 timestamp_ns;
	struct completion done;
	struct spi_message message;
	struct spi_transfer transfer;
	u8 *tx_buf;
	u32 bytes;
};

struct visiong_hw_spi_slot {
	bool active;
	u32 handle;
	u32 bus;
	u32 chip_select;
	u32 speed_hz;
	u32 flags;
	struct spi_device *spi;
	struct visiong_hw_spi_job *jobs[VISIONG_HW_MAX_SPI_JOBS_PER_DISPLAY];
};

struct visiong_hw_spi_reg_dma_bus {
	struct mutex lock;
	struct dma_chan *tx_chan;
	struct device *platform_dev;
	bool unavailable;
};

struct visiong_hw_session {
	u32 id;
	struct mutex lock;
	struct visiong_hw_irq_slot irqs[VISIONG_HW_MAX_IRQS_PER_SESSION];
	struct visiong_hw_dma_job dma_jobs[VISIONG_HW_MAX_DMA_JOBS_PER_SESSION];
	struct visiong_hw_spi_slot spi_slots[VISIONG_HW_MAX_SPI_DISPLAYS_PER_SESSION];
};

static DEFINE_MUTEX(visiong_hw_reg_lock);
static DEFINE_MUTEX(visiong_hw_dma_chan_lock);
static DEFINE_MUTEX(visiong_hw_spi_reg_lock);
static struct device *visiong_hw_device;
static struct dma_chan *visiong_hw_memcpy_chan;
static struct visiong_hw_spi_reg_dma_bus visiong_hw_spi_reg_dma[2];

static int visiong_hw_spi_probe(struct spi_device *spi)
{
	dev_info(&spi->dev, "bound to visiong_hw_spi\n");
	return 0;
}

static int visiong_hw_spi_remove(struct spi_device *spi)
{
	dev_info(&spi->dev, "unbound from visiong_hw_spi\n");
	return 0;
}

static struct spi_driver visiong_hw_spi_driver = {
	.driver = {
		.name = "visiong_hw_spi",
		.owner = THIS_MODULE,
	},
	.probe = visiong_hw_spi_probe,
	.remove = visiong_hw_spi_remove,
};

struct visiong_hw_dma_buffer {
	struct device *dev;
	size_t size;
	void *cpu_addr;
	dma_addr_t dma_addr;
	unsigned long attrs;
};

static struct visiong_hw_reg_region visiong_hw_regions[] = {
	{ "ioc", VISIONG_HW_REG_BLOCK_IOC, 0xff538000, 0x40000, true },
	{ "pmuioc", VISIONG_HW_REG_BLOCK_PMUIOC, 0xff388000, 0x1000, true },
	{ "cru", VISIONG_HW_REG_BLOCK_CRU, 0xff3a0000, 0x20000, true },
	{ "gpio0", VISIONG_HW_REG_BLOCK_GPIO0, 0xff380000, 0x1000, true },
	{ "gpio1", VISIONG_HW_REG_BLOCK_GPIO1, 0xff530000, 0x1000, true },
	{ "gpio2", VISIONG_HW_REG_BLOCK_GPIO2, 0xff540000, 0x1000, true },
	{ "gpio3", VISIONG_HW_REG_BLOCK_GPIO3, 0xff550000, 0x1000, true },
	{ "gpio4", VISIONG_HW_REG_BLOCK_GPIO4, 0xff560000, 0x1000, true },
	{ "spi0", VISIONG_HW_REG_BLOCK_SPI0, 0xff500000, 0x1000, true },
	{ "spi1", VISIONG_HW_REG_BLOCK_SPI1, 0xff510000, 0x1000, true },
	{ "i2c0", VISIONG_HW_REG_BLOCK_I2C0, 0xff310000, 0x1000, true },
	{ "i2c1", VISIONG_HW_REG_BLOCK_I2C1, 0xff320000, 0x1000, true },
	{ "i2c2", VISIONG_HW_REG_BLOCK_I2C2, 0xff450000, 0x1000, true },
	{ "i2c3", VISIONG_HW_REG_BLOCK_I2C3, 0xff460000, 0x1000, true },
	{ "i2c4", VISIONG_HW_REG_BLOCK_I2C4, 0xff470000, 0x1000, true },
	{ "uart0", VISIONG_HW_REG_BLOCK_UART0, 0xff4a0000, 0x100, true },
	{ "uart1", VISIONG_HW_REG_BLOCK_UART1, 0xff4b0000, 0x100, true },
	{ "uart2", VISIONG_HW_REG_BLOCK_UART2, 0xff4c0000, 0x100, true },
	{ "uart3", VISIONG_HW_REG_BLOCK_UART3, 0xff4d0000, 0x100, true },
	{ "uart4", VISIONG_HW_REG_BLOCK_UART4, 0xff4e0000, 0x100, true },
	{ "uart5", VISIONG_HW_REG_BLOCK_UART5, 0xff4f0000, 0x100, true },
	{ "pwm0_3", VISIONG_HW_REG_BLOCK_PWM0_3, 0xff350000, 0x1000, true },
	{ "pwm4_7", VISIONG_HW_REG_BLOCK_PWM4_7, 0xff360000, 0x1000, true },
	{ "pwm8_11", VISIONG_HW_REG_BLOCK_PWM8_11, 0xff490000, 0x1000, true },
	{ "dmac", VISIONG_HW_REG_BLOCK_DMAC, 0xff420000, 0x4000, false },
	{ "gicd", VISIONG_HW_REG_BLOCK_GICD, 0xff1f1000, 0x1000, false },
};

static struct visiong_hw_reg_region *visiong_hw_find_region(u32 block)
{
	unsigned int i;

	for (i = 0; i < ARRAY_SIZE(visiong_hw_regions); ++i) {
		if (visiong_hw_regions[i].block == block)
			return &visiong_hw_regions[i];
	}
	return NULL;
}

static enum dma_data_direction visiong_hw_dma_direction(u32 direction)
{
	switch (direction) {
	case VISIONG_HW_DMA_SYNC_TO_DEVICE:
		return DMA_TO_DEVICE;
	case VISIONG_HW_DMA_SYNC_FROM_DEVICE:
		return DMA_FROM_DEVICE;
	case VISIONG_HW_DMA_SYNC_BIDIRECTIONAL:
	default:
		return DMA_BIDIRECTIONAL;
	}
}

static int visiong_hw_dma_ensure_memcpy_chan(void)
{
	dma_cap_mask_t mask;
	int ret = 0;

	mutex_lock(&visiong_hw_dma_chan_lock);
	if (visiong_hw_memcpy_chan)
		goto out;

	dma_cap_zero(mask);
	dma_cap_set(DMA_MEMCPY, mask);
	visiong_hw_memcpy_chan = dma_request_chan_by_mask(&mask);
	if (IS_ERR(visiong_hw_memcpy_chan)) {
		ret = PTR_ERR(visiong_hw_memcpy_chan);
		visiong_hw_memcpy_chan = NULL;
	}

out:
	mutex_unlock(&visiong_hw_dma_chan_lock);
	return ret;
}

static void visiong_hw_dma_complete_func(void *param)
{
	complete(param);
}

static void visiong_hw_dma_job_complete_func(void *param)
{
	struct visiong_hw_dma_job *job = param;

	job->status = VISIONG_HW_DMA_MEMCPY_STATUS_DONE;
	job->timestamp_ns = ktime_get_ns();
	complete(&job->done);
}

static void visiong_hw_dma_unmap_fd(struct visiong_hw_dma_map_ctx *ctx,
				    enum dma_data_direction direction)
{
	if (!ctx)
		return;
	if (ctx->attach && ctx->sgt)
		dma_buf_unmap_attachment(ctx->attach, ctx->sgt, direction);
	if (ctx->dmabuf && ctx->attach)
		dma_buf_detach(ctx->dmabuf, ctx->attach);
	if (ctx->dmabuf)
		dma_buf_put(ctx->dmabuf);
	memset(ctx, 0, sizeof(*ctx));
}

static int visiong_hw_dma_map_fd(int fd,
				 u32 offset,
				 u32 bytes,
				 enum dma_data_direction direction,
				 struct visiong_hw_dma_map_ctx *ctx)
{
	struct scatterlist *sg;
	u64 end = (u64)offset + (u64)bytes;

	memset(ctx, 0, sizeof(*ctx));
	if (fd < 0 || !bytes || !visiong_hw_device)
		return -EINVAL;

	ctx->dmabuf = dma_buf_get(fd);
	if (IS_ERR(ctx->dmabuf)) {
		int ret = PTR_ERR(ctx->dmabuf);

		ctx->dmabuf = NULL;
		return ret;
	}
	if (end > ctx->dmabuf->size) {
		visiong_hw_dma_unmap_fd(ctx, direction);
		return -EINVAL;
	}

	ctx->attach = dma_buf_attach(ctx->dmabuf, visiong_hw_device);
	if (IS_ERR(ctx->attach)) {
		int ret = PTR_ERR(ctx->attach);

		ctx->attach = NULL;
		visiong_hw_dma_unmap_fd(ctx, direction);
		return ret;
	}

	ctx->sgt = dma_buf_map_attachment(ctx->attach, direction);
	if (IS_ERR(ctx->sgt)) {
		int ret = PTR_ERR(ctx->sgt);

		ctx->sgt = NULL;
		visiong_hw_dma_unmap_fd(ctx, direction);
		return ret;
	}

	if (!ctx->sgt->sgl || ctx->sgt->nents != 1) {
		visiong_hw_dma_unmap_fd(ctx, direction);
		return -EOPNOTSUPP;
	}

	sg = ctx->sgt->sgl;
	if (offset > sg_dma_len(sg) || bytes > sg_dma_len(sg) - offset) {
		visiong_hw_dma_unmap_fd(ctx, direction);
		return -EINVAL;
	}

	ctx->addr = sg_dma_address(sg) + offset;
	ctx->size = bytes;
	return 0;
}

static struct sg_table *visiong_hw_dma_map(struct dma_buf_attachment *attach,
					   enum dma_data_direction direction)
{
	struct visiong_hw_dma_buffer *buffer = attach->dmabuf->priv;
	struct sg_table *sgt;
	int ret;

	sgt = kzalloc(sizeof(*sgt), GFP_KERNEL);
	if (!sgt)
		return ERR_PTR(-ENOMEM);

	ret = sg_alloc_table(sgt, 1, GFP_KERNEL);
	if (ret) {
		kfree(sgt);
		return ERR_PTR(ret);
	}

	sg_set_page(sgt->sgl, virt_to_page(buffer->cpu_addr), buffer->size, 0);
	sg_dma_address(sgt->sgl) = buffer->dma_addr;
	sg_dma_len(sgt->sgl) = buffer->size;

	return sgt;
}

static void visiong_hw_dma_unmap(struct dma_buf_attachment *attach,
				 struct sg_table *sgt,
				 enum dma_data_direction direction)
{
	if (!sgt)
		return;
	sg_free_table(sgt);
	kfree(sgt);
}

static int visiong_hw_dma_mmap(struct dma_buf *dmabuf, struct vm_area_struct *vma)
{
	struct visiong_hw_dma_buffer *buffer = dmabuf->priv;

	return dma_mmap_attrs(buffer->dev, vma, buffer->cpu_addr,
			      buffer->dma_addr, buffer->size, buffer->attrs);
}

static void *visiong_hw_dma_vmap(struct dma_buf *dmabuf)
{
	struct visiong_hw_dma_buffer *buffer = dmabuf->priv;

	return buffer->cpu_addr;
}

static void visiong_hw_dma_vunmap(struct dma_buf *dmabuf, void *vaddr)
{
}

static int visiong_hw_dma_begin_cpu_access(struct dma_buf *dmabuf,
					   enum dma_data_direction direction)
{
	return 0;
}

static int visiong_hw_dma_end_cpu_access(struct dma_buf *dmabuf,
					 enum dma_data_direction direction)
{
	return 0;
}

static void visiong_hw_dma_release(struct dma_buf *dmabuf)
{
	struct visiong_hw_dma_buffer *buffer = dmabuf->priv;

	if (!buffer)
		return;

	if (buffer->cpu_addr)
		dma_free_attrs(buffer->dev, buffer->size, buffer->cpu_addr,
			       buffer->dma_addr, buffer->attrs);
	kfree(buffer);
}

static const struct dma_buf_ops visiong_hw_dma_buf_ops = {
	.map_dma_buf = visiong_hw_dma_map,
	.unmap_dma_buf = visiong_hw_dma_unmap,
	.mmap = visiong_hw_dma_mmap,
	.vmap = visiong_hw_dma_vmap,
	.vunmap = visiong_hw_dma_vunmap,
	.begin_cpu_access = visiong_hw_dma_begin_cpu_access,
	.end_cpu_access = visiong_hw_dma_end_cpu_access,
	.release = visiong_hw_dma_release,
};

static int visiong_hw_check_access(const struct visiong_hw_reg_region *region,
				   const struct visiong_hw_reg_access *access,
				   bool write)
{
	if (!region || !region->map)
		return -ENODEV;
	if (access->size < sizeof(*access))
		return -EINVAL;
	if (access->offset & 0x3)
		return -EINVAL;
	if (access->offset > region->size || region->size - access->offset < sizeof(u32))
		return -EINVAL;
	if (write && !region->writable)
		return -EPERM;
	if (write && access->mask == 0)
		return -EINVAL;
	if ((access->flags & ~VISIONG_HW_REG_FLAG_HIWORD_UPDATE) != 0)
		return -EINVAL;
	if ((access->flags & VISIONG_HW_REG_FLAG_HIWORD_UPDATE) && access->mask > 0xffff)
		return -EINVAL;
	return 0;
}

static int visiong_hw_get_caps(unsigned long arg)
{
	struct visiong_hw_caps caps;

	memset(&caps, 0, sizeof(caps));
	caps.size = sizeof(caps);
	caps.abi_version = VISIONG_HW_ABI_VERSION;
	caps.driver_version = VISIONG_HW_DRIVER_VERSION;
	caps.feature_flags = VISIONG_HW_FEATURE_REG_ACCESS |
			     VISIONG_HW_FEATURE_PIN_SESSION |
			     VISIONG_HW_FEATURE_GPIO_IRQ |
			     VISIONG_HW_FEATURE_DMA_BUFFER |
			     VISIONG_HW_FEATURE_DMA_FILL |
			     VISIONG_HW_FEATURE_SPI_DISPLAY |
			     VISIONG_HW_FEATURE_SPI_REG;
	if (visiong_hw_dma_ensure_memcpy_chan() == 0)
		caps.feature_flags |= VISIONG_HW_FEATURE_DMA_MEMCPY;
	caps.chip_id = 0x1106;
	caps.max_dma_bytes = VISIONG_HW_MAX_DMA_BYTES;
	caps.max_transfer_bytes = VISIONG_HW_MAX_TRANSFER_BYTES;

	if (copy_to_user((void __user *)arg, &caps, sizeof(caps)))
		return -EFAULT;
	return 0;
}

static int visiong_hw_reg_read(unsigned long arg)
{
	struct visiong_hw_reg_access access;
	struct visiong_hw_reg_region *region;
	int ret;

	if (copy_from_user(&access, (void __user *)arg, sizeof(access)))
		return -EFAULT;

	region = visiong_hw_find_region(access.block);
	ret = visiong_hw_check_access(region, &access, false);
	if (ret)
		return ret;

	access.value = readl(region->map + access.offset);
	access.size = sizeof(access);

	if (copy_to_user((void __user *)arg, &access, sizeof(access)))
		return -EFAULT;
	return 0;
}

static int visiong_hw_reg_write(unsigned long arg)
{
	struct visiong_hw_reg_access access;
	struct visiong_hw_reg_region *region;
	u32 old_value;
	u32 new_value;
	int ret;

	if (!capable(CAP_SYS_RAWIO))
		return -EPERM;

	if (copy_from_user(&access, (void __user *)arg, sizeof(access)))
		return -EFAULT;

	region = visiong_hw_find_region(access.block);
	ret = visiong_hw_check_access(region, &access, true);
	if (ret)
		return ret;

	mutex_lock(&visiong_hw_reg_lock);
	if (access.flags & VISIONG_HW_REG_FLAG_HIWORD_UPDATE) {
		writel((access.mask << 16) | (access.value & access.mask),
		       region->map + access.offset);
	} else {
		old_value = readl(region->map + access.offset);
		new_value = (old_value & ~access.mask) | (access.value & access.mask);
		writel(new_value, region->map + access.offset);
	}
	mutex_unlock(&visiong_hw_reg_lock);

	return 0;
}

static int visiong_hw_dma_alloc(unsigned long arg)
{
	struct visiong_hw_dma_alloc request;
	struct visiong_hw_dma_buffer *buffer;
	DEFINE_DMA_BUF_EXPORT_INFO(exp_info);
	struct dma_buf *dmabuf;
	size_t size;
	int fd;

	if (copy_from_user(&request, (void __user *)arg, sizeof(request)))
		return -EFAULT;
	if (request.size < sizeof(request))
		return -EINVAL;
	if (!request.bytes || request.bytes > VISIONG_HW_MAX_DMA_BYTES)
		return -EINVAL;
	if (request.flags & ~VISIONG_HW_DMA_ALLOC_WRITE_COMBINE)
		return -EINVAL;
	if (!visiong_hw_device)
		return -ENODEV;

	size = PAGE_ALIGN(request.bytes);
	buffer = kzalloc(sizeof(*buffer), GFP_KERNEL);
	if (!buffer)
		return -ENOMEM;

	buffer->dev = visiong_hw_device;
	buffer->size = size;
	if (request.flags & VISIONG_HW_DMA_ALLOC_WRITE_COMBINE)
		buffer->attrs |= DMA_ATTR_WRITE_COMBINE;
	buffer->attrs |= DMA_ATTR_FORCE_CONTIGUOUS;

	buffer->cpu_addr = dma_alloc_attrs(buffer->dev, buffer->size,
					   &buffer->dma_addr, GFP_KERNEL | __GFP_ZERO,
					   buffer->attrs);
	if (!buffer->cpu_addr) {
		kfree(buffer);
		return -ENOMEM;
	}

	exp_info.ops = &visiong_hw_dma_buf_ops;
	exp_info.size = buffer->size;
	exp_info.flags = O_RDWR;
	exp_info.priv = buffer;

	dmabuf = dma_buf_export(&exp_info);
	if (IS_ERR(dmabuf)) {
		dma_free_attrs(buffer->dev, buffer->size, buffer->cpu_addr,
			       buffer->dma_addr, buffer->attrs);
		kfree(buffer);
		return PTR_ERR(dmabuf);
	}

	fd = dma_buf_fd(dmabuf, O_CLOEXEC | O_RDWR);
	if (fd < 0) {
		dma_buf_put(dmabuf);
		return fd;
	}

	request.size = sizeof(request);
	request.fd = fd;
	request.bytes = buffer->size;

	if (copy_to_user((void __user *)arg, &request, sizeof(request))) {
		ksys_close(fd);
		return -EFAULT;
	}

	return 0;
}

static int visiong_hw_dma_sync(unsigned long arg)
{
	struct visiong_hw_dma_sync request;
	struct dma_buf *dmabuf;
	enum dma_data_direction direction;
	int ret = 0;

	if (copy_from_user(&request, (void __user *)arg, sizeof(request)))
		return -EFAULT;
	if (request.size < sizeof(request))
		return -EINVAL;
	if (request.fd < 0)
		return -EINVAL;
	if (request.flags & ~(VISIONG_HW_DMA_SYNC_START | VISIONG_HW_DMA_SYNC_END))
		return -EINVAL;
	if (!(request.flags & (VISIONG_HW_DMA_SYNC_START | VISIONG_HW_DMA_SYNC_END)))
		return -EINVAL;
	if ((request.flags & VISIONG_HW_DMA_SYNC_START) &&
	    (request.flags & VISIONG_HW_DMA_SYNC_END))
		return -EINVAL;

	direction = visiong_hw_dma_direction(request.direction);
	dmabuf = dma_buf_get(request.fd);
	if (IS_ERR(dmabuf))
		return PTR_ERR(dmabuf);

	if (request.bytes) {
		u64 end = (u64)request.offset + (u64)request.bytes;

		if (end > dmabuf->size) {
			dma_buf_put(dmabuf);
			return -EINVAL;
		}
	}

	if (request.flags & VISIONG_HW_DMA_SYNC_START)
		ret = dma_buf_begin_cpu_access(dmabuf, direction);
	else
		ret = dma_buf_end_cpu_access(dmabuf, direction);

	dma_buf_put(dmabuf);
	return ret;
}

static int visiong_hw_dma_fill(unsigned long arg)
{
	struct visiong_hw_dma_fill request;
	struct dma_buf *dmabuf;
	void *vaddr;
	u64 end;
	int ret;
	int end_ret;

	if (copy_from_user(&request, (void __user *)arg, sizeof(request)))
		return -EFAULT;
	if (request.size < sizeof(request))
		return -EINVAL;
	if (request.fd < 0 || !request.bytes ||
	    request.bytes > VISIONG_HW_MAX_TRANSFER_BYTES)
		return -EINVAL;
	if (request.flags)
		return -EINVAL;

	end = (u64)request.offset + (u64)request.bytes;
	dmabuf = dma_buf_get(request.fd);
	if (IS_ERR(dmabuf))
		return PTR_ERR(dmabuf);
	if (end > dmabuf->size) {
		ret = -EINVAL;
		goto out_put;
	}

	ret = dma_buf_begin_cpu_access(dmabuf, DMA_BIDIRECTIONAL);
	if (ret)
		goto out_put;

	vaddr = dma_buf_vmap(dmabuf);
	if (IS_ERR_OR_NULL(vaddr)) {
		ret = vaddr ? PTR_ERR(vaddr) : -ENOMEM;
		goto out_cpu_end;
	}

	memset((u8 *)vaddr + request.offset, request.value & 0xff,
	       request.bytes);
	dma_buf_vunmap(dmabuf, vaddr);

out_cpu_end:
	end_ret = dma_buf_end_cpu_access(dmabuf, DMA_BIDIRECTIONAL);
	if (!ret)
		ret = end_ret;
out_put:
	dma_buf_put(dmabuf);

	request.status = ret ? VISIONG_HW_DMA_MEMCPY_STATUS_ERROR :
			       VISIONG_HW_DMA_MEMCPY_STATUS_DONE;
	request.size = sizeof(request);
	if (copy_to_user((void __user *)arg, &request, sizeof(request)))
		return -EFAULT;
	return ret;
}

static int visiong_hw_dma_submit_memcpy(struct visiong_hw_dma_map_ctx *dst,
					struct visiong_hw_dma_map_ctx *src,
					u32 bytes,
					dma_async_tx_callback callback,
					void *callback_param,
					dma_cookie_t *cookie_out)
{
	struct dma_async_tx_descriptor *desc;
	dma_cookie_t cookie;
	int ret;

	mutex_lock(&visiong_hw_dma_chan_lock);
	desc = dmaengine_prep_dma_memcpy(visiong_hw_memcpy_chan,
					 dst->addr, src->addr, bytes,
					 DMA_CTRL_ACK | DMA_PREP_INTERRUPT);
	if (!desc) {
		mutex_unlock(&visiong_hw_dma_chan_lock);
		return -EOPNOTSUPP;
	}

	desc->callback = callback;
	desc->callback_param = callback_param;

	cookie = dmaengine_submit(desc);
	ret = dma_submit_error(cookie);
	if (ret) {
		mutex_unlock(&visiong_hw_dma_chan_lock);
		return ret;
	}

	dma_async_issue_pending(visiong_hw_memcpy_chan);
	mutex_unlock(&visiong_hw_dma_chan_lock);
	*cookie_out = cookie;
	return 0;
}

static void visiong_hw_dma_release_job(struct visiong_hw_dma_job *job)
{
	if (!job || !job->active)
		return;

	if (job->submitted && !completion_done(&job->done)) {
		mutex_lock(&visiong_hw_dma_chan_lock);
		if (visiong_hw_memcpy_chan)
			dmaengine_terminate_sync(visiong_hw_memcpy_chan);
		mutex_unlock(&visiong_hw_dma_chan_lock);
	}

	visiong_hw_dma_unmap_fd(&job->src, DMA_BIDIRECTIONAL);
	visiong_hw_dma_unmap_fd(&job->dst, DMA_BIDIRECTIONAL);
	memset(job, 0, sizeof(*job));
}

static void visiong_hw_release_dma_jobs(struct visiong_hw_session *session)
{
	unsigned int i;

	if (!session)
		return;

	mutex_lock(&session->lock);
	for (i = 0; i < VISIONG_HW_MAX_DMA_JOBS_PER_SESSION; ++i)
		visiong_hw_dma_release_job(&session->dma_jobs[i]);
	mutex_unlock(&session->lock);
}

static int visiong_hw_dma_memcpy(struct visiong_hw_session *session,
				 unsigned long arg)
{
	struct visiong_hw_dma_memcpy request;
	struct visiong_hw_dma_map_ctx dst;
	struct visiong_hw_dma_map_ctx src;
	struct visiong_hw_dma_job *job = NULL;
	struct completion done;
	dma_cookie_t cookie;
	unsigned long timeout;
	enum dma_status status;
	bool async;
	unsigned int i;
	int ret;

	if (copy_from_user(&request, (void __user *)arg, sizeof(request)))
		return -EFAULT;
	if (request.size < sizeof(request))
		return -EINVAL;
	if (!request.bytes || request.bytes > VISIONG_HW_MAX_TRANSFER_BYTES)
		return -EINVAL;
	if (request.flags & ~VISIONG_HW_DMA_MEMCPY_ASYNC)
		return -EINVAL;

	async = !!(request.flags & VISIONG_HW_DMA_MEMCPY_ASYNC);
	if (async && !session)
		return -EINVAL;

	ret = visiong_hw_dma_ensure_memcpy_chan();
	if (ret)
		return ret;

	if (async) {
		mutex_lock(&session->lock);
		for (i = 0; i < VISIONG_HW_MAX_DMA_JOBS_PER_SESSION; ++i) {
			if (!session->dma_jobs[i].active) {
				job = &session->dma_jobs[i];
				memset(job, 0, sizeof(*job));
				job->active = true;
				job->handle = i + 1;
				job->status = VISIONG_HW_DMA_MEMCPY_STATUS_PENDING;
				init_completion(&job->done);
				break;
			}
		}
		mutex_unlock(&session->lock);
		if (!job)
			return -EBUSY;

		ret = visiong_hw_dma_map_fd(request.dst_fd, request.dst_offset,
					    request.bytes, DMA_BIDIRECTIONAL,
					    &job->dst);
		if (ret)
			goto out_job;

		ret = visiong_hw_dma_map_fd(request.src_fd, request.src_offset,
					    request.bytes, DMA_BIDIRECTIONAL,
					    &job->src);
		if (ret)
			goto out_job;

		ret = visiong_hw_dma_submit_memcpy(&job->dst, &job->src,
						   request.bytes,
						   visiong_hw_dma_job_complete_func,
						   job, &job->cookie);
		if (ret)
			goto out_job;

		job->submitted = true;
		request.handle = job->handle;
		request.status = VISIONG_HW_DMA_MEMCPY_STATUS_PENDING;
		request.size = sizeof(request);
		if (copy_to_user((void __user *)arg, &request, sizeof(request))) {
			mutex_lock(&session->lock);
			visiong_hw_dma_release_job(job);
			mutex_unlock(&session->lock);
			return -EFAULT;
		}
		return 0;

out_job:
		visiong_hw_dma_release_job(job);
		return ret;
	}

	ret = visiong_hw_dma_map_fd(request.dst_fd, request.dst_offset,
				    request.bytes, DMA_BIDIRECTIONAL, &dst);
	if (ret)
		return ret;

	ret = visiong_hw_dma_map_fd(request.src_fd, request.src_offset,
				    request.bytes, DMA_BIDIRECTIONAL, &src);
	if (ret)
		goto out_dst;

	init_completion(&done);
	ret = visiong_hw_dma_submit_memcpy(&dst, &src, request.bytes,
					   visiong_hw_dma_complete_func, &done,
					   &cookie);
	if (ret)
		goto out_src;

	timeout = wait_for_completion_timeout(&done, msecs_to_jiffies(5000));
	if (!timeout) {
		request.status = VISIONG_HW_DMA_MEMCPY_STATUS_TIMEOUT;
		ret = -ETIMEDOUT;
		goto out_copy;
	}

	status = dma_async_is_tx_complete(visiong_hw_memcpy_chan, cookie, NULL, NULL);
	if (status != DMA_COMPLETE) {
		request.status = VISIONG_HW_DMA_MEMCPY_STATUS_ERROR;
		ret = -EIO;
		goto out_copy;
	}

	request.status = VISIONG_HW_DMA_MEMCPY_STATUS_DONE;
	ret = 0;

out_copy:
	request.handle = 0;
	request.size = sizeof(request);
	if (copy_to_user((void __user *)arg, &request, sizeof(request)))
		ret = -EFAULT;
out_src:
	visiong_hw_dma_unmap_fd(&src, DMA_BIDIRECTIONAL);
out_dst:
	visiong_hw_dma_unmap_fd(&dst, DMA_BIDIRECTIONAL);
	return ret;
}

static int visiong_hw_dma_wait(struct visiong_hw_session *session,
			       unsigned long arg)
{
	struct visiong_hw_wait request;
	struct visiong_hw_dma_job *job;
	long timeout;
	u32 status;
	u64 timestamp;
	int ret = 0;

	if (!session)
		return -EINVAL;
	if (copy_from_user(&request, (void __user *)arg, sizeof(request)))
		return -EFAULT;
	if (request.size < sizeof(request))
		return -EINVAL;
	if (!request.handle ||
	    request.handle > VISIONG_HW_MAX_DMA_JOBS_PER_SESSION)
		return -EINVAL;

	mutex_lock(&session->lock);
	job = &session->dma_jobs[request.handle - 1];
	if (!job->active || job->handle != request.handle) {
		mutex_unlock(&session->lock);
		return -ENOENT;
	}
	if (job->waiting) {
		mutex_unlock(&session->lock);
		return -EBUSY;
	}
	job->waiting = true;
	mutex_unlock(&session->lock);

	if (request.timeout_ms < 0) {
		ret = wait_for_completion_interruptible(&job->done);
		if (ret) {
			mutex_lock(&session->lock);
			if (job->active && job->handle == request.handle)
				job->waiting = false;
			mutex_unlock(&session->lock);
			return ret;
		}
	} else {
		timeout = wait_for_completion_interruptible_timeout(
			&job->done, msecs_to_jiffies(request.timeout_ms));
		if (timeout < 0) {
			mutex_lock(&session->lock);
			if (job->active && job->handle == request.handle)
				job->waiting = false;
			mutex_unlock(&session->lock);
			return timeout;
		}
		if (timeout == 0) {
			request.status = VISIONG_HW_DMA_MEMCPY_STATUS_TIMEOUT;
			request.timestamp_ns_lo = 0;
			request.timestamp_ns_hi = 0;
			request.size = sizeof(request);
			mutex_lock(&session->lock);
			if (job->active && job->handle == request.handle)
				job->waiting = false;
			mutex_unlock(&session->lock);
			if (copy_to_user((void __user *)arg, &request,
					 sizeof(request)))
				return -EFAULT;
			return 0;
		}
	}

	status = job->status;
	if (dma_async_is_tx_complete(visiong_hw_memcpy_chan, job->cookie,
				     NULL, NULL) != DMA_COMPLETE)
		status = VISIONG_HW_DMA_MEMCPY_STATUS_ERROR;
	timestamp = job->timestamp_ns;

	request.status = status;
	request.timestamp_ns_lo = lower_32_bits(timestamp);
	request.timestamp_ns_hi = upper_32_bits(timestamp);
	request.size = sizeof(request);
	if (copy_to_user((void __user *)arg, &request, sizeof(request)))
		ret = -EFAULT;

	mutex_lock(&session->lock);
	visiong_hw_dma_release_job(job);
	mutex_unlock(&session->lock);
	return ret;
}

static u32 visiong_hw_spi_make_job_handle(u32 slot_handle, u32 job_index)
{
	return (slot_handle << 16) | ((job_index + 1) & 0xffff);
}

static void visiong_hw_spi_decode_job_handle(u32 handle,
					     u32 *slot_handle,
					     u32 *job_index)
{
	*slot_handle = handle >> 16;
	*job_index = (handle & 0xffff) - 1;
}

static void visiong_hw_spi_complete_func(void *context)
{
	struct visiong_hw_spi_job *job = context;

	job->status = job->message.status ? VISIONG_HW_SPI_STATUS_ERROR :
					    VISIONG_HW_SPI_STATUS_DONE;
	job->timestamp_ns = ktime_get_ns();
	job->completed = true;
	complete(&job->done);
}

static void visiong_hw_spi_release_job(struct visiong_hw_spi_job *job)
{
	if (!job || !job->active)
		return;

	if (job->submitted && !job->completed &&
	    !wait_for_completion_timeout(&job->done, msecs_to_jiffies(5000))) {
		pr_warn("visiong_hw: SPI job 0x%x did not complete before release; leaking job to avoid UAF\n",
			job->handle);
		return;
	}
	kfree(job->tx_buf);
	kfree(job);
}

static void visiong_hw_spi_release_slot(struct visiong_hw_spi_slot *slot)
{
	unsigned int i;

	if (!slot || !slot->active)
		return;

	for (i = 0; i < VISIONG_HW_MAX_SPI_JOBS_PER_DISPLAY; ++i) {
		visiong_hw_spi_release_job(slot->jobs[i]);
		slot->jobs[i] = NULL;
	}
	if (slot->spi)
		put_device(&slot->spi->dev);
	memset(slot, 0, sizeof(*slot));
}

static void visiong_hw_release_spi_slots(struct visiong_hw_session *session)
{
	unsigned int i;

	if (!session)
		return;

	mutex_lock(&session->lock);
	for (i = 0; i < VISIONG_HW_MAX_SPI_DISPLAYS_PER_SESSION; ++i)
		visiong_hw_spi_release_slot(&session->spi_slots[i]);
	mutex_unlock(&session->lock);
}

static struct visiong_hw_spi_slot *
visiong_hw_spi_find_slot(struct visiong_hw_session *session, u32 handle)
{
	if (!session || !handle ||
	    handle > VISIONG_HW_MAX_SPI_DISPLAYS_PER_SESSION)
		return NULL;
	if (!session->spi_slots[handle - 1].active ||
	    session->spi_slots[handle - 1].handle != handle)
		return NULL;
	return &session->spi_slots[handle - 1];
}

static int visiong_hw_spi_display_open(struct visiong_hw_session *session,
				       unsigned long arg)
{
	struct visiong_hw_spi_display_open request;
	struct visiong_hw_spi_slot *slot = NULL;
	struct device *dev;
	char name[24];
	unsigned int i;

	if (!session)
		return -EINVAL;
	if (copy_from_user(&request, (void __user *)arg, sizeof(request)))
		return -EFAULT;
	if (request.size < sizeof(request))
		return -EINVAL;
	if (request.bus > 16 || request.chip_select > 8)
		return -EINVAL;

	snprintf(name, sizeof(name), "spi%u.%u", request.bus,
		 request.chip_select);
	dev = bus_find_device_by_name(&spi_bus_type, NULL, name);
	if (!dev)
		return -ENODEV;
	if (dev->driver != &visiong_hw_spi_driver.driver) {
		put_device(dev);
		return -EBUSY;
	}

	mutex_lock(&session->lock);
	for (i = 0; i < VISIONG_HW_MAX_SPI_DISPLAYS_PER_SESSION; ++i) {
		if (!session->spi_slots[i].active) {
			slot = &session->spi_slots[i];
			memset(slot, 0, sizeof(*slot));
			slot->active = true;
			slot->handle = i + 1;
			slot->bus = request.bus;
			slot->chip_select = request.chip_select;
			slot->speed_hz = request.speed_hz ? request.speed_hz :
						 24000000U;
			slot->flags = request.flags;
			slot->spi = to_spi_device(dev);
			request.handle = slot->handle;
			break;
		}
	}
	mutex_unlock(&session->lock);

	if (!slot) {
		put_device(dev);
		return -EBUSY;
	}

	request.size = sizeof(request);
	if (copy_to_user((void __user *)arg, &request, sizeof(request))) {
		mutex_lock(&session->lock);
		visiong_hw_spi_release_slot(slot);
		mutex_unlock(&session->lock);
		return -EFAULT;
	}
	return 0;
}

static int visiong_hw_spi_copy_dmabuf(struct visiong_hw_spi_job *job,
				      s32 fd,
				      u32 offset,
				      u32 bytes)
{
	struct dma_buf *dmabuf;
	void *vaddr;
	u64 end = (u64)offset + (u64)bytes;
	int ret;
	int end_ret;

	job->tx_buf = kmalloc(bytes, GFP_KERNEL);
	if (!job->tx_buf)
		return -ENOMEM;

	dmabuf = dma_buf_get(fd);
	if (IS_ERR(dmabuf))
		return PTR_ERR(dmabuf);
	if (end > dmabuf->size) {
		ret = -EINVAL;
		goto out_put;
	}

	ret = dma_buf_begin_cpu_access(dmabuf, DMA_BIDIRECTIONAL);
	if (ret)
		goto out_put;

	vaddr = dma_buf_vmap(dmabuf);
	if (IS_ERR_OR_NULL(vaddr)) {
		ret = vaddr ? PTR_ERR(vaddr) : -ENOMEM;
		goto out_cpu_end;
	}

	memcpy(job->tx_buf, (u8 *)vaddr + offset, bytes);
	dma_buf_vunmap(dmabuf, vaddr);

out_cpu_end:
	end_ret = dma_buf_end_cpu_access(dmabuf, DMA_BIDIRECTIONAL);
	if (!ret)
		ret = end_ret;
out_put:
	dma_buf_put(dmabuf);
	return ret;
}

static int visiong_hw_spi_display_submit(struct visiong_hw_session *session,
					 unsigned long arg)
{
	struct visiong_hw_spi_display_submit request;
	struct visiong_hw_spi_slot *slot;
	struct visiong_hw_spi_job *job = NULL;
	unsigned int i;
	u32 job_index = 0;
	int ret;

	if (!session)
		return -EINVAL;
	if (copy_from_user(&request, (void __user *)arg, sizeof(request)))
		return -EFAULT;
	if (request.size < sizeof(request))
		return -EINVAL;
	if (request.dmabuf_fd < 0 || !request.bytes ||
	    request.bytes > VISIONG_HW_MAX_SPI_TRANSFER_BYTES)
		return -EINVAL;
	if (request.flags)
		return -EINVAL;

	mutex_lock(&session->lock);
	slot = visiong_hw_spi_find_slot(session, request.handle);
	if (!slot) {
		mutex_unlock(&session->lock);
		return -ENOENT;
	}

	for (i = 0; i < VISIONG_HW_MAX_SPI_JOBS_PER_DISPLAY; ++i) {
		if (!slot->jobs[i]) {
			job = kzalloc(sizeof(*job), GFP_KERNEL);
			if (!job) {
				mutex_unlock(&session->lock);
				return -ENOMEM;
			}
			job->active = true;
			job->handle = visiong_hw_spi_make_job_handle(slot->handle,
								     i);
			job->status = VISIONG_HW_SPI_STATUS_PENDING;
			job->bytes = request.bytes;
			init_completion(&job->done);
			slot->jobs[i] = job;
			job_index = i;
			break;
		}
	}
	mutex_unlock(&session->lock);
	if (!job)
		return -EBUSY;

	ret = visiong_hw_spi_copy_dmabuf(job, request.dmabuf_fd,
					 request.offset, request.bytes);
	if (ret)
		goto out_job;

	memset(&job->transfer, 0, sizeof(job->transfer));
	spi_message_init(&job->message);
	job->transfer.tx_buf = job->tx_buf;
	job->transfer.len = request.bytes;
	job->transfer.speed_hz = slot->speed_hz;
	job->transfer.bits_per_word = 8;
	spi_message_add_tail(&job->transfer, &job->message);
	job->message.complete = visiong_hw_spi_complete_func;
	job->message.context = job;

	ret = spi_async(slot->spi, &job->message);
	if (ret)
		goto out_job;

	job->submitted = true;
	request.job_handle = job->handle;
	request.size = sizeof(request);
	if (copy_to_user((void __user *)arg, &request, sizeof(request))) {
		mutex_lock(&session->lock);
		if (slot->jobs[job_index] == job)
			slot->jobs[job_index] = NULL;
		mutex_unlock(&session->lock);
		visiong_hw_spi_release_job(job);
		return -EFAULT;
	}
	return 0;

out_job:
	mutex_lock(&session->lock);
	if (slot && slot->active && slot->jobs[job_index] == job)
		slot->jobs[job_index] = NULL;
	mutex_unlock(&session->lock);
	visiong_hw_spi_release_job(job);
	return ret;
}

static int visiong_hw_spi_display_wait(struct visiong_hw_session *session,
				       unsigned long arg)
{
	struct visiong_hw_wait request;
	struct visiong_hw_spi_slot *slot;
	struct visiong_hw_spi_job *job;
	u32 slot_handle;
	u32 job_index;
	long timeout;
	u32 status;
	u64 timestamp;
	int ret = 0;

	if (!session)
		return -EINVAL;
	if (copy_from_user(&request, (void __user *)arg, sizeof(request)))
		return -EFAULT;
	if (request.size < sizeof(request))
		return -EINVAL;

	visiong_hw_spi_decode_job_handle(request.handle, &slot_handle,
					 &job_index);
	if (!slot_handle || job_index >= VISIONG_HW_MAX_SPI_JOBS_PER_DISPLAY)
		return -EINVAL;

	mutex_lock(&session->lock);
	slot = visiong_hw_spi_find_slot(session, slot_handle);
	if (!slot || !slot->jobs[job_index] ||
	    !slot->jobs[job_index]->active ||
	    slot->jobs[job_index]->handle != request.handle) {
		mutex_unlock(&session->lock);
		return -ENOENT;
	}
	job = slot->jobs[job_index];
	if (job->waiting) {
		mutex_unlock(&session->lock);
		return -EBUSY;
	}
	job->waiting = true;
	mutex_unlock(&session->lock);

	if (request.timeout_ms < 0) {
		ret = wait_for_completion_interruptible(&job->done);
		if (ret)
			goto out_unwait;
	} else {
		timeout = wait_for_completion_interruptible_timeout(
			&job->done, msecs_to_jiffies(request.timeout_ms));
		if (timeout < 0) {
			ret = timeout;
			goto out_unwait;
		}
		if (timeout == 0) {
			request.status = VISIONG_HW_SPI_STATUS_TIMEOUT;
			request.timestamp_ns_lo = 0;
			request.timestamp_ns_hi = 0;
			request.size = sizeof(request);
			mutex_lock(&session->lock);
			if (job->active && job->handle == request.handle)
				job->waiting = false;
			mutex_unlock(&session->lock);
			if (copy_to_user((void __user *)arg, &request,
					 sizeof(request)))
				return -EFAULT;
			return 0;
		}
	}

	status = job->status;
	timestamp = job->timestamp_ns;
	request.status = status;
	request.timestamp_ns_lo = lower_32_bits(timestamp);
	request.timestamp_ns_hi = upper_32_bits(timestamp);
	request.size = sizeof(request);
	if (copy_to_user((void __user *)arg, &request, sizeof(request)))
		ret = -EFAULT;

	mutex_lock(&session->lock);
	if (slot->jobs[job_index] == job)
		slot->jobs[job_index] = NULL;
	mutex_unlock(&session->lock);
	visiong_hw_spi_release_job(job);
	return ret;

out_unwait:
	mutex_lock(&session->lock);
	if (job->active && job->handle == request.handle)
		job->waiting = false;
	mutex_unlock(&session->lock);
	return ret;
}

static int visiong_hw_spi_display_close(struct visiong_hw_session *session,
					unsigned long arg)
{
	struct visiong_hw_wait request;
	struct visiong_hw_spi_slot *slot;

	if (!session)
		return -EINVAL;
	if (copy_from_user(&request, (void __user *)arg, sizeof(request)))
		return -EFAULT;
	if (request.size < sizeof(request))
		return -EINVAL;

	mutex_lock(&session->lock);
	slot = visiong_hw_spi_find_slot(session, request.handle);
	if (!slot) {
		mutex_unlock(&session->lock);
		return -ENOENT;
	}
	visiong_hw_spi_release_slot(slot);
	mutex_unlock(&session->lock);
	return 0;
}

static u32 visiong_hw_spi_fifo_len(void __iomem *regs)
{
	u32 version = readl(regs + 0x48);

	return (version == 0x05ec0002 || version == 0x00110002) ? 64U : 32U;
}

static unsigned long visiong_hw_spi_reg_timeout(u32 bytes, u32 speed_hz)
{
	u64 ms;

	if (!speed_hz)
		speed_hz = 1000000U;
	ms = div_u64((u64)bytes * 8ULL * 1000ULL, speed_hz);
	ms = ms * 4ULL + 500ULL;
	if (ms < 500ULL)
		ms = 500ULL;
	if (ms > 5000ULL)
		ms = 5000ULL;
	return msecs_to_jiffies((unsigned int)ms);
}

static const char *visiong_hw_spi_platform_name(u32 bus)
{
	switch (bus) {
	case 0:
		return "ff500000.spi";
	case 1:
		return "ff510000.spi";
	default:
		return NULL;
	}
}

static struct dma_chan *visiong_hw_spi_reg_dma_tx_chan(u32 bus)
{
	struct visiong_hw_spi_reg_dma_bus *dma;
	struct dma_chan *chan;
	struct device *dev;
	const char *name;

	if (bus >= ARRAY_SIZE(visiong_hw_spi_reg_dma))
		return ERR_PTR(-EINVAL);

	dma = &visiong_hw_spi_reg_dma[bus];
	mutex_lock(&dma->lock);
	if (dma->tx_chan) {
		chan = dma->tx_chan;
		goto out;
	}
	if (dma->unavailable) {
		chan = ERR_PTR(-EOPNOTSUPP);
		goto out;
	}

	name = visiong_hw_spi_platform_name(bus);
	if (!name) {
		chan = ERR_PTR(-EINVAL);
		goto out_mark;
	}

	dev = bus_find_device_by_name(&platform_bus_type, NULL, name);
	if (!dev) {
		chan = ERR_PTR(-ENODEV);
		goto out_mark;
	}

	chan = dma_request_chan(dev, "tx");
	if (IS_ERR(chan)) {
		pr_info("visiong_hw: spi%u register TX DMA unavailable (%ld), using PIO fallback\n",
			bus, PTR_ERR(chan));
		put_device(dev);
		goto out_mark;
	}

	dma->platform_dev = dev;
	dma->tx_chan = chan;
	pr_info("visiong_hw: spi%u register TX DMA channel acquired\n", bus);
	goto out;

out_mark:
	dma->unavailable = true;
out:
	mutex_unlock(&dma->lock);
	return chan;
}

static void visiong_hw_spi_reg_release_dma_bus(u32 bus)
{
	struct visiong_hw_spi_reg_dma_bus *dma;

	if (bus >= ARRAY_SIZE(visiong_hw_spi_reg_dma))
		return;

	dma = &visiong_hw_spi_reg_dma[bus];
	mutex_lock(&dma->lock);
	if (dma->tx_chan) {
		dmaengine_terminate_sync(dma->tx_chan);
		dma_release_channel(dma->tx_chan);
		dma->tx_chan = NULL;
		pr_info("visiong_hw: spi%u register TX DMA channel released\n", bus);
	}
	if (dma->platform_dev) {
		put_device(dma->platform_dev);
		dma->platform_dev = NULL;
	}
	dma->unavailable = false;
	mutex_unlock(&dma->lock);
}

static void visiong_hw_spi_reg_dma_complete(void *param)
{
	complete(param);
}

static int visiong_hw_spi_reg_wait_idle(void __iomem *regs, u32 bytes, u32 speed_hz)
{
	unsigned long deadline = jiffies + visiong_hw_spi_reg_timeout(bytes, speed_hz);

	while (readl(regs + 0x24) & 1U) {
		if (time_after(jiffies, deadline))
			return -ETIMEDOUT;
		cpu_relax();
		cond_resched();
	}
	return 0;
}

static int visiong_hw_spi_reg_dma_tx(struct visiong_hw_reg_region *region,
				     const struct visiong_hw_spi_reg_transfer *request,
				     const u8 *tx_buf,
				     u32 count,
				     u32 mode,
				     u32 div,
				     u32 fifo)
{
	struct dma_chan *chan;
	struct dma_slave_config config;
	struct dma_async_tx_descriptor *desc;
	struct completion done;
	dma_addr_t dma_addr;
	dma_cookie_t cookie;
	unsigned long timeout;
	u32 burst;
	int ret;

	chan = visiong_hw_spi_reg_dma_tx_chan(request->bus);
	if (IS_ERR(chan))
		return PTR_ERR(chan);

	burst = max_t(u32, 1U, fifo / 4U);
	memset(&config, 0, sizeof(config));
	config.direction = DMA_MEM_TO_DEV;
	config.dst_addr = region->base + 0x400;
	config.dst_addr_width = DMA_SLAVE_BUSWIDTH_1_BYTE;
	config.dst_maxburst = burst;

	ret = dmaengine_slave_config(chan, &config);
	if (ret)
		return ret;

	dma_addr = dma_map_single(chan->device->dev, (void *)tx_buf, count,
				  DMA_TO_DEVICE);
	if (dma_mapping_error(chan->device->dev, dma_addr))
		return -ENOMEM;

	desc = dmaengine_prep_slave_single(chan, dma_addr, count,
					   DMA_MEM_TO_DEV,
					   DMA_PREP_INTERRUPT | DMA_CTRL_ACK);
	if (!desc) {
		ret = -EOPNOTSUPP;
		goto out_unmap;
	}

	init_completion(&done);
	desc->callback = visiong_hw_spi_reg_dma_complete;
	desc->callback_param = &done;

	cookie = dmaengine_submit(desc);
	ret = dma_submit_error(cookie);
	if (ret)
		goto out_unmap;

	writel(0, region->map + 0x08);
	writel(0, region->map + 0x2c);
	writel(0xffffffff, region->map + 0x38);
	writel(0, region->map + 0x3c);
	writel(0x1 | (1U << 10) | (1U << 11) | (1U << 13) |
		       mode | (1U << 18),
	       region->map + 0x00);
	writel(count - 1, region->map + 0x04);
	writel(max_t(u32, 1U, fifo / 2U), region->map + 0x14);
	writel(0, region->map + 0x18);
	writel(max_t(u32, 1U, fifo / 2U) - 1U, region->map + 0x40);
	writel(0, region->map + 0x44);
	writel(div, region->map + 0x10);
	writel(1U << request->chip_select, region->map + 0x0c);
	writel(1U << 1, region->map + 0x3c);
	writel(1, region->map + 0x08);

	dma_async_issue_pending(chan);
	timeout = wait_for_completion_timeout(
		&done, visiong_hw_spi_reg_timeout(count, request->speed_hz));
	if (!timeout) {
		dmaengine_terminate_sync(chan);
		ret = -ETIMEDOUT;
		goto out_disable;
	}

	if (dma_async_is_tx_complete(chan, cookie, NULL, NULL) != DMA_COMPLETE) {
		ret = -EIO;
		goto out_disable;
	}

	ret = visiong_hw_spi_reg_wait_idle(region->map, count,
					   request->speed_hz);

out_disable:
	writel(0, region->map + 0x3c);
	writel(0, region->map + 0x08);
	writel(0, region->map + 0x0c);
out_unmap:
	dma_unmap_single(chan->device->dev, dma_addr, count, DMA_TO_DEVICE);
	return ret;
}

static int visiong_hw_spi_reg_enable_clocks(u32 bus)
{
	struct visiong_hw_reg_region *cru;
	u32 mask;

	cru = visiong_hw_find_region(VISIONG_HW_REG_BLOCK_CRU);
	if (!cru || !cru->map)
		return -ENODEV;

	switch (bus) {
	case 0:
		mask = GENMASK(13, 12);
		writel((mask << 16) | 0, cru->map + 0x1a300);
		writel(((1U << 2) | (1U << 3) | (1U << 4)) << 16,
		       cru->map + 0x1a804);
		break;
	case 1:
		mask = GENMASK(4, 3);
		writel((mask << 16) | 0, cru->map + 0x12318);
		writel(((1U << 6) | (1U << 7)) << 16,
		       cru->map + 0x1280c);
		break;
	default:
		return -EINVAL;
	}

	return 0;
}

static int visiong_hw_spi_reg_transfer(unsigned long arg)
{
	struct visiong_hw_spi_reg_transfer request;
	struct visiong_hw_reg_region *region;
	void __user *tx_user;
	void __user *rx_user;
	u8 *tx_buf = NULL;
	u8 rx_buf[64];
	u32 mode;
	u32 xfm;
	u32 div;
	u32 fifo;
	u32 total_frames;
	u32 total_written = 0;
	u32 total_rx = 0;
	u32 transferred = 0;
	u32 block;
	int ret = 0;

	if (!capable(CAP_SYS_RAWIO))
		return -EPERM;
	if (copy_from_user(&request, (void __user *)arg, sizeof(request)))
		return -EFAULT;
	if (request.size < sizeof(request))
		return -EINVAL;
	if (request.bus > 1 || request.chip_select > 3 ||
	    request.bits_per_word != 8 || !request.speed_hz ||
	    !request.source_clock_hz)
		return -EINVAL;
	if (request.mode & ~0x3U)
		return -EINVAL;
	if (request.flags & ~VISIONG_HW_SPI_REG_TX_ONLY)
		return -EINVAL;
	if ((request.flags & VISIONG_HW_SPI_REG_TX_ONLY) && request.rx_len)
		return -EINVAL;
	if (request.tx_len > VISIONG_HW_MAX_TRANSFER_BYTES ||
	    request.rx_len > VISIONG_HW_MAX_TRANSFER_BYTES)
		return -EINVAL;
	if (request.tx_len && !request.tx_ptr)
		return -EINVAL;
	if (request.rx_len && !request.rx_ptr)
		return -EINVAL;
	if (!request.tx_len && !request.rx_len)
		return -EINVAL;

	block = request.bus == 0 ? VISIONG_HW_REG_BLOCK_SPI0 :
				   VISIONG_HW_REG_BLOCK_SPI1;
	region = visiong_hw_find_region(block);
	if (!region || !region->map)
		return -ENODEV;

	ret = visiong_hw_spi_reg_enable_clocks(request.bus);
	if (ret)
		return ret;

	tx_user = (void __user *)(uintptr_t)request.tx_ptr;
	rx_user = (void __user *)(uintptr_t)request.rx_ptr;
	total_frames = (request.flags & VISIONG_HW_SPI_REG_TX_ONLY) ?
			       request.tx_len :
			       max(request.tx_len, request.rx_len);
	if (!total_frames)
		return -EINVAL;

	tx_buf = kmalloc(min_t(u32, total_frames, 0xffffU), GFP_KERNEL);
	if (!tx_buf)
		return -ENOMEM;

	mode = ((request.mode & 0x1) ? 1U : 0U) |
	       ((request.mode & 0x2) ? 2U : 0U);
	mode <<= 6;
	xfm = (request.flags & VISIONG_HW_SPI_REG_TX_ONLY) ?
		      (1U << 18) :
		      (0U << 18);
	div = max_t(u32, 2U,
		    DIV_ROUND_UP(request.source_clock_hz, request.speed_hz));
	if (div & 1U)
		div++;
	div = min_t(u32, div, 65534U);

	mutex_lock(&visiong_hw_spi_reg_lock);
	fifo = visiong_hw_spi_fifo_len(region->map);

	while (transferred < total_frames) {
		u32 count = min_t(u32, total_frames - transferred, 0xffffU);
		u32 tx_available = 0;
		u32 tx_pos = 0;
		u32 rx_target;
		u32 rx_pos = 0;
		unsigned long deadline;

		if (transferred < request.tx_len)
			tx_available = min_t(u32, request.tx_len - transferred,
					     count);
		if (tx_available &&
		    copy_from_user(tx_buf, tx_user + transferred, tx_available)) {
			ret = -EFAULT;
			goto out_unlock;
		}
			if (tx_available < count)
				memset(tx_buf + tx_available, request.dummy & 0xff,
				       count - tx_available);

			if ((request.flags & VISIONG_HW_SPI_REG_TX_ONLY) &&
			    count >= VISIONG_HW_SPI_REG_DMA_MIN_BYTES) {
				ret = visiong_hw_spi_reg_dma_tx(region, &request,
								tx_buf, count,
								mode, div, fifo);
				if (!ret) {
					total_written += count;
					transferred += count;
					continue;
				}
				if (ret == -ETIMEDOUT || ret == -EIO)
					goto out_disable;
				ret = 0;
			}

			writel(0, region->map + 0x08);
			writel(0, region->map + 0x2c);
			writel(0xffffffff, region->map + 0x38);
		writel(0, region->map + 0x3c);
		writel(0x1 | (1U << 10) | (1U << 11) | (1U << 13) |
			       mode | xfm,
		       region->map + 0x00);
		writel(count - 1, region->map + 0x04);
		writel(max_t(u32, 1U, fifo / 2U), region->map + 0x14);
		writel(0, region->map + 0x18);
		writel(div, region->map + 0x10);
		writel(1U << request.chip_select, region->map + 0x0c);
		writel(1, region->map + 0x08);

		rx_target = (request.flags & VISIONG_HW_SPI_REG_TX_ONLY) ? 0 :
								   count;
			deadline = jiffies + visiong_hw_spi_reg_timeout(count,
									request.speed_hz);
			while (tx_pos < count || rx_pos < rx_target) {
			bool progressed = false;

			if (tx_pos < count) {
				u32 level = readl(region->map + 0x1c);

				if (level < fifo) {
					u32 writable = min_t(u32, fifo - level,
							     count - tx_pos);
					u32 i;

					for (i = 0; i < writable; ++i)
						writel_relaxed(tx_buf[tx_pos + i],
							       region->map + 0x400);
					wmb();
					tx_pos += writable;
					total_written += writable;
					progressed = true;
				}
			}

			if (rx_pos < rx_target) {
				u32 readable = min_t(u32,
						     readl(region->map + 0x20),
						     rx_target - rx_pos);

				while (readable) {
					u32 batch = min_t(u32, readable,
							  ARRAY_SIZE(rx_buf));
					u32 i;

					for (i = 0; i < batch; ++i)
						rx_buf[i] = readl(region->map + 0x800) & 0xff;
					if (total_rx < request.rx_len) {
						u32 keep = min_t(u32, batch,
								 request.rx_len - total_rx);

						if (copy_to_user(rx_user + total_rx,
								 rx_buf, keep)) {
							ret = -EFAULT;
							goto out_disable;
						}
						total_rx += keep;
					}
					rx_pos += batch;
					readable -= batch;
				}
				progressed = true;
			}

				if (progressed) {
					deadline = jiffies +
						   visiong_hw_spi_reg_timeout(count,
									       request.speed_hz);
				} else {
				if (time_after(jiffies, deadline)) {
					ret = -ETIMEDOUT;
					goto out_disable;
				}
				cpu_relax();
				cond_resched();
			}
			}

			ret = visiong_hw_spi_reg_wait_idle(region->map, count,
							   request.speed_hz);
			if (ret)
					goto out_disable;
			writel(0, region->map + 0x08);
			writel(0, region->map + 0x0c);
		transferred += count;
	}

out_disable:
	writel(0, region->map + 0x08);
	writel(0, region->map + 0x0c);
out_unlock:
	mutex_unlock(&visiong_hw_spi_reg_lock);
	kfree(tx_buf);

	request.status = ret ? VISIONG_HW_SPI_STATUS_ERROR :
			       VISIONG_HW_SPI_STATUS_DONE;
	request.transferred = (request.flags & VISIONG_HW_SPI_REG_TX_ONLY) ?
				      total_written :
				      total_rx;
	request.size = sizeof(request);
	if (copy_to_user((void __user *)arg, &request, sizeof(request)))
		return -EFAULT;
	return ret;
}

static int visiong_hw_spi_reg_release(unsigned long arg)
{
	struct visiong_hw_spi_reg_release request;
	struct visiong_hw_reg_region *region;
	u32 block;

	if (!capable(CAP_SYS_RAWIO))
		return -EPERM;
	if (copy_from_user(&request, (void __user *)arg, sizeof(request)))
		return -EFAULT;
	if (request.size < sizeof(request))
		return -EINVAL;
	if (request.bus > 1 || request.flags)
		return -EINVAL;

	block = request.bus == 0 ? VISIONG_HW_REG_BLOCK_SPI0 :
				   VISIONG_HW_REG_BLOCK_SPI1;
	region = visiong_hw_find_region(block);
	if (!region || !region->map)
		return -ENODEV;

	mutex_lock(&visiong_hw_spi_reg_lock);
	writel(0, region->map + 0x3c);
	writel(0, region->map + 0x08);
	writel(0, region->map + 0x0c);
	mutex_unlock(&visiong_hw_spi_reg_lock);

	visiong_hw_spi_reg_release_dma_bus(request.bus);
	return 0;
}

static irqreturn_t visiong_hw_irq_thread(int irq, void *data)
{
	struct visiong_hw_irq_slot *slot = data;
	unsigned long flags;

	spin_lock_irqsave(&slot->lock, flags);
	slot->seq++;
	slot->timestamp_ns = ktime_get_ns();
	spin_unlock_irqrestore(&slot->lock, flags);

	wake_up_interruptible(&slot->waitq);
	return IRQ_HANDLED;
}

static void visiong_hw_irq_release_slot(struct visiong_hw_irq_slot *slot)
{
	if (!slot->active)
		return;

	free_irq(slot->irq, slot);
	gpio_free(slot->gpio);
	memset(slot, 0, sizeof(*slot));
}

static void visiong_hw_release_irqs(struct visiong_hw_session *session)
{
	unsigned int i;

	if (!session)
		return;

	mutex_lock(&session->lock);
	for (i = 0; i < VISIONG_HW_MAX_IRQS_PER_SESSION; ++i)
		visiong_hw_irq_release_slot(&session->irqs[i]);
	mutex_unlock(&session->lock);
}

static unsigned long visiong_hw_irq_flags(u32 edge)
{
	unsigned long flags = IRQF_ONESHOT;

	if (edge & VISIONG_HW_IRQ_EDGE_RISING)
		flags |= IRQF_TRIGGER_RISING;
	if (edge & VISIONG_HW_IRQ_EDGE_FALLING)
		flags |= IRQF_TRIGGER_FALLING;
	return flags;
}

static int visiong_hw_irq_request(struct visiong_hw_session *session,
				  unsigned long arg)
{
	struct visiong_hw_irq_request request;
	struct visiong_hw_irq_slot *slot = NULL;
	unsigned int i;
	unsigned int gpio;
	int irq;
	int ret;

	if (!session)
		return -EINVAL;
	if (copy_from_user(&request, (void __user *)arg, sizeof(request)))
		return -EFAULT;
	if (request.size < sizeof(request))
		return -EINVAL;
	if (request.bank > 4 || request.pin > 31)
		return -EINVAL;
	if (!request.edge || (request.edge & ~VISIONG_HW_IRQ_EDGE_BOTH))
		return -EINVAL;
	if (request.flags)
		return -EINVAL;

	gpio = request.bank * 32U + request.pin;

	mutex_lock(&session->lock);
	for (i = 0; i < VISIONG_HW_MAX_IRQS_PER_SESSION; ++i) {
		if (!session->irqs[i].active) {
			slot = &session->irqs[i];
			break;
		}
	}
	if (!slot) {
		mutex_unlock(&session->lock);
		return -ENOSPC;
	}

	memset(slot, 0, sizeof(*slot));
	slot->handle = i + 1U;
	slot->bank = request.bank;
	slot->pin = request.pin;
	slot->gpio = gpio;
	spin_lock_init(&slot->lock);
	init_waitqueue_head(&slot->waitq);

	ret = gpio_request_one(gpio, GPIOF_IN, "visiong-hw-irq");
	if (ret)
		goto fail;

	irq = gpio_to_irq(gpio);
	if (irq < 0) {
		ret = irq;
		goto fail_gpio;
	}
	slot->irq = irq;

	ret = request_threaded_irq(slot->irq, NULL, visiong_hw_irq_thread,
				   visiong_hw_irq_flags(request.edge),
				   "visiong-hw-irq", slot);
	if (ret)
		goto fail_gpio;

	slot->active = true;
	request.handle = slot->handle;
	request.size = sizeof(request);
	mutex_unlock(&session->lock);

	if (copy_to_user((void __user *)arg, &request, sizeof(request))) {
		mutex_lock(&session->lock);
		visiong_hw_irq_release_slot(slot);
		mutex_unlock(&session->lock);
		return -EFAULT;
	}

	return 0;

fail_gpio:
	gpio_free(gpio);
fail:
	memset(slot, 0, sizeof(*slot));
	mutex_unlock(&session->lock);
	return ret;
}

static int visiong_hw_irq_wait(struct visiong_hw_session *session,
			       unsigned long arg)
{
	struct visiong_hw_wait request;
	struct visiong_hw_irq_slot *slot;
	long timeout;
	u32 last_seen;
	u32 seq;
	u64 timestamp;
	unsigned long flags;
	int ret = 0;

	if (!session)
		return -EINVAL;
	if (copy_from_user(&request, (void __user *)arg, sizeof(request)))
		return -EFAULT;
	if (request.size < sizeof(request))
		return -EINVAL;
	if (request.handle == 0 ||
	    request.handle > VISIONG_HW_MAX_IRQS_PER_SESSION)
		return -EINVAL;

	mutex_lock(&session->lock);
	slot = &session->irqs[request.handle - 1U];
	if (!slot->active) {
		mutex_unlock(&session->lock);
		return -EINVAL;
	}
	last_seen = request.status;
	mutex_unlock(&session->lock);

	if (request.timeout_ms < 0) {
		ret = wait_event_interruptible(slot->waitq, slot->seq != last_seen);
		if (ret)
			return ret;
	} else {
		timeout = msecs_to_jiffies(request.timeout_ms);
		ret = wait_event_interruptible_timeout(slot->waitq,
						       slot->seq != last_seen,
						       timeout);
		if (ret < 0)
			return ret;
		if (ret == 0 && slot->seq == last_seen) {
			request.status = last_seen;
			request.timestamp_ns_lo = 0;
			request.timestamp_ns_hi = 0;
			if (copy_to_user((void __user *)arg, &request, sizeof(request)))
				return -EFAULT;
			return 0;
		}
	}

	spin_lock_irqsave(&slot->lock, flags);
	seq = slot->seq;
	timestamp = slot->timestamp_ns;
	spin_unlock_irqrestore(&slot->lock, flags);

	request.status = seq;
	request.timestamp_ns_lo = lower_32_bits(timestamp);
	request.timestamp_ns_hi = upper_32_bits(timestamp);
	request.size = sizeof(request);

	if (copy_to_user((void __user *)arg, &request, sizeof(request)))
		return -EFAULT;
	return 0;
}

static int visiong_hw_irq_release(struct visiong_hw_session *session,
				  unsigned long arg)
{
	struct visiong_hw_wait request;
	struct visiong_hw_irq_slot *slot;

	if (!session)
		return -EINVAL;
	if (copy_from_user(&request, (void __user *)arg, sizeof(request)))
		return -EFAULT;
	if (request.size < sizeof(request))
		return -EINVAL;
	if (!request.handle || request.handle > VISIONG_HW_MAX_IRQS_PER_SESSION)
		return -EINVAL;

	mutex_lock(&session->lock);
	slot = &session->irqs[request.handle - 1];
	if (!slot->active || slot->handle != request.handle) {
		mutex_unlock(&session->lock);
		return -ENOENT;
	}
	visiong_hw_irq_release_slot(slot);
	mutex_unlock(&session->lock);
	return 0;
}

static long visiong_hw_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
	struct visiong_hw_session *session = file->private_data;

	switch (cmd) {
	case VISIONG_HW_GET_CAPS:
		return visiong_hw_get_caps(arg);
	case VISIONG_HW_REG_READ:
		return visiong_hw_reg_read(arg);
	case VISIONG_HW_REG_WRITE:
		return visiong_hw_reg_write(arg);
	case VISIONG_HW_DMA_ALLOC:
		return visiong_hw_dma_alloc(arg);
	case VISIONG_HW_DMA_SYNC:
		return visiong_hw_dma_sync(arg);
	case VISIONG_HW_DMA_MEMCPY:
		return visiong_hw_dma_memcpy(session, arg);
	case VISIONG_HW_DMA_WAIT:
		return visiong_hw_dma_wait(session, arg);
	case VISIONG_HW_DMA_FILL:
		return visiong_hw_dma_fill(arg);
	case VISIONG_HW_IRQ_REQUEST:
		return visiong_hw_irq_request(session, arg);
	case VISIONG_HW_IRQ_WAIT:
		return visiong_hw_irq_wait(session, arg);
	case VISIONG_HW_IRQ_RELEASE:
		return visiong_hw_irq_release(session, arg);
	case VISIONG_HW_SPI_DISPLAY_OPEN:
		return visiong_hw_spi_display_open(session, arg);
	case VISIONG_HW_SPI_DISPLAY_SUBMIT:
		return visiong_hw_spi_display_submit(session, arg);
	case VISIONG_HW_SPI_DISPLAY_WAIT:
		return visiong_hw_spi_display_wait(session, arg);
	case VISIONG_HW_SPI_DISPLAY_CLOSE:
		return visiong_hw_spi_display_close(session, arg);
	case VISIONG_HW_SPI_REG_TRANSFER:
		return visiong_hw_spi_reg_transfer(arg);
	case VISIONG_HW_SPI_REG_RELEASE:
		return visiong_hw_spi_reg_release(arg);
	default:
		return -ENOTTY;
	}
}

static int visiong_hw_open(struct inode *inode, struct file *file)
{
	struct visiong_hw_session *session;

	session = kzalloc(sizeof(*session), GFP_KERNEL);
	if (!session)
		return -ENOMEM;
	mutex_init(&session->lock);
	file->private_data = session;
	return 0;
}

static int visiong_hw_release(struct inode *inode, struct file *file)
{
	visiong_hw_release_spi_slots(file->private_data);
	visiong_hw_release_dma_jobs(file->private_data);
	visiong_hw_release_irqs(file->private_data);
	kfree(file->private_data);
	file->private_data = NULL;
	return 0;
}

static const struct file_operations visiong_hw_fops = {
	.owner = THIS_MODULE,
	.open = visiong_hw_open,
	.release = visiong_hw_release,
	.unlocked_ioctl = visiong_hw_ioctl,
#ifdef CONFIG_COMPAT
	.compat_ioctl = visiong_hw_ioctl,
#endif
	.llseek = no_llseek,
};

static struct miscdevice visiong_hw_miscdev = {
	.minor = MISC_DYNAMIC_MINOR,
	.name = VISIONG_HW_DEVICE_NAME,
	.fops = &visiong_hw_fops,
	.mode = 0600,
};

static void visiong_hw_unmap_regions(void)
{
	unsigned int i;

	for (i = 0; i < ARRAY_SIZE(visiong_hw_regions); ++i) {
		if (visiong_hw_regions[i].map) {
			iounmap(visiong_hw_regions[i].map);
			visiong_hw_regions[i].map = NULL;
		}
	}
}

static int visiong_hw_map_regions(void)
{
	unsigned int i;

	for (i = 0; i < ARRAY_SIZE(visiong_hw_regions); ++i) {
		visiong_hw_regions[i].map =
			ioremap(visiong_hw_regions[i].base, visiong_hw_regions[i].size);
		if (!visiong_hw_regions[i].map) {
			pr_err("visiong_hw: failed to map %s at %pa\n",
			       visiong_hw_regions[i].name, &visiong_hw_regions[i].base);
			visiong_hw_unmap_regions();
			return -ENOMEM;
		}
	}
	return 0;
}

static int __init visiong_hw_init(void)
{
	int ret;
	unsigned int i;

	for (i = 0; i < ARRAY_SIZE(visiong_hw_spi_reg_dma); ++i)
		mutex_init(&visiong_hw_spi_reg_dma[i].lock);

	ret = visiong_hw_map_regions();
	if (ret)
		return ret;

	ret = spi_register_driver(&visiong_hw_spi_driver);
	if (ret) {
		visiong_hw_unmap_regions();
		return ret;
	}

	ret = misc_register(&visiong_hw_miscdev);
	if (ret) {
		spi_unregister_driver(&visiong_hw_spi_driver);
		visiong_hw_unmap_regions();
		return ret;
	}

	visiong_hw_device = visiong_hw_miscdev.this_device;
	dma_set_mask_and_coherent(visiong_hw_device, DMA_BIT_MASK(32));

	pr_info("visiong_hw: registered /dev/%s ABI %u\n",
		VISIONG_HW_DEVICE_NAME, VISIONG_HW_ABI_VERSION);
	return 0;
}

static void __exit visiong_hw_exit(void)
{
	unsigned int i;

	mutex_lock(&visiong_hw_dma_chan_lock);
	if (visiong_hw_memcpy_chan) {
		dma_release_channel(visiong_hw_memcpy_chan);
		visiong_hw_memcpy_chan = NULL;
	}
	mutex_unlock(&visiong_hw_dma_chan_lock);
	misc_deregister(&visiong_hw_miscdev);
	spi_unregister_driver(&visiong_hw_spi_driver);
	visiong_hw_device = NULL;
	for (i = 0; i < ARRAY_SIZE(visiong_hw_spi_reg_dma); ++i) {
		mutex_lock(&visiong_hw_spi_reg_dma[i].lock);
		if (visiong_hw_spi_reg_dma[i].tx_chan) {
			dmaengine_terminate_sync(visiong_hw_spi_reg_dma[i].tx_chan);
			dma_release_channel(visiong_hw_spi_reg_dma[i].tx_chan);
			visiong_hw_spi_reg_dma[i].tx_chan = NULL;
		}
		if (visiong_hw_spi_reg_dma[i].platform_dev) {
			put_device(visiong_hw_spi_reg_dma[i].platform_dev);
			visiong_hw_spi_reg_dma[i].platform_dev = NULL;
		}
		visiong_hw_spi_reg_dma[i].unavailable = false;
		mutex_unlock(&visiong_hw_spi_reg_dma[i].lock);
	}
	visiong_hw_unmap_regions();
	pr_info("visiong_hw: unloaded\n");
}

module_init(visiong_hw_init);
module_exit(visiong_hw_exit);

MODULE_DESCRIPTION("VisionG RV1103/RV1106 low-level hardware helper");
MODULE_AUTHOR("VisionG contributors");
MODULE_LICENSE("GPL");
