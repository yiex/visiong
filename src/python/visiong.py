# SPDX-License-Identifier: LGPL-3.0-or-later
import ctypes
import importlib
import os
import re
import struct
import sys
import threading
import time

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_RTLD_NOW = getattr(os, "RTLD_NOW", 2)
_RTLD_GLOBAL = getattr(os, "RTLD_GLOBAL", 0x100)
_RTLD_MODE = _RTLD_NOW | _RTLD_GLOBAL
_LD_MARKER = "_VISIONG_LD_PATH_READY"


def _ensure_loader_library_path():
    required_paths = [_MODULE_DIR, "/oem/usr/lib"]
    current = os.environ.get("LD_LIBRARY_PATH", "")
    entries = [p for p in current.split(":") if p]
    missing = [p for p in required_paths if p not in entries]

    if not missing:
        os.environ[_LD_MARKER] = "1"
        return

    if os.environ.get(_LD_MARKER) == "1":
        return

    os.environ[_LD_MARKER] = "1"
    os.environ["LD_LIBRARY_PATH"] = ":".join(required_paths + entries)

    exe = sys.executable
    if exe:
        argv = [exe] + (sys.argv if sys.argv else [])
        os.execv(exe, argv)


def _preload_rockit_global():
    candidates = (
        os.path.join(_MODULE_DIR, "librockit.so"),
        "/oem/usr/lib/librockit.so",
        os.path.join(_MODULE_DIR, "librockit_full.so"),
        "/oem/usr/lib/librockit_full.so",
    )
    for lib in candidates:
        try:
            ctypes.CDLL(lib, mode=_RTLD_MODE)
            return
        except OSError:
            continue


_ensure_loader_library_path()

_old_flags = sys.getdlopenflags()
try:
    if _MODULE_DIR not in sys.path:
        sys.path.insert(0, _MODULE_DIR)
    sys.setdlopenflags(_RTLD_MODE)
    _preload_rockit_global()
    _mod = importlib.import_module("_visiong")
finally:
    try:
        sys.setdlopenflags(_old_flags)
    except Exception:
        pass

for _name in dir(_mod):
    if _name.startswith("__") and _name not in ("__doc__", "__version__"):
        continue
    globals()[_name] = getattr(_mod, _name)


def _pin_name(pin):
    return str(getattr(pin, "id", pin))


def _append_pin_names(out, value):
    if value is None:
        return
    if isinstance(value, dict):
        iterable = value.keys()
    elif isinstance(value, (list, tuple, set)):
        iterable = value
    else:
        iterable = (value,)
    for item in iterable:
        if item is not None:
            out.append(_pin_name(item))


def _normal_token(value):
    return str(value).strip().lower().replace("_", "-")


def _backend_name(value):
    text = str(value or "auto").strip().lower()
    return text or "auto"


def _ioc(direction, magic, number, size):
    nr_bits = 8
    type_bits = 8
    size_bits = 14
    nr_shift = 0
    type_shift = nr_shift + nr_bits
    size_shift = type_shift + type_bits
    dir_shift = size_shift + size_bits
    return (
        (int(direction) << dir_shift)
        | (int(magic) << type_shift)
        | (int(number) << nr_shift)
        | (int(size) << size_shift)
    )


class HWReg:
    def __init__(self, hw, block, base_offset=0):
        self._hw = hw
        self.block = block
        self.base_offset = int(base_offset)

    def read32(self, offset):
        return self._hw.read32(self.block, self.base_offset + int(offset))

    def write32(self, offset, value, mask=0xFFFFFFFF, *, hiword=False):
        return self._hw.write32(self.block, self.base_offset + int(offset), value, mask=mask, hiword=hiword)

    def update32(self, offset, mask, value):
        return self.write32(offset, value, mask=mask)

    def set_bits(self, offset, mask):
        old = self.read32(offset)
        return self.write32(offset, old | int(mask), mask=int(mask))

    def clear_bits(self, offset, mask):
        return self.write32(offset, 0, mask=int(mask))

    def __repr__(self):
        return f"HWReg(block='{self.block}', base_offset=0x{self.base_offset:x})"


class HW:
    FEATURE_REG_ACCESS = 1 << 0
    FEATURE_PIN_SESSION = 1 << 1
    FEATURE_GPIO_IRQ = 1 << 2
    FEATURE_DMA_BUFFER = 1 << 3
    FEATURE_DMA_MEMCPY = 1 << 4
    FEATURE_SPI_DISPLAY = 1 << 5
    FEATURE_DMA_FILL = 1 << 6
    FEATURE_SPI_REG = 1 << 7

    REG_FLAG_HIWORD_UPDATE = 1 << 0
    DMA_ALLOC_WRITE_COMBINE = 1 << 0
    DMA_SYNC_BIDIRECTIONAL = 0
    DMA_SYNC_TO_DEVICE = 1
    DMA_SYNC_FROM_DEVICE = 2
    DMA_SYNC_START = 1 << 0
    DMA_SYNC_END = 1 << 1
    DMA_MEMCPY_ASYNC = 1 << 0
    DMA_MEMCPY_STATUS_DONE = 0
    DMA_MEMCPY_STATUS_TIMEOUT = 1
    DMA_MEMCPY_STATUS_ERROR = 2
    DMA_MEMCPY_STATUS_PENDING = 3
    SPI_REG_TX_ONLY = 1 << 0
    IRQ_EDGE_RISING = 1 << 0
    IRQ_EDGE_FALLING = 1 << 1
    IRQ_EDGE_BOTH = IRQ_EDGE_RISING | IRQ_EDGE_FALLING

    _PATH = "/dev/visiong-hw"
    _MAGIC = ord("V")
    _IOC_WRITE = 1
    _IOC_READ = 2
    _CAPS_STRUCT = struct.Struct("=16I")
    _REG_STRUCT = struct.Struct("=8I")
    _DMA_ALLOC_STRUCT = struct.Struct("=IIIiIIII")
    _DMA_SYNC_STRUCT = struct.Struct("=IiIIIIIIII")
    _DMA_MEMCPY_STRUCT = struct.Struct("=IiiIIIIIIIII")
    _DMA_FILL_STRUCT = struct.Struct("=Ii10I")
    _IRQ_REQ_STRUCT = struct.Struct("=10I")
    _SPI_OPEN_STRUCT = struct.Struct("=16I")
    _SPI_SUBMIT_STRUCT = struct.Struct("=IIi13I")
    _SPI_REG_TRANSFER_STRUCT = struct.Struct("=8IQQ8I")
    _SPI_REG_RELEASE_STRUCT = struct.Struct("=8I")
    _WAIT_STRUCT = struct.Struct("=IIiIIIIIII")
    _GET_CAPS = _ioc(_IOC_READ, _MAGIC, 0x00, _CAPS_STRUCT.size)
    _REG_READ = _ioc(_IOC_READ | _IOC_WRITE, _MAGIC, 0x10, _REG_STRUCT.size)
    _REG_WRITE = _ioc(_IOC_WRITE, _MAGIC, 0x11, _REG_STRUCT.size)
    _DMA_ALLOC = _ioc(_IOC_READ | _IOC_WRITE, _MAGIC, 0x20, _DMA_ALLOC_STRUCT.size)
    _DMA_SYNC = _ioc(_IOC_WRITE, _MAGIC, 0x21, _DMA_SYNC_STRUCT.size)
    _DMA_MEMCPY = _ioc(_IOC_READ | _IOC_WRITE, _MAGIC, 0x22, _DMA_MEMCPY_STRUCT.size)
    _DMA_WAIT = _ioc(_IOC_READ | _IOC_WRITE, _MAGIC, 0x23, _WAIT_STRUCT.size)
    _DMA_FILL = _ioc(_IOC_READ | _IOC_WRITE, _MAGIC, 0x24, _DMA_FILL_STRUCT.size)
    _IRQ_REQUEST = _ioc(_IOC_READ | _IOC_WRITE, _MAGIC, 0x30, _IRQ_REQ_STRUCT.size)
    _IRQ_WAIT = _ioc(_IOC_READ | _IOC_WRITE, _MAGIC, 0x31, _WAIT_STRUCT.size)
    _IRQ_RELEASE = _ioc(_IOC_WRITE, _MAGIC, 0x32, _WAIT_STRUCT.size)
    _SPI_DISPLAY_OPEN = _ioc(_IOC_READ | _IOC_WRITE, _MAGIC, 0x40, _SPI_OPEN_STRUCT.size)
    _SPI_DISPLAY_SUBMIT = _ioc(_IOC_READ | _IOC_WRITE, _MAGIC, 0x41, _SPI_SUBMIT_STRUCT.size)
    _SPI_DISPLAY_WAIT = _ioc(_IOC_READ | _IOC_WRITE, _MAGIC, 0x42, _WAIT_STRUCT.size)
    _SPI_DISPLAY_CLOSE = _ioc(_IOC_WRITE, _MAGIC, 0x43, _WAIT_STRUCT.size)
    _SPI_REG_TRANSFER = _ioc(_IOC_READ | _IOC_WRITE, _MAGIC, 0x44, _SPI_REG_TRANSFER_STRUCT.size)
    _SPI_REG_RELEASE = _ioc(_IOC_WRITE, _MAGIC, 0x45, _SPI_REG_RELEASE_STRUCT.size)

    _BLOCKS = {
        "ioc": 0,
        "gpio": 0,
        "pinctrl": 0,
        "pmuioc": 1,
        "cru": 2,
        "clock": 2,
        "gpio0": 3,
        "gpio1": 4,
        "gpio2": 5,
        "gpio3": 6,
        "gpio4": 7,
        "spi0": 8,
        "spi1": 9,
        "i2c0": 10,
        "i2c1": 11,
        "i2c2": 12,
        "i2c3": 13,
        "i2c4": 14,
        "uart0": 15,
        "serial0": 15,
        "uart1": 16,
        "serial1": 16,
        "uart2": 17,
        "serial2": 17,
        "uart3": 18,
        "serial3": 18,
        "uart4": 19,
        "serial4": 19,
        "uart5": 20,
        "serial5": 20,
        "pwm0_3": 21,
        "pwm4_7": 22,
        "pwm8_11": 23,
        "dmac": 24,
        "dma": 24,
        "gicd": 25,
        "gic": 25,
    }

    def __init__(self, path=None, *, required=False, autoload=True):
        self.path = path or self._PATH
        self._fd = None
        if autoload and not os.path.exists(self.path):
            self.load()
        try:
            self._fd = os.open(self.path, os.O_RDWR | getattr(os, "O_CLOEXEC", 0))
        except OSError:
            if required:
                raise

    @staticmethod
    def is_available(path=None):
        path = path or HW._PATH
        try:
            fd = os.open(path, os.O_RDWR | getattr(os, "O_CLOEXEC", 0))
            os.close(fd)
            return True
        except OSError:
            return False

    @staticmethod
    def module_candidates(module_path=None):
        if module_path:
            return [module_path]
        return [
            os.path.join(_MODULE_DIR, "visiong_hw.ko"),
            os.path.join(_MODULE_DIR, "drivers", "visiong_hw.ko"),
            os.path.join(os.getcwd(), "visiong_hw.ko"),
        ]

    @staticmethod
    def load(module_path=None):
        if HW.is_available():
            return True
        import subprocess

        for candidate in HW.module_candidates(module_path):
            if not candidate or not os.path.exists(candidate):
                continue
            proc = subprocess.run(["insmod", candidate], capture_output=True, text=True)
            if proc.returncode == 0 or HW.is_available():
                return True
            message = (proc.stderr or proc.stdout or "").lower()
            if "file exists" in message or "already exists" in message:
                return HW.is_available()
        return HW.is_available()

    ensure_loaded = load

    def available(self):
        return self._fd is not None

    def _require_fd(self):
        if self._fd is None:
            raise RuntimeError(f"{self.path} is unavailable; build/load visiong_hw.ko or use existing fallbacks")
        return self._fd

    @classmethod
    def _block_offset(cls, block):
        if isinstance(block, int):
            return int(block), 0
        token = str(block).strip().lower().replace("-", "_")
        match = re.fullmatch(r"pwm(\d+)", token)
        if match:
            channel = int(match.group(1))
            if 0 <= channel <= 3:
                return cls._BLOCKS["pwm0_3"], channel * 0x10
            if 4 <= channel <= 7:
                return cls._BLOCKS["pwm4_7"], (channel - 4) * 0x10
            if 8 <= channel <= 11:
                return cls._BLOCKS["pwm8_11"], (channel - 8) * 0x10
        if token not in cls._BLOCKS:
            raise ValueError(f"unknown visiong-hw register block: {block!r}")
        return cls._BLOCKS[token], 0

    def caps(self):
        import fcntl

        fd = self._require_fd()
        buf = bytearray(self._CAPS_STRUCT.size)
        self._CAPS_STRUCT.pack_into(buf, 0, self._CAPS_STRUCT.size, *([0] * 15))
        fcntl.ioctl(fd, self._GET_CAPS, buf, True)
        values = self._CAPS_STRUCT.unpack(bytes(buf))
        feature_flags = values[3]
        return {
            "size": values[0],
            "abi_version": values[1],
            "driver_version": values[2],
            "feature_flags": feature_flags,
            "reg_access": bool(feature_flags & self.FEATURE_REG_ACCESS),
            "pin_session": bool(feature_flags & self.FEATURE_PIN_SESSION),
            "gpio_irq": bool(feature_flags & self.FEATURE_GPIO_IRQ),
            "dma_buffer": bool(feature_flags & self.FEATURE_DMA_BUFFER),
            "dma_memcpy": bool(feature_flags & self.FEATURE_DMA_MEMCPY),
            "spi_display": bool(feature_flags & self.FEATURE_SPI_DISPLAY),
            "dma_fill": bool(feature_flags & self.FEATURE_DMA_FILL),
            "spi_reg": bool(feature_flags & self.FEATURE_SPI_REG),
            "chip_id": values[4],
            "max_dma_bytes": values[5],
            "max_transfer_bytes": values[6],
        }

    def read32(self, block, offset):
        import fcntl

        block_id, base_offset = self._block_offset(block)
        buf = bytearray(self._REG_STRUCT.size)
        self._REG_STRUCT.pack_into(buf, 0, self._REG_STRUCT.size, block_id, base_offset + int(offset), 0, 0, 0, 0, 0)
        fcntl.ioctl(self._require_fd(), self._REG_READ, buf, True)
        return self._REG_STRUCT.unpack(bytes(buf))[3]

    def write32(self, block, offset, value, mask=0xFFFFFFFF, *, hiword=False):
        import fcntl

        block_id, base_offset = self._block_offset(block)
        flags = self.REG_FLAG_HIWORD_UPDATE if hiword else 0
        buf = self._REG_STRUCT.pack(
            self._REG_STRUCT.size,
            block_id,
            base_offset + int(offset),
            int(value) & 0xFFFFFFFF,
            int(mask) & 0xFFFFFFFF,
            flags,
            0,
            0,
        )
        fcntl.ioctl(self._require_fd(), self._REG_WRITE, buf)
        return None

    def update32(self, block, offset, mask, value):
        return self.write32(block, offset, value, mask=mask)

    def reg(self, block):
        block_id, base_offset = self._block_offset(block)
        return HWReg(self, block_id, base_offset)

    @classmethod
    def _dma_direction(cls, direction):
        if isinstance(direction, int):
            return int(direction)
        text = str(direction or "bidirectional").strip().lower().replace("-", "_")
        if text in ("to_device", "cpu_to_device", "write", "out", "tx"):
            return cls.DMA_SYNC_TO_DEVICE
        if text in ("from_device", "device_to_cpu", "read", "in", "rx"):
            return cls.DMA_SYNC_FROM_DEVICE
        if text in ("bidirectional", "bidir", "both"):
            return cls.DMA_SYNC_BIDIRECTIONAL
        raise ValueError("DMA direction must be to_device, from_device, or bidirectional")

    @classmethod
    def _irq_edge(cls, edge):
        if isinstance(edge, int):
            return int(edge)
        text = str(edge or "both").strip().lower()
        if text in ("rising", "rise"):
            return cls.IRQ_EDGE_RISING
        if text in ("falling", "fall"):
            return cls.IRQ_EDGE_FALLING
        if text in ("both", "any"):
            return cls.IRQ_EDGE_BOTH
        raise ValueError("IRQ edge must be rising, falling, or both")

    def dma_alloc(self, size, *, write_combine=False):
        import fcntl

        flags = self.DMA_ALLOC_WRITE_COMBINE if write_combine else 0
        buf = bytearray(self._DMA_ALLOC_STRUCT.size)
        self._DMA_ALLOC_STRUCT.pack_into(buf, 0, self._DMA_ALLOC_STRUCT.size, int(size), flags, -1, 0, 0, 0, 0)
        fcntl.ioctl(self._require_fd(), self._DMA_ALLOC, buf, True)
        values = self._DMA_ALLOC_STRUCT.unpack(bytes(buf))
        fd = values[3]
        actual_size = values[1]
        if fd < 0:
            raise OSError("visiong-hw DMA allocation did not return a valid fd")
        return HWDmaBuffer(self, actual_size, fd)

    def dma_sync_for_cpu(self, buffer, direction="from_device", *, offset=0, size=0):
        return self._dma_sync(buffer, direction, self.DMA_SYNC_START, offset=offset, size=size)

    def dma_sync_for_device(self, buffer, direction="to_device", *, offset=0, size=0):
        return self._dma_sync(buffer, direction, self.DMA_SYNC_END, offset=offset, size=size)

    def _dma_sync(self, buffer, direction, flags, *, offset=0, size=0):
        import fcntl

        fd = getattr(buffer, "fd", buffer)
        data = self._DMA_SYNC_STRUCT.pack(
            self._DMA_SYNC_STRUCT.size,
            int(fd),
            self._dma_direction(direction),
            int(flags),
            int(offset),
            int(size),
            0,
            0,
            0,
            0,
        )
        fcntl.ioctl(self._require_fd(), self._DMA_SYNC, data)
        return None

    def dma_memcpy(self, dst, src, size=None, *, dst_offset=0, src_offset=0, wait=True):
        import fcntl

        dst_fd = getattr(dst, "fd", dst)
        src_fd = getattr(src, "fd", src)
        if size is None:
            dst_size = getattr(dst, "size", None)
            src_size = getattr(src, "size", None)
            if dst_size is None or src_size is None:
                raise ValueError("dma_memcpy size is required when dst/src are raw fds")
            size = min(int(dst_size) - int(dst_offset), int(src_size) - int(src_offset))
        if int(size) <= 0:
            raise ValueError("dma_memcpy size must be positive")
        flags = 0 if wait else self.DMA_MEMCPY_ASYNC
        buf = bytearray(self._DMA_MEMCPY_STRUCT.size)
        self._DMA_MEMCPY_STRUCT.pack_into(
            buf,
            0,
            self._DMA_MEMCPY_STRUCT.size,
            int(dst_fd),
            int(src_fd),
            int(dst_offset),
            int(src_offset),
            int(size),
            flags,
            0,
            0,
            0,
            0,
            0,
        )
        fcntl.ioctl(self._require_fd(), self._DMA_MEMCPY, buf, True)
        values = self._DMA_MEMCPY_STRUCT.unpack(bytes(buf))
        status = values[7]
        handle = values[8]
        if wait:
            if status != self.DMA_MEMCPY_STATUS_DONE:
                raise OSError(f"visiong-hw DMA memcpy failed with status {status}")
            return int(size)
        if status != self.DMA_MEMCPY_STATUS_PENDING or not handle:
            raise OSError(f"visiong-hw DMA memcpy failed with status {status}")
        return HWDmaCopy(self, handle, int(size))

    def dma_submit_memcpy(self, dst, src, size=None, *, dst_offset=0, src_offset=0):
        return self.dma_memcpy(dst, src, size, dst_offset=dst_offset, src_offset=src_offset, wait=False)

    def dma_fill(self, buffer, value=0, size=None, *, offset=0):
        import fcntl

        fd = getattr(buffer, "fd", buffer)
        if size is None:
            buffer_size = getattr(buffer, "size", None)
            if buffer_size is None:
                raise ValueError("dma_fill size is required when buffer is a raw fd")
            size = int(buffer_size) - int(offset)
        if int(size) <= 0:
            raise ValueError("dma_fill size must be positive")
        buf = bytearray(self._DMA_FILL_STRUCT.size)
        self._DMA_FILL_STRUCT.pack_into(
            buf,
            0,
            self._DMA_FILL_STRUCT.size,
            int(fd),
            int(offset),
            int(size),
            int(value) & 0xFF,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )
        fcntl.ioctl(self._require_fd(), self._DMA_FILL, buf, True)
        values = self._DMA_FILL_STRUCT.unpack(bytes(buf))
        status = values[6]
        if status != self.DMA_MEMCPY_STATUS_DONE:
            raise OSError(f"visiong-hw DMA fill failed with status {status}")
        return int(size)

    def irq(self, pin=None, *, bank=None, pin_index=None, edge="both"):
        import fcntl

        if pin is not None:
            if isinstance(pin, (list, tuple)) and len(pin) == 2:
                bank, pin_index = int(pin[0]), int(pin[1])
            else:
                bank, pin_index = _parse_gpio_pin(pin)
        if bank is None or pin_index is None:
            raise ValueError("irq() requires pin='GPIO1_C3' or bank=1, pin_index=19")
        buf = bytearray(self._IRQ_REQ_STRUCT.size)
        self._IRQ_REQ_STRUCT.pack_into(
            buf,
            0,
            self._IRQ_REQ_STRUCT.size,
            int(bank),
            int(pin_index),
            self._irq_edge(edge),
            0,
            0,
            0,
            0,
            0,
            0,
        )
        fcntl.ioctl(self._require_fd(), self._IRQ_REQUEST, buf, True)
        values = self._IRQ_REQ_STRUCT.unpack(bytes(buf))
        return HWIRQ(self, values[5], int(bank), int(pin_index), edge)

    def spi_open(self, bus="spi0.0", *, chip_select=None, speed_hz=24_000_000, width=0, height=0, rotation=0):
        import fcntl

        if isinstance(bus, str):
            match = re.fullmatch(r"spi(\d+)(?:\.(\d+))?", bus.strip().lower())
            if not match:
                raise ValueError("spi_open bus must be like 'spi0.0' or an integer bus id")
            bus_id = int(match.group(1))
            if chip_select is None:
                chip_select = int(match.group(2) or 0)
        else:
            bus_id = int(bus)
            if chip_select is None:
                chip_select = 0
        buf = bytearray(self._SPI_OPEN_STRUCT.size)
        self._SPI_OPEN_STRUCT.pack_into(
            buf,
            0,
            self._SPI_OPEN_STRUCT.size,
            bus_id,
            int(chip_select),
            int(width),
            int(height),
            int(rotation),
            int(speed_hz),
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )
        fcntl.ioctl(self._require_fd(), self._SPI_DISPLAY_OPEN, buf, True)
        values = self._SPI_OPEN_STRUCT.unpack(bytes(buf))
        handle = values[8]
        if not handle:
            raise OSError("visiong-hw SPI open did not return a handle")
        return HWSPI(self, handle, bus_id, int(chip_select), int(speed_hz))

    def _spi_reg_transfer(
        self,
        bus,
        chip_select,
        tx_data=b"",
        rx_len=0,
        *,
        tx_only=False,
        speed_hz=50_000_000,
        source_clock_hz=200_000_000,
        mode=0,
        bits_per_word=8,
        dummy=0xFF,
    ):
        import ctypes
        import fcntl

        tx_data = bytes(tx_data or b"")
        rx_len = int(rx_len)
        flags = self.SPI_REG_TX_ONLY if tx_only else 0
        keepalive = []
        tx_ptr = 0
        rx_ptr = 0
        if tx_data:
            tx_buf = ctypes.create_string_buffer(tx_data)
            keepalive.append(tx_buf)
            tx_ptr = ctypes.addressof(tx_buf)
        rx_data = bytearray(rx_len)
        if rx_len:
            rx_buf = (ctypes.c_uint8 * rx_len).from_buffer(rx_data)
            keepalive.append(rx_buf)
            rx_ptr = ctypes.addressof(rx_buf)
        buf = bytearray(self._SPI_REG_TRANSFER_STRUCT.size)
        self._SPI_REG_TRANSFER_STRUCT.pack_into(
            buf,
            0,
            self._SPI_REG_TRANSFER_STRUCT.size,
            int(bus),
            int(chip_select),
            int(speed_hz),
            int(source_clock_hz),
            int(mode),
            int(bits_per_word),
            flags,
            tx_ptr,
            rx_ptr,
            len(tx_data),
            rx_len,
            0,
            0,
            int(dummy) & 0xFF,
            0,
            0,
            0,
        )
        fcntl.ioctl(self._require_fd(), self._SPI_REG_TRANSFER, buf, True)
        values = self._SPI_REG_TRANSFER_STRUCT.unpack(bytes(buf))
        status = values[12]
        transferred = values[13]
        if status != self.DMA_MEMCPY_STATUS_DONE:
            raise OSError(f"visiong-hw SPI register transfer failed with status {status}")
        return int(transferred) if tx_only else bytes(rx_data[:rx_len])

    def _spi_reg_release(self, bus):
        import fcntl

        buf = self._SPI_REG_RELEASE_STRUCT.pack(
            self._SPI_REG_RELEASE_STRUCT.size,
            int(bus),
            0,
            0,
            0,
            0,
            0,
            0,
        )
        fcntl.ioctl(self._require_fd(), self._SPI_REG_RELEASE, buf)
        return None

    def close(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        self._require_fd()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __repr__(self):
        state = "open" if self._fd is not None else "unavailable"
        return f"HW(path='{self.path}', state='{state}')"


class HWDmaBuffer:
    def __init__(self, hw, size, fd):
        self.hw = hw
        self.size = int(size)
        self.fd = int(fd)

    def mmap(self, *, access=None):
        import mmap

        if access is None:
            access = mmap.ACCESS_WRITE
        return mmap.mmap(self.fd, self.size, access=access)

    def sync_for_cpu(self, direction="from_device", *, offset=0, size=0):
        return self.hw.dma_sync_for_cpu(self, direction, offset=offset, size=size)

    def sync_for_device(self, direction="to_device", *, offset=0, size=0):
        return self.hw.dma_sync_for_device(self, direction, offset=offset, size=size)

    def fill(self, value=0, size=None, *, offset=0):
        return self.hw.dma_fill(self, value=value, size=size, offset=offset)

    def close(self):
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __repr__(self):
        return f"HWDmaBuffer(size={self.size}, fd={self.fd})"


class HWDmaCopy:
    def __init__(self, hw, handle, size):
        self.hw = hw
        self.handle = int(handle)
        self.size = int(size)
        self.timestamp_ns = 0
        self._done = False

    def wait(self, timeout_ms=-1):
        import fcntl

        if self._done:
            return True
        buf = bytearray(self.hw._WAIT_STRUCT.size)
        self.hw._WAIT_STRUCT.pack_into(
            buf,
            0,
            self.hw._WAIT_STRUCT.size,
            self.handle,
            int(timeout_ms),
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )
        fcntl.ioctl(self.hw._require_fd(), self.hw._DMA_WAIT, buf, True)
        values = self.hw._WAIT_STRUCT.unpack(bytes(buf))
        status = values[3]
        self.timestamp_ns = int(values[4]) | (int(values[5]) << 32)
        if status == self.hw.DMA_MEMCPY_STATUS_TIMEOUT:
            return False
        if status != self.hw.DMA_MEMCPY_STATUS_DONE:
            raise OSError(f"visiong-hw DMA memcpy wait failed with status {status}")
        self._done = True
        return True

    def done(self):
        return self._done

    def __del__(self):
        if not self._done:
            try:
                self.wait(0)
            except Exception:
                pass

    def __repr__(self):
        state = "done" if self._done else "pending"
        return f"HWDmaCopy(handle={self.handle}, size={self.size}, state='{state}')"


class HWSPITransfer:
    def __init__(self, spi, handle, size):
        self.spi = spi
        self.hw = spi.hw
        self.handle = int(handle)
        self.size = int(size)
        self.timestamp_ns = 0
        self._done = False

    def wait(self, timeout_ms=-1):
        import fcntl

        if self._done:
            return True
        buf = bytearray(self.hw._WAIT_STRUCT.size)
        self.hw._WAIT_STRUCT.pack_into(
            buf,
            0,
            self.hw._WAIT_STRUCT.size,
            self.handle,
            int(timeout_ms),
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )
        fcntl.ioctl(self.hw._require_fd(), self.hw._SPI_DISPLAY_WAIT, buf, True)
        values = self.hw._WAIT_STRUCT.unpack(bytes(buf))
        status = values[3]
        self.timestamp_ns = int(values[4]) | (int(values[5]) << 32)
        if status == self.hw.DMA_MEMCPY_STATUS_TIMEOUT:
            return False
        if status != self.hw.DMA_MEMCPY_STATUS_DONE:
            raise OSError(f"visiong-hw SPI transfer failed with status {status}")
        self._done = True
        return True

    def done(self):
        return self._done

    def __del__(self):
        if not self._done:
            try:
                self.wait(0)
            except Exception:
                pass

    def __repr__(self):
        state = "done" if self._done else "pending"
        return f"HWSPITransfer(handle={self.handle}, size={self.size}, state='{state}')"


class HWSPI:
    def __init__(self, hw, handle, bus, chip_select, speed_hz):
        self.hw = hw
        self.handle = int(handle)
        self.bus = int(bus)
        self.chip_select = int(chip_select)
        self.speed_hz = int(speed_hz)

    def submit_dma(self, buffer, size=None, *, offset=0, wait=False):
        import fcntl

        if self.handle <= 0:
            raise RuntimeError("HWSPI is closed")
        fd = getattr(buffer, "fd", buffer)
        if size is None:
            buffer_size = getattr(buffer, "size", None)
            if buffer_size is None:
                raise ValueError("submit_dma size is required when buffer is a raw fd")
            size = int(buffer_size) - int(offset)
        if int(size) <= 0:
            raise ValueError("submit_dma size must be positive")
        buf = bytearray(self.hw._SPI_SUBMIT_STRUCT.size)
        self.hw._SPI_SUBMIT_STRUCT.pack_into(
            buf,
            0,
            self.hw._SPI_SUBMIT_STRUCT.size,
            self.handle,
            int(fd),
            int(offset),
            int(size),
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )
        fcntl.ioctl(self.hw._require_fd(), self.hw._SPI_DISPLAY_SUBMIT, buf, True)
        values = self.hw._SPI_SUBMIT_STRUCT.unpack(bytes(buf))
        job_handle = values[12]
        if not job_handle:
            raise OSError("visiong-hw SPI submit did not return a job handle")
        transfer = HWSPITransfer(self, job_handle, int(size))
        if wait:
            transfer.wait()
        return transfer

    def write_dma(self, buffer, size=None, *, offset=0, wait=True):
        transfer = self.submit_dma(buffer, size=size, offset=offset, wait=wait)
        return int(size if size is not None else getattr(buffer, "size", 0) - int(offset)) if wait else transfer

    def close(self):
        import fcntl

        if self.handle <= 0:
            return None
        buf = self.hw._WAIT_STRUCT.pack(
            self.hw._WAIT_STRUCT.size,
            self.handle,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )
        fcntl.ioctl(self.hw._require_fd(), self.hw._SPI_DISPLAY_CLOSE, buf)
        self.handle = 0
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        return f"HWSPI(handle={self.handle}, bus='spi{self.bus}.{self.chip_select}', speed_hz={self.speed_hz})"


class HWIRQEvent:
    def __init__(self, sequence, timestamp_ns, bank, pin):
        self.sequence = int(sequence)
        self.timestamp_ns = int(timestamp_ns)
        self.bank = int(bank)
        self.pin = int(pin)

    def __bool__(self):
        return self.sequence > 0

    def __repr__(self):
        group = chr(ord("A") + self.pin // 8)
        return f"HWIRQEvent(sequence={self.sequence}, timestamp_ns={self.timestamp_ns}, pin='GPIO{self.bank}_{group}{self.pin % 8}')"


class HWIRQ:
    def __init__(self, hw, handle, bank, pin, edge):
        self.hw = hw
        self.handle = int(handle)
        self.bank = int(bank)
        self.pin = int(pin)
        self.edge = edge
        self.sequence = 0

    def wait(self, timeout_ms=-1):
        import fcntl

        if self.handle <= 0:
            raise RuntimeError("HWIRQ is closed")
        buf = bytearray(self.hw._WAIT_STRUCT.size)
        self.hw._WAIT_STRUCT.pack_into(
            buf,
            0,
            self.hw._WAIT_STRUCT.size,
            self.handle,
            int(timeout_ms),
            self.sequence,
            0,
            0,
            0,
            0,
            0,
            0,
        )
        fcntl.ioctl(self.hw._require_fd(), self.hw._IRQ_WAIT, buf, True)
        values = self.hw._WAIT_STRUCT.unpack(bytes(buf))
        sequence = values[3]
        timestamp_ns = int(values[4]) | (int(values[5]) << 32)
        if sequence == self.sequence and timestamp_ns == 0:
            return None
        self.sequence = sequence
        return HWIRQEvent(sequence, timestamp_ns, self.bank, self.pin)

    def __repr__(self):
        group = chr(ord("A") + self.pin // 8)
        return f"HWIRQ(handle={self.handle}, pin='GPIO{self.bank}_{group}{self.pin % 8}', edge='{self.edge}')"

    def close(self):
        import fcntl

        if self.handle <= 0:
            return None
        buf = self.hw._WAIT_STRUCT.pack(
            self.hw._WAIT_STRUCT.size,
            self.handle,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )
        fcntl.ioctl(self.hw._require_fd(), self.hw._IRQ_RELEASE, buf)
        self.handle = 0
        return None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


_PERIPHERAL_LOCKS_GUARD = threading.Lock()
_PERIPHERAL_LOCKS = {}


def _peripheral_lock(name):
    with _PERIPHERAL_LOCKS_GUARD:
        lock = _PERIPHERAL_LOCKS.get(name)
        if lock is None:
            lock = threading.RLock()
            _PERIPHERAL_LOCKS[name] = lock
        return lock


def _parse_bus_id(value, prefix):
    if value is None or value == "auto":
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    if text.isdigit():
        return int(text)
    if prefix == "uart" and text.startswith("serial"):
        prefix_pattern = r"(?:uart|serial)"
    else:
        prefix_pattern = re.escape(prefix)
    match = re.match(rf"^{prefix_pattern}(\d+)(?:\.\d+)?$", text)
    if match:
        return int(match.group(1))
    raise ValueError(f"invalid {prefix.upper()} id: {value!r}")


def _make_status(**kwargs):
    return type("PeripheralStatus", (), kwargs)()


def _cru_hiword_update(cru, offset, shift, width, value):
    mask = ((1 << width) - 1) << shift
    cru.write32(offset, (mask << 16) | ((int(value) << shift) & mask))


def _cru_ungate(cru, offset, bits):
    cru.write32(offset, int(bits) << 16)


def _infer_spi_from_pins(pinmux, pins, requested_bus=None, requested_cs=None):
    if len(pins) < 3:
        raise ValueError("SPI pins must include clk/sck, cs, and at least one of mosi/miso")
    pattern = re.compile(r"^(?P<group>spi(?P<bus>\d+)m\d+)-(?P<role>clk|sck|mosi|miso|cs(?P<cs>\d+))$")
    all_candidates = []
    for pin in pins:
        candidates = []
        for entry in pinmux.list_functions(pin):
            function = _normal_token(entry.function)
            group = _normal_token(entry.group)
            match = pattern.match(group)
            if not match:
                continue
            bus = int(match.group("bus"))
            if requested_bus is not None and bus != requested_bus:
                continue
            role = match.group("role")
            chip = int(match.group("cs")) if match.group("cs") is not None else None
            if chip is not None and requested_cs is not None and chip != requested_cs:
                continue
            if function and function != f"spi{bus}":
                continue
            candidates.append((match.group("group"), bus, role, chip))
        if not candidates:
            suffix = "" if requested_bus is None else f" for spi{requested_bus}"
            raise ValueError(f"{pin} has no SPI alternate function{suffix}")
        all_candidates.append(candidates)

    common_groups = {item[0] for item in all_candidates[0]}
    for candidates in all_candidates[1:]:
        common_groups &= {item[0] for item in candidates}
    if not common_groups:
        raise ValueError("SPI pins do not belong to one common spiXmY mux group")
    if len(common_groups) > 1:
        names = ", ".join(sorted(common_groups))
        raise ValueError(f"SPI pins are ambiguous across groups: {names}")

    group = next(iter(common_groups))
    roles = {}
    bus = None
    chip_select = requested_cs
    for candidates in all_candidates:
        item = next(value for value in candidates if value[0] == group)
        bus = item[1]
        role = "clk" if item[2] == "sck" else item[2]
        if role.startswith("cs"):
            chip_select = item[3]
            role = "cs"
        if role in roles:
            raise ValueError(f"duplicate SPI role {role!r} in pin list")
        roles[role] = True

    if "clk" not in roles or "cs" not in roles or ("mosi" not in roles and "miso" not in roles):
        raise ValueError("SPI pins must contain clk/sck, cs, and at least one of mosi/miso")
    return bus, 0 if chip_select is None else chip_select


def _infer_i2c_from_pins(pinmux, pins, requested_bus=None):
    if len(pins) < 2:
        raise ValueError("I2C pins must include scl and sda")
    pattern = re.compile(r"^(?P<group>i2c(?P<bus>\d+)m\d+)-(?P<role>scl|sda|xfer)$")
    all_candidates = []
    for pin in pins:
        candidates = []
        for entry in pinmux.list_functions(pin):
            function = _normal_token(entry.function)
            group = _normal_token(entry.group)
            match = pattern.match(group)
            if not match:
                continue
            bus = int(match.group("bus"))
            if requested_bus is not None and bus != requested_bus:
                continue
            if function and function != f"i2c{bus}":
                continue
            candidates.append((match.group("group"), bus, match.group("role")))
        if not candidates:
            suffix = "" if requested_bus is None else f" for i2c{requested_bus}"
            raise ValueError(f"{pin} has no I2C alternate function{suffix}")
        all_candidates.append(candidates)

    common_groups = {item[0] for item in all_candidates[0]}
    for candidates in all_candidates[1:]:
        common_groups &= {item[0] for item in candidates}
    if not common_groups:
        raise ValueError("I2C pins do not belong to one common i2cXmY mux group")
    if len(common_groups) > 1:
        names = ", ".join(sorted(common_groups))
        raise ValueError(f"I2C pins are ambiguous across groups: {names}")

    group = next(iter(common_groups))
    roles = {}
    bus = None
    for candidates in all_candidates:
        group_items = [value for value in candidates if value[0] == group]
        item = next((value for value in group_items if value[2] != "xfer"), group_items[0])
        bus = item[1]
        role = item[2]
        if role == "xfer":
            continue
        if role in roles:
            raise ValueError(f"duplicate I2C role {role!r} in pin list")
        roles[role] = True
    if "scl" not in roles or "sda" not in roles:
        raise ValueError("I2C pins must contain scl and sda from the same group")
    return bus


def _infer_uart_from_pins(pinmux, pins, requested_bus=None):
    if not pins:
        raise ValueError("UART pins must include at least tx or rx")
    pattern = re.compile(r"^(?P<group>uart(?P<bus>\d+)m\d+)-(?P<role>tx|rx|rts|cts|xfer)$")
    all_candidates = []
    for pin in pins:
        candidates = []
        for entry in pinmux.list_functions(pin):
            function = _normal_token(entry.function)
            group = _normal_token(entry.group)
            match = pattern.match(group)
            if not match:
                continue
            bus = int(match.group("bus"))
            if requested_bus is not None and bus != requested_bus:
                continue
            if function and function != f"uart{bus}":
                continue
            candidates.append((match.group("group"), bus, match.group("role")))
        if not candidates:
            suffix = "" if requested_bus is None else f" for uart{requested_bus}"
            raise ValueError(f"{pin} has no UART alternate function{suffix}")
        all_candidates.append(candidates)

    common_groups = {item[0] for item in all_candidates[0]}
    for candidates in all_candidates[1:]:
        common_groups &= {item[0] for item in candidates}
    if not common_groups:
        raise ValueError("UART pins do not belong to one common uartXmY mux group")
    if len(common_groups) > 1:
        names = ", ".join(sorted(common_groups))
        raise ValueError(f"UART pins are ambiguous across groups: {names}")

    group = next(iter(common_groups))
    roles = {}
    bus = None
    for candidates in all_candidates:
        group_items = [value for value in candidates if value[0] == group]
        item = next((value for value in group_items if value[2] != "xfer"), group_items[0])
        bus = item[1]
        role = item[2]
        if role == "xfer":
            role = "txrx"
        if role in roles:
            raise ValueError(f"duplicate UART role {role!r} in pin list")
        roles[role] = True
    if not any(role in roles for role in ("tx", "rx", "txrx")):
        raise ValueError("UART pins must contain tx or rx from the same group")
    return bus


def _infer_pwm_from_pin(pinmux, pin, requested_channel=None):
    pattern = re.compile(r"^pwm(?P<channel>\d+)(?:ir)?m\d+$")
    candidates = []
    for entry in pinmux.list_functions(pin):
        function = _normal_token(entry.function)
        group = _normal_token(entry.group)
        match = pattern.match(group)
        if not match:
            continue
        channel = int(match.group("channel"))
        if requested_channel is not None and channel != requested_channel:
            continue
        if function and function != f"pwm{channel}":
            continue
        candidates.append(channel)
    if not candidates:
        suffix = "" if requested_channel is None else f" for pwm{requested_channel}"
        raise ValueError(f"{pin} has no PWM alternate function{suffix}")
    unique = sorted(set(candidates))
    if len(unique) > 1:
        names = ", ".join(f"pwm{item}" for item in unique)
        raise ValueError(f"PWM pin is ambiguous across channels: {names}")
    return unique[0]


_GPIO_PIN_PATTERN = re.compile(r"^GPIO(?P<bank>\d+)_?(?P<group>[A-Da-d])(?P<index>[0-7])$")


def _parse_gpio_pin(pin):
    text = _pin_name(pin).strip()
    match = _GPIO_PIN_PATTERN.match(text)
    if not match:
        raise ValueError(f"invalid GPIO pin name: {pin!r}; expected GPIO1_C3")
    bank = int(match.group("bank"))
    pin_index = (ord(match.group("group").upper()) - ord("A")) * 8 + int(match.group("index"))
    if bank < 0 or bank > 4:
        raise ValueError(f"GPIO bank out of range for RV1103/RV1106: {bank}")
    return bank, pin_index


def _gpio_half_offset(low_offset, pin):
    return low_offset if pin < 16 else low_offset + 4, pin & 0x0F


class Pin:
    IN = "input"
    OUT = "output"
    ALT = "alt"
    IRQ_RISING = 0x01
    IRQ_FALLING = 0x02
    IRQ_BOTH = IRQ_RISING | IRQ_FALLING
    PULL_UP = "pull_up"
    PULL_DOWN = "pull_down"
    PULL_NONE = "disable"
    OPEN_DRAIN = "open_drain"
    OPEN_SOURCE = "open_source"
    PUSH_PULL = "push_pull"

    def __init__(
        self,
        pin,
        mode=None,
        pull=None,
        value=0,
        function=None,
        drive=None,
        *,
        backend="auto",
        fast=None,
    ):
        self.id = pin
        self.backend = "reg" if fast is True else _backend_name(backend)
        self._pinmux = PinMux()
        self._requested = False
        self._reg = None
        self._bank = None
        self._pin = None
        self._fast_gpio = False
        self._direction = None
        self._irq_thread = None
        self._irq_stop = threading.Event()
        self._irq_pipe_r = None
        self._irq_pipe_w = None
        self._irq_handler = None
        self._last_irq_event = None
        self._irq_error = None
        if function is not None:
            self.init(mode=self.ALT, function=function)
        elif mode is not None:
            self.init(mode=mode, pull=pull, value=value, drive=drive)

    @staticmethod
    def _edge_from_trigger(trigger):
        if trigger is None:
            return "both"
        if isinstance(trigger, str):
            text = trigger.strip().lower()
            if text in ("rise", "rising"):
                return "rising"
            if text in ("fall", "falling"):
                return "falling"
            if text in ("both", "any"):
                return "both"
            if text in ("none", "off", "disable", "disabled"):
                return "none"
            raise ValueError("Pin IRQ trigger must be rising, falling, both, or none")
        value = int(trigger)
        rising = bool(value & Pin.IRQ_RISING)
        falling = bool(value & Pin.IRQ_FALLING)
        if rising and falling:
            return "both"
        if rising:
            return "rising"
        if falling:
            return "falling"
        return "none"

    def _close_reg_gpio(self):
        if self._reg is not None:
            self._reg.close()
            self._reg = None
        self._fast_gpio = False

    def _open_reg_gpio(self):
        if self._reg is None:
            self._bank, self._pin = _parse_gpio_pin(self.id)
            self._reg = Reg(f"gpio{self._bank}")
        return self._reg

    def _write_reg_gpio_bit(self, low_offset, value):
        if self._pin is None:
            self._open_reg_gpio()
        offset, bit = _gpio_half_offset(low_offset, self._pin)
        self._reg.write32(offset, (1 << (bit + 16)) | ((1 if value else 0) << bit))

    def _read_reg_gpio_bit(self):
        if self._pin is None:
            self._open_reg_gpio()
        if self._direction == self.OUT:
            offset, bit = _gpio_half_offset(0x00, self._pin)
            return 1 if (self._reg.read32(offset) & (1 << bit)) else 0
        return 1 if (self._reg.read32(0x70) & (1 << self._pin)) else 0

    def level(self):
        if self._fast_gpio:
            if self._pin is None:
                self._open_reg_gpio()
            return 1 if (self._reg.read32(0x70) & (1 << self._pin)) else 0
        return self.value()

    def _init_reg_gpio(self, mode=None, pull=None, value=0, drive=None):
        if self._requested:
            self._pinmux.gpio_release_line(self.id)
            self._requested = False
        self._pinmux.set_function(self.id, "gpio")
        if pull is not None:
            self._pinmux.set_pull(self.id, pull)
        if drive is not None:
            if isinstance(drive, int):
                self._pinmux.set_drive_strength(self.id, drive)
        self._open_reg_gpio()
        direction = self.OUT if mode == self.OUT else self.IN
        if direction == self.OUT:
            self._write_reg_gpio_bit(0x00, value)
            self._write_reg_gpio_bit(0x08, 1)
        else:
            self._write_reg_gpio_bit(0x08, 0)
        self._direction = direction
        self._fast_gpio = True
        return self

    def init(self, mode=None, pull=None, value=0, function=None, drive=None, edge=None, backend=None, fast=None):
        self._stop_irq_thread()
        if backend is not None:
            self.backend = _backend_name(backend)
        if fast is True:
            self.backend = "reg"
        if function is not None or mode == self.ALT:
            if function is None:
                raise ValueError("Pin ALT mode requires function='...'")
            if self._requested:
                self._pinmux.gpio_release_line(self.id)
                self._requested = False
            self._close_reg_gpio()
            self._pinmux.set_function(self.id, function)
            return self

        edge = self._edge_from_trigger(edge) if edge is not None else "none"
        if self.backend in ("reg", "register", "direct", "mmio") and edge == "none":
            return self._init_reg_gpio(mode=mode, pull=pull, value=value, drive=drive)

        self._close_reg_gpio()

        cfg = GpioLineConfig()
        cfg.direction = self.OUT if mode == self.OUT else self.IN
        cfg.bias = pull or "default"
        cfg.default_value = 1 if value else 0
        cfg.edge = edge
        if drive is not None:
            cfg.drive = drive
        cfg.consumer = "visiong-pin"
        self._pinmux.gpio_request_line(self.id, cfg)
        self._requested = True
        self._direction = cfg.direction
        return self

    def value(self, value=None):
        if self._fast_gpio:
            if value is None:
                return self._read_reg_gpio_bit()
            self._write_reg_gpio_bit(0x00, value)
            return None
        if value is None:
            return self._pinmux.gpio_get_value(self.id)
        self._pinmux.gpio_set_value(self.id, 1 if value else 0)
        return None

    def on(self):
        self.value(1)

    def off(self):
        self.value(0)

    high = on
    low = off

    def toggle(self):
        value = 0 if self.value() else 1
        self.value(value)
        return value

    def irq(self, handler=None, trigger=None, *, pull=None, hard=False):
        del hard
        edge = self._edge_from_trigger(trigger)
        self._stop_irq_thread()
        if self._requested:
            self._pinmux.gpio_release_line(self.id)
            self._requested = False
        self._close_reg_gpio()
        if edge == "none":
            self._irq_handler = None
            return self

        cfg = GpioLineConfig()
        cfg.direction = self.IN
        cfg.bias = pull or "default"
        cfg.edge = edge
        cfg.consumer = "visiong-pin-irq"
        if not self._pinmux.gpio_request_line(self.id, cfg):
            raise RuntimeError(f"failed to request GPIO IRQ line {self.id}")
        self._requested = True
        self._irq_handler = handler
        self._irq_error = None
        if handler is not None:
            self._irq_stop.clear()
            self._ensure_irq_pipe()
            self._irq_thread = threading.Thread(target=self._irq_loop, name=f"visiong-irq-{self.id}", daemon=True)
            self._irq_thread.start()
        return self

    def _irq_loop(self):
        try:
            while not self._irq_stop.is_set():
                if self._irq_pipe_r is not None and hasattr(self._pinmux, "gpio_wait_event_cancelable"):
                    event = self._pinmux.gpio_wait_event_cancelable(self.id, self._irq_pipe_r, -1)
                    if getattr(event, "cancelled", False) or self._irq_stop.is_set():
                        break
                    if not event or event.timed_out:
                        continue
                    self._last_irq_event = event
                else:
                    event = self.wait_irq(-1)
                    if self._irq_stop.is_set():
                        break
                    if not event:
                        continue
                handler = self._irq_handler
                if handler is not None:
                    handler(self)
        except Exception as exc:
            if not self._irq_stop.is_set():
                self._irq_error = exc

    def _stop_irq_thread(self):
        thread = self._irq_thread
        if thread is not None:
            self._irq_stop.set()
            self._signal_irq_stop()
            if threading.current_thread() is not thread:
                thread.join(timeout=2.0)
                if thread.is_alive():
                    raise RuntimeError(f"IRQ thread for {self.id} did not stop")
                self._close_irq_pipe()
            self._irq_thread = None
        else:
            self._signal_irq_stop()

    def _ensure_irq_pipe(self):
        if self._irq_pipe_r is not None and self._irq_pipe_w is not None:
            return
        flags = getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
        if hasattr(os, "pipe2"):
            read_fd, write_fd = os.pipe2(flags)
        else:
            read_fd, write_fd = os.pipe()
            try:
                os.set_inheritable(read_fd, False)
                os.set_inheritable(write_fd, False)
            except Exception:
                pass
            try:
                os.set_blocking(read_fd, False)
                os.set_blocking(write_fd, False)
            except Exception:
                pass
        self._irq_pipe_r = read_fd
        self._irq_pipe_w = write_fd

    def _signal_irq_stop(self):
        if self._irq_pipe_w is None:
            return
        try:
            os.write(self._irq_pipe_w, b"\0")
        except BlockingIOError:
            pass
        except OSError:
            pass

    def _close_irq_pipe(self):
        for attr in ("_irq_pipe_r", "_irq_pipe_w"):
            fd = getattr(self, attr)
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
                setattr(self, attr, None)

    def irq_error(self):
        return self._irq_error

    def wait_irq(self, timeout_ms=-1):
        event = self._pinmux.gpio_wait_event(self.id, int(timeout_ms))
        if not event or event.timed_out:
            return None
        self._last_irq_event = event
        return event

    def irq_event(self):
        return self._last_irq_event

    def deinit(self):
        self._stop_irq_thread()
        if self._requested:
            self._pinmux.gpio_release_line(self.id)
            self._requested = False
        self._close_irq_pipe()
        self._close_reg_gpio()
        self._pinmux.close()

    close = deinit

    def __repr__(self):
        backend = "reg" if self._fast_gpio else "gpiochip"
        return f"Pin('{self.id}', backend='{backend}')"


def _write_sysfs(path, value):
    try:
        with open(path, "w", encoding="ascii") as f:
            f.write(str(value))
        return True
    except OSError:
        return False


def _driver_name(link_path):
    try:
        return os.path.basename(os.path.realpath(link_path))
    except OSError:
        return ""


def _wait_path(path, attempts=20, delay=0.01):
    for _ in range(int(attempts)):
        if os.path.exists(path):
            return True
        time.sleep(float(delay))
    return os.path.exists(path)


def _spi_child_name(bus, chip_select):
    return f"spi{int(bus)}.{int(chip_select)}"


def _spi_platform_device(bus):
    return {0: "ff500000.spi", 1: "ff510000.spi"}.get(int(bus), "")


class SPI:
    MSB = 0
    LSB = 1

    def __init__(
        self,
        id=None,
        *,
        baudrate=50_000_000,
        polarity=0,
        phase=0,
        bits=8,
        clk=None,
        sck=None,
        mosi=None,
        miso=None,
        cs=None,
        pins=None,
        chip_select=None,
        bind=False,
        backend="auto",
        source_clock_hz=200_000_000,
        dummy=0xFF,
    ):
        self.id = id
        self.baudrate = baudrate
        self.polarity = polarity
        self.phase = phase
        self.bits = bits
        self.backend = _backend_name(backend)
        self.source_clock_hz = source_clock_hz
        self.dummy = int(dummy) & 0xFF
        self._pinmux = PinMux()
        self._fd = None
        self._reg = None
        self._cru = None
        self._fifo_len_cached = None
        self._hw_reg = None
        self._hw_reg_unavailable = False
        self._reg_backend = False
        self._released_spi_child = ""
        self._released_spi_driver = ""
        self._released_spi_driver_path = ""
        self._released_platform_device = ""
        self._released_platform_driver = ""
        self._released_platform_driver_path = ""

        pin_list = []
        _append_pin_names(pin_list, pins)
        for pin in (clk if clk is not None else sck, mosi, miso, cs):
            _append_pin_names(pin_list, pin)

        requested_bus = _parse_bus_id(id, "spi")
        if pin_list:
            bus, inferred_cs = _infer_spi_from_pins(self._pinmux, pin_list, requested_bus, chip_select)
            chip_select = inferred_cs
            self.id = bus
            self.status = self._pinmux.spi(bus, pin_list, chip_select=chip_select, bind_spidev=bind)
        elif requested_bus is not None:
            chip_select = 0 if chip_select is None else int(chip_select)
            self.id = requested_bus
            self.status = _make_status(
                ok=False,
                bus=requested_bus,
                chip_select=chip_select,
                device=f"spi{requested_bus}.{chip_select}",
                dev_path=f"/dev/spidev{requested_bus}.{chip_select}",
                group="",
                pins=[],
            )
        else:
            raise ValueError("SPI requires pins when id is omitted; use SPI(clk=..., mosi=..., cs=...)")
        self.path = self.status.dev_path
        self._lock = _peripheral_lock(f"spi{self.status.bus}")
        if self.backend in ("spidev", "linux", "dev") or (
            bind and self.backend == "auto" and os.path.exists(self.path)
        ):
            self._open_spidev()
        elif self.backend in ("auto", "reg", "register", "direct"):
            self._reg_backend = True
        else:
            raise ValueError("SPI backend must be auto, reg, register, direct, spidev, or linux")

    def _release_reg_drivers(self):
        child = _spi_child_name(self.status.bus, self.status.chip_select)
        child_link = f"/sys/bus/spi/devices/{child}/driver"
        child_driver = _driver_name(child_link)
        if child_driver and not self._released_spi_child:
            if _write_sysfs(f"/sys/bus/spi/drivers/{child_driver}/unbind", child):
                self._released_spi_child = child
                self._released_spi_driver = child_driver
                self._released_spi_driver_path = f"/sys/bus/spi/drivers/{child_driver}"
                print(
                    f"[SPI] Warning: released {child} from SPI driver {child_driver} for register SPI.",
                    file=sys.stderr,
                )
            else:
                print(
                    f"[SPI] Warning: failed to release {child} from SPI driver {child_driver}; continuing.",
                    file=sys.stderr,
                )

        platform = _spi_platform_device(self.status.bus)
        platform_driver = _driver_name(f"/sys/bus/platform/devices/{platform}/driver") if platform else ""
        if platform_driver and not self._released_platform_device:
            if _write_sysfs(f"/sys/bus/platform/drivers/{platform_driver}/unbind", platform):
                self._released_platform_device = platform
                self._released_platform_driver = platform_driver
                self._released_platform_driver_path = f"/sys/bus/platform/drivers/{platform_driver}"
                print(
                    f"[SPI] Warning: released platform SPI device {platform} from driver {platform_driver} "
                    "for register SPI DMA.",
                    file=sys.stderr,
                )
            else:
                print(
                    f"[SPI] Warning: failed to release platform SPI device {platform} from driver "
                    f"{platform_driver}; continuing.",
                    file=sys.stderr,
                )

    def _restore_reg_drivers(self):
        if self._hw_reg is not None and self._hw_reg.available():
            try:
                self._hw_reg._spi_reg_release(self.status.bus)
            except OSError:
                pass

        if self._released_platform_device:
            current = _driver_name(f"/sys/bus/platform/devices/{self._released_platform_device}/driver")
            if current != self._released_platform_driver:
                if current:
                    _write_sysfs(f"/sys/bus/platform/drivers/{current}/unbind", self._released_platform_device)
                if not _write_sysfs(
                    os.path.join(self._released_platform_driver_path, "bind"),
                    self._released_platform_device,
                ):
                    print(
                        f"[SPI] Warning: failed to restore platform SPI device "
                        f"{self._released_platform_device} to driver {self._released_platform_driver}.",
                        file=sys.stderr,
                    )
                else:
                    _wait_path(f"/sys/bus/spi/devices/{_spi_child_name(self.status.bus, self.status.chip_select)}")
            self._released_platform_device = ""
            self._released_platform_driver = ""
            self._released_platform_driver_path = ""

        if self._released_spi_child:
            current = _driver_name(f"/sys/bus/spi/devices/{self._released_spi_child}/driver")
            if current != self._released_spi_driver:
                if current:
                    _write_sysfs(f"/sys/bus/spi/drivers/{current}/unbind", self._released_spi_child)
                override_path = f"/sys/bus/spi/devices/{self._released_spi_child}/driver_override"
                _write_sysfs(override_path, self._released_spi_driver + "\n")
                if not _write_sysfs(os.path.join(self._released_spi_driver_path, "bind"), self._released_spi_child):
                    print(
                        f"[SPI] Warning: failed to restore {self._released_spi_child} to SPI driver "
                        f"{self._released_spi_driver}.",
                        file=sys.stderr,
                    )
                else:
                    _write_sysfs(override_path, "\n")
            self._released_spi_child = ""
            self._released_spi_driver = ""
            self._released_spi_driver_path = ""

    def _open_spidev(self):
        import fcntl
        import struct

        self._fd = os.open(self.path, os.O_RDWR | getattr(os, "O_CLOEXEC", 0))
        mode = (1 if self.phase else 0) | (2 if self.polarity else 0)
        fcntl.ioctl(self._fd, 0x40016B01, struct.pack("B", mode))
        fcntl.ioctl(self._fd, 0x40016B03, struct.pack("B", self.bits))
        fcntl.ioctl(self._fd, 0x40046B04, struct.pack("I", self.baudrate))

    def init(self, *, baudrate=None, polarity=None, phase=None, bits=None):
        if baudrate is not None:
            self.baudrate = baudrate
        if polarity is not None:
            self.polarity = polarity
        if phase is not None:
            self.phase = phase
        if bits is not None:
            self.bits = bits
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        if not self._reg_backend and os.path.exists(self.path):
            self._open_spidev()
        return self

    def _enable_register_clock(self):
        if self._cru is None:
            self._cru = Reg("cru")
        bus = int(self.status.bus)
        if bus == 0:
            _cru_hiword_update(self._cru, 0x1A000 + 0x300, 12, 2, 0)
            _cru_ungate(self._cru, 0x1A000 + 0x800 + 0x04, (1 << 2) | (1 << 3) | (1 << 4))
        elif bus == 1:
            _cru_hiword_update(self._cru, 0x12000 + 0x300 + 0x18, 3, 2, 0)
            _cru_ungate(self._cru, 0x12000 + 0x800 + 0x0C, (1 << 6) | (1 << 7))

    def _ensure_legacy_reg(self):
        if self._reg is not None:
            return
        self._reg = Reg(f"spi{self.status.bus}")
        self._enable_register_clock()
        self._fifo_len_cached = None
        self._fifo_len_cached = self._fifo_len()

    def write(self, data):
        if self._fd is None and self._reg_backend:
            with self._lock:
                return self._transfer_reg(bytes(data), 0, tx_only=True)
        if self._fd is None:
            raise RuntimeError(f"{self.path} is not open; use backend='reg' or bind=True for SPI.write()")
        return os.write(self._fd, bytes(data))

    def _fifo_len(self):
        if self._fifo_len_cached is not None:
            return self._fifo_len_cached
        try:
            version = self._reg.read32(0x48)
        except Exception:
            return 64
        return 64 if version in (0x05EC0002, 0x00110002) else 32

    def _transfer_reg_fast(self, tx_data, rx_len=0, *, tx_only=False):
        if self._hw_reg_unavailable:
            return None
        try:
            if self._hw_reg is None:
                self._hw_reg = HW(required=False)
            if not self._hw_reg.available():
                self._hw_reg_unavailable = True
                return None
            mode = (1 if self.phase else 0) | (2 if self.polarity else 0)
            return self._hw_reg._spi_reg_transfer(
                int(self.status.bus),
                int(self.status.chip_select),
                tx_data,
                rx_len,
                tx_only=tx_only,
                speed_hz=int(self.baudrate),
                source_clock_hz=int(self.source_clock_hz),
                mode=mode,
                bits_per_word=int(self.bits),
                dummy=self.dummy,
            )
        except OSError as exc:
            if getattr(exc, "errno", None) in (19, 25, 38, 45, 95):
                self._hw_reg_unavailable = True
                return None
            raise

    def _transfer_reg(self, tx_data, rx_len=0, *, tx_only=False):
        if self.bits != 8:
            raise ValueError("register SPI currently supports bits=8 only")
        tx_data = bytes(tx_data)
        rx_len = int(rx_len)
        if rx_len < 0:
            raise ValueError("rx_len must be >= 0")
        if not tx_data and not rx_len:
            return b"" if not tx_only else 0
        self._release_reg_drivers()
        fast = self._transfer_reg_fast(tx_data, rx_len, tx_only=tx_only)
        if fast is not None:
            return fast

        self._ensure_legacy_reg()
        CTRL0 = 0x00
        CTRL1 = 0x04
        SSIENR = 0x08
        SER = 0x0C
        BAUDR = 0x10
        TXFTLR = 0x14
        RXFTLR = 0x18
        TXFLR = 0x1C
        RXFLR = 0x20
        SR = 0x24
        IMR = 0x2C
        ICR = 0x38
        DMACR = 0x3C
        TXDR = 0x400
        RXDR = 0x800

        SR_BUSY = 1 << 0
        XFM_TR = 0 << 18
        XFM_TO = 1 << 18
        CR0_BASE = 0x1 | (1 << 10) | (1 << 11) | (1 << 13)
        mode = ((1 if self.phase else 0) | (2 if self.polarity else 0)) << 6
        xfm = XFM_TO if tx_only and rx_len == 0 else XFM_TR

        div = max(2, (self.source_clock_hz + self.baudrate - 1) // self.baudrate)
        if div & 1:
            div += 1
        fifo = self._fifo_len()

        total_frames = len(tx_data) if tx_only else max(len(tx_data), rx_len)
        if not tx_only and len(tx_data) < total_frames:
            tx_data += bytes([self.dummy]) * (total_frames - len(tx_data))

        received = bytearray()
        written = 0
        for start in range(0, total_frames, 0xFFFF):
            chunk = tx_data[start : start + 0xFFFF]
            count = len(chunk)
            if count == 0:
                continue
            self._reg.write32(SSIENR, 0)
            self._reg.write32(IMR, 0)
            self._reg.write32(ICR, 0xFFFFFFFF)
            self._reg.write32(DMACR, 0)
            self._reg.write32(CTRL0, CR0_BASE | mode | xfm)
            self._reg.write32(CTRL1, count - 1)
            self._reg.write32(TXFTLR, max(1, fifo // 2))
            self._reg.write32(RXFTLR, 0)
            self._reg.write32(BAUDR, div)
            self._reg.write32(SER, 1 << self.status.chip_select)
            self._reg.write32(SSIENR, 1)
            tx_pos = 0
            rx_target = 0 if tx_only else count
            rx_pos = 0
            idle_guard = 0
            while tx_pos < count or rx_pos < rx_target:
                progressed = False
                level = self._reg.read32(TXFLR)
                if tx_pos < count and level < fifo:
                    writable = min(fifo - level, count - tx_pos)
                    self._reg.write8_repeat(TXDR, chunk[tx_pos : tx_pos + writable])
                    tx_pos += writable
                    written += writable
                    progressed = True

                if rx_pos < rx_target:
                    readable = min(self._reg.read32(RXFLR), rx_target - rx_pos)
                    if readable:
                        values = self._reg.read8_repeat(RXDR, readable)
                        remaining = max(0, rx_len - len(received))
                        if remaining:
                            received.extend(values[:remaining])
                    rx_pos += readable
                    progressed = progressed or readable > 0

                if progressed:
                    idle_guard = 0
                else:
                    idle_guard += 1
                    if idle_guard > 5_000_000:
                        self._reg.write32(SSIENR, 0)
                        self._reg.write32(SER, 0)
                        raise TimeoutError("register SPI transfer timed out")
            idle_guard = 0
            while self._reg.read32(SR) & SR_BUSY:
                idle_guard += 1
                if idle_guard > 5_000_000:
                    self._reg.write32(SSIENR, 0)
                    self._reg.write32(SER, 0)
                    raise TimeoutError("register SPI transfer did not become idle")
            self._reg.write32(SSIENR, 0)
            self._reg.write32(SER, 0)
        return written if tx_only else bytes(received[:rx_len])

    def read(self, nbytes, write=0xFF):
        if self._fd is None and self._reg_backend:
            with self._lock:
                return self._transfer_reg(bytes([int(write) & 0xFF]) * int(nbytes), int(nbytes))
        if self._fd is None:
            raise RuntimeError(f"{self.path} is not open")
        return os.read(self._fd, nbytes)

    def readinto(self, buf, write=0xFF):
        data = self.read(len(buf), write=write)
        buf[: len(data)] = data
        return len(data)

    def write_readinto(self, write_buf, read_buf):
        if self._fd is None and self._reg_backend:
            with self._lock:
                data = self._transfer_reg(bytes(write_buf), len(read_buf))
            read_buf[: len(data)] = data
            return None
        if self._fd is None:
            raise RuntimeError(f"{self.path} is not open")
        import fcntl
        import struct

        tx = bytes(write_buf)
        count = max(len(tx), len(read_buf))
        if len(tx) < count:
            tx += bytes([self.dummy]) * (count - len(tx))
        rx = ctypes.create_string_buffer(count)
        tx_buf = ctypes.create_string_buffer(tx)
        transfer = struct.pack(
            "QQIIHBBI",
            ctypes.addressof(tx_buf),
            ctypes.addressof(rx),
            count,
            self.baudrate,
            0,
            self.bits,
            0,
            0,
        )
        fcntl.ioctl(self._fd, 0x40206B00, transfer)
        read_buf[:] = rx.raw[: len(read_buf)]
        return None

    def transfer(self, data, read=None):
        rx_len = len(data) if read is None else int(read)
        out = bytearray(rx_len)
        self.write_readinto(data, out)
        return bytes(out)

    xfer = transfer

    def send(self, data):
        return self.write(data)

    def recv(self, nbytes, write=0xFF):
        return self.read(nbytes, write=write)

    def display(self, **kwargs):
        kwargs.setdefault("spi_bus", self.path)
        kwargs.setdefault("backend", "reg" if self._reg_backend else "auto")
        kwargs.setdefault("speed_hz", self.baudrate)
        return DisplaySPI(**kwargs)

    def mmio(self):
        return Reg(f"spi{self.status.bus}")

    def deinit(self):
        self._restore_reg_drivers()
        if self._hw_reg is not None:
            self._hw_reg.close()
            self._hw_reg = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        if self._reg is not None:
            self._reg.close()
            self._reg = None
            self._fifo_len_cached = None
        if self._cru is not None:
            self._cru.close()
            self._cru = None
        self._pinmux.close()

    close = deinit

    def __repr__(self):
        return f"SPI({self.id}, path='{self.path}', group='{self.status.group}')"


class UART:
    def __init__(
        self,
        id=None,
        baudrate=115200,
        *,
        tx=None,
        rx=None,
        rts=None,
        cts=None,
        pins=None,
        timeout=0,
        bind=False,
        backend="auto",
        bits=8,
        parity=None,
        stop=1,
        flow=0,
        loopback=False,
        source_clock_hz=24_000_000,
    ):
        self.id = id
        self.baudrate = baudrate
        self.timeout = timeout
        self.backend = _backend_name(backend)
        self.bits = int(bits)
        self.parity = parity
        self.stop = stop
        self.flow = flow
        self.loopback_mode = bool(loopback)
        self.source_clock_hz = int(source_clock_hz)
        self._pinmux = PinMux()
        self._fd = None
        self._reg = None
        self._cru = None

        pin_list = []
        _append_pin_names(pin_list, pins)
        for pin in (tx, rx, rts, cts):
            _append_pin_names(pin_list, pin)

        requested_bus = _parse_bus_id(id, "uart")
        if pin_list:
            bus = _infer_uart_from_pins(self._pinmux, pin_list, requested_bus)
            self.id = bus
            self.status = self._pinmux.uart(bus, pin_list, bind_driver=bind)
        elif requested_bus is not None:
            self.id = requested_bus
            self.status = _make_status(
                ok=False,
                bus=requested_bus,
                device=f"uart{requested_bus}",
                dev_path=f"/dev/ttyS{requested_bus}",
                group="",
                pins=[],
            )
        else:
            raise ValueError("UART requires pins when id is omitted; use UART(tx=..., rx=...)")
        self.path = self.status.dev_path
        self._lock = _peripheral_lock(f"uart{self.status.bus}")
        if self.backend in ("linux", "dev", "tty") or (bind and self.backend == "auto" and os.path.exists(self.path)):
            self._open_uart()
        elif self.backend in ("auto", "reg", "register", "direct"):
            self._reg = Reg(f"uart{self.status.bus}")
            self._enable_register_clock()
            self._open_register_uart()
        else:
            raise ValueError("UART backend must be auto, reg, register, direct, linux, dev, or tty")

    def _open_uart(self):
        import termios

        self._fd = os.open(self.path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0))
        attrs = termios.tcgetattr(self._fd)
        speed = getattr(termios, f"B{self.baudrate}", termios.B115200)
        attrs[0] = 0
        attrs[1] = 0
        attrs[2] = termios.CS8 | termios.CLOCAL | termios.CREAD
        attrs[3] = 0
        attrs[4] = speed
        attrs[5] = speed
        attrs[6][termios.VMIN] = 0
        attrs[6][termios.VTIME] = max(0, int(self.timeout * 10))
        termios.tcsetattr(self._fd, termios.TCSANOW, attrs)

    def _enable_register_clock(self):
        self._cru = Reg("cru")
        bus = int(self.status.bus)
        top = 0x10000
        source_gate = {
            0: [(top + 0x800, (1 << 11) | (1 << 13))],
            1: [(top + 0x800, (1 << 14) | (1 << 15)), (top + 0x804, 1 << 0)],
            2: [(top + 0x804, (1 << 1) | (1 << 3))],
            3: [(top + 0x804, (1 << 4) | (1 << 6))],
            4: [(top + 0x804, (1 << 7) | (1 << 9))],
            5: [(top + 0x804, (1 << 10) | (1 << 12))],
        }
        peri_gate = {
            0: [(0x12000 + 0x800 + 0x18, (1 << 11) | (1 << 14))],
            1: [(0x12000 + 0x800 + 0x18, 1 << 15), (0x12000 + 0x800 + 0x1C, 1 << 2)],
            2: [(0x12000 + 0x800 + 0x08, (1 << 3) | (1 << 6))],
            3: [(0x12000 + 0x800 + 0x08, (1 << 7) | (1 << 10))],
            4: [(0x12000 + 0x800 + 0x08, (1 << 11) | (1 << 14))],
            5: [(0x12000 + 0x800 + 0x08, 1 << 15), (0x12000 + 0x800 + 0x0C, 1 << 2)],
        }
        sclk_sel_offsets = {
            0: top + 0x31C,
            1: top + 0x324,
            2: top + 0x32C,
            3: top + 0x334,
            4: top + 0x33C,
            5: top + 0x344,
        }
        for offset, bits in source_gate.get(bus, ()):
            _cru_ungate(self._cru, offset, bits)
        if bus in sclk_sel_offsets:
            _cru_hiword_update(self._cru, sclk_sel_offsets[bus], 0, 2, 2)
            self.source_clock_hz = 24_000_000
        for offset, bits in peri_gate.get(bus, ()):
            _cru_ungate(self._cru, offset, bits)

    def _uart_lcr_value(self):
        if self.bits not in (5, 6, 7, 8):
            raise ValueError("UART bits must be 5, 6, 7, or 8")
        lcr = self.bits - 5
        if self.stop in (1.5, 2):
            lcr |= 1 << 2
        if self.parity is not None:
            parity = str(self.parity).lower()
            lcr |= 1 << 3
            if parity in ("even", "e", "0"):
                lcr |= 1 << 4
            elif parity not in ("odd", "o", "1"):
                raise ValueError("UART parity must be None, 'odd', or 'even'")
        return lcr

    def _open_register_uart(self):
        UART_DLL = 0x00
        UART_DLH = 0x04
        UART_IER = 0x04
        UART_FCR = 0x08
        UART_LCR = 0x0C
        UART_MCR = 0x10
        UART_USR = 0x7C
        UART_SRR = 0x88
        UART_SFE = 0x98
        UART_SRT = 0x9C
        UART_STET = 0xA0

        deadline = time.monotonic() + max(0.01, float(self.timeout) if self.timeout else 0.1)
        while (self._reg.read32(UART_USR) & 0x01) and time.monotonic() < deadline:
            pass

        lcr = self._uart_lcr_value()
        divisor = max(1, int((self.source_clock_hz + self.baudrate * 8) // (self.baudrate * 16)))
        divisor = min(divisor, 0xFFFF)
        self._reg.write8(UART_IER, 0)
        self._reg.write8(UART_SRR, 0x7)
        time.sleep(0.001)
        deadline = time.monotonic() + max(0.01, float(self.timeout) if self.timeout else 0.1)
        while (self._reg.read32(UART_USR) & 0x01) and time.monotonic() < deadline:
            pass
        self._reg.write8(UART_LCR, lcr | 0x80)
        self._reg.write8(UART_DLL, divisor & 0xFF)
        self._reg.write8(UART_DLH, (divisor >> 8) & 0xFF)
        for _ in range(20):
            self._reg.write8(UART_LCR, lcr)
            time.sleep(0.001)
            if (self._reg.read8(UART_LCR) & 0x80) == 0:
                break
        if self._reg.read8(UART_LCR) & 0x80:
            time.sleep(0.001)
            self._reg.write8(UART_LCR, lcr)
        self._reg.write8(UART_FCR, 0x07)
        self._reg.write8(UART_SFE, 1)
        self._reg.write8(UART_SRT, 0)
        self._reg.write8(UART_STET, 0)
        self._reg.write8(UART_MCR, self._mcr_value())
        if self._reg.read8(UART_LCR) & 0x80:
            time.sleep(0.005)
            self._reg.write8(UART_LCR, lcr)

    def _mcr_value(self):
        value = 0
        if self.flow:
            value |= 0x03 | (1 << 5)
        if self.loopback_mode:
            value |= 1 << 4
        return value

    def loopback(self, value=True):
        self.loopback_mode = bool(value)
        if self._reg is not None:
            self._reg.write8(0x10, self._mcr_value())
        return self

    def write(self, data):
        if self._fd is None and self._reg is not None:
            with self._lock:
                return self._write_reg(data)
        if self._fd is None:
            raise RuntimeError(f"{self.path} is not open")
        return os.write(self._fd, bytes(data))

    def _write_reg(self, data):
        UART_THR = 0x00
        UART_LSR = 0x14
        UART_TFL = 0x80
        LSR_THRE = 1 << 5
        FIFO_DEPTH = 64
        payload = bytes(data)
        written = 0
        deadline = time.monotonic() + max(0.01, float(self.timeout) if self.timeout else 1.0)
        while written < len(payload):
            level = min(FIFO_DEPTH, int(self._reg.read32(UART_TFL)))
            space = max(0, FIFO_DEPTH - level)
            if space == 0 and (self._reg.read32(UART_LSR) & LSR_THRE):
                space = FIFO_DEPTH
            if space == 0:
                if time.monotonic() > deadline:
                    raise TimeoutError("register UART TX FIFO did not drain")
                continue
            count = min(space, len(payload) - written)
            self._reg.write8_repeat(UART_THR, payload[written : written + count])
            written += count
            deadline = time.monotonic() + max(0.01, float(self.timeout) if self.timeout else 1.0)
        return written

    def read(self, nbytes=1):
        if self._fd is None and self._reg is not None:
            with self._lock:
                return self._read_reg(nbytes)
        if self._fd is None:
            raise RuntimeError(f"{self.path} is not open")
        try:
            return os.read(self._fd, nbytes)
        except BlockingIOError:
            return b""

    def _read_reg(self, nbytes=1):
        UART_RBR = 0x00
        UART_LSR = 0x14
        UART_RFL = 0x84
        LSR_DR = 1 << 0
        out = bytearray()
        target = int(nbytes)
        deadline = time.monotonic() + max(0.0, float(self.timeout))
        while len(out) < target:
            level = self._reg.read32(UART_RFL)
            if level == 0 and (self._reg.read32(UART_LSR) & LSR_DR):
                level = 1
            if level:
                count = min(level, target - len(out))
                out += self._reg.read8_repeat(UART_RBR, count)
                continue
            if self.timeout == 0 or time.monotonic() >= deadline:
                break
        return bytes(out)

    def readinto(self, buf):
        data = self.read(len(buf))
        buf[: len(data)] = data
        return len(data)

    def send(self, data):
        return self.write(data)

    def recv(self, nbytes=1):
        return self.read(nbytes)

    def any(self):
        if self._fd is None and self._reg is not None:
            with self._lock:
                return int(self._reg.read32(0x84))
        if self._fd is None:
            return 0
        try:
            import fcntl
            import termios
            import array

            value = array.array("i", [0])
            fcntl.ioctl(self._fd, termios.FIONREAD, value, True)
            return int(value[0])
        except Exception:
            return 0

    def mmio(self):
        return Reg(f"uart{self.status.bus}")

    def deinit(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        if self._reg is not None:
            self._reg.close()
            self._reg = None
        if self._cru is not None:
            self._cru.close()
            self._cru = None
        self._pinmux.close()

    close = deinit

    def __repr__(self):
        return f"UART({self.id}, path='{self.path}', group='{self.status.group}')"


class I2C:
    def __init__(
        self,
        id=None,
        *,
        scl=None,
        sda=None,
        pins=None,
        freq=400_000,
        timeout=1.0,
        bind=False,
        backend="auto",
        source_clock_hz=200_000_000,
        irq_guard=True,
    ):
        self.id = id
        self.freq_hz = freq
        self.timeout = timeout
        self.backend = _backend_name(backend)
        self.source_clock_hz = source_clock_hz
        self._irq_guard = bool(irq_guard)
        self._pinmux = PinMux()
        self._fd = None
        self._reg = None
        self._cru = None
        self._gicd = None

        pin_list = []
        _append_pin_names(pin_list, pins)
        for pin in (scl, sda):
            _append_pin_names(pin_list, pin)

        requested_bus = _parse_bus_id(id, "i2c")
        if pin_list:
            bus = _infer_i2c_from_pins(self._pinmux, pin_list, requested_bus)
            self.id = bus
            self.status = self._pinmux.i2c(bus, pin_list, bind_driver=bind)
        elif requested_bus is not None:
            self.id = requested_bus
            self.status = _make_status(
                ok=False,
                bus=requested_bus,
                device=f"i2c{requested_bus}",
                dev_path=f"/dev/i2c-{requested_bus}",
                group="",
                pins=[],
            )
        else:
            raise ValueError("I2C requires pins when id is omitted; use I2C(scl=..., sda=...)")
        self.path = self.status.dev_path
        self._lock = _peripheral_lock(f"i2c{self.status.bus}")
        if self.backend in ("linux", "dev", "i2cdev") or (bind and self.backend == "auto" and os.path.exists(self.path)):
            self._fd = os.open(self.path, os.O_RDWR | getattr(os, "O_CLOEXEC", 0))
        elif self.backend in ("auto", "reg", "register", "direct"):
            self._reg = Reg(f"i2c{self.status.bus}")
            self._enable_register_clock()
            self._configure_register_bus()
        else:
            raise ValueError("I2C backend must be auto, reg, register, direct, linux, dev, or i2cdev")

    def _i2c_irq_id(self):
        bus = int(self.status.bus)
        if bus < 0 or bus > 4:
            return None
        return 32 + 18 + bus

    def _mask_kernel_irq(self):
        if not self._irq_guard:
            return None
        irq_id = self._i2c_irq_id()
        if irq_id is None:
            return None
        if self._gicd is None:
            self._gicd = Reg("gicd")
        index = irq_id // 32
        bit = 1 << (irq_id % 32)
        enabled_offset = 0x100 + index * 4
        clear_offset = 0x180 + index * 4
        was_enabled = bool(self._gicd.read32(enabled_offset) & bit)
        if was_enabled:
            self._gicd.write32(clear_offset, bit)
        return (index, bit, was_enabled)

    def _restore_kernel_irq(self, token):
        if not token:
            return
        index, bit, was_enabled = token
        if self._gicd is None or not was_enabled:
            return
        pending_clear_offset = 0x280 + index * 4
        enable_offset = 0x100 + index * 4
        self._gicd.write32(pending_clear_offset, bit)
        self._gicd.write32(enable_offset, bit)

    def _select_slave(self, addr):
        if self._fd is None:
            raise RuntimeError(f"{self.path} is not open")
        import fcntl

        fcntl.ioctl(self._fd, 0x0703, int(addr))

    def _enable_register_clock(self):
        self._cru = Reg("cru")
        bus = int(self.status.bus)
        if bus != 1:
            _cru_ungate(
                self._cru,
                0x12000 + 0x800 + 0x00,
                (1 << 0) | (1 << 1) | (1 << 2) | (1 << 4) | (1 << 5) | (1 << 6),
            )
        if bus == 0:
            _cru_hiword_update(self._cru, 0x12000 + 0x300 + 0x04, 8, 2, 0)
            _cru_ungate(self._cru, 0x12000 + 0x800 + 0x04, (1 << 6) | (1 << 7))
            _cru_hiword_update(self._cru, 0x12000 + 0xA00 + 0x04, 6, 2, 0)
        elif bus == 1:
            _cru_ungate(self._cru, 0x800 + 0x00, (1 << 0) | (1 << 1) | (1 << 2))
            _cru_hiword_update(self._cru, 0x300 + 0x00, 6, 2, 0)
            _cru_ungate(self._cru, 0x800 + 0x00, (1 << 3) | (1 << 4))
            _cru_hiword_update(self._cru, 0xA00 + 0x00, 3, 2, 0)
        elif bus == 2:
            _cru_hiword_update(self._cru, 0x12000 + 0x300 + 0x04, 12, 2, 0)
            _cru_ungate(self._cru, 0x12000 + 0x800 + 0x04, (1 << 10) | (1 << 11))
            _cru_hiword_update(self._cru, 0x12000 + 0xA00 + 0x04, 10, 2, 0)
        elif bus == 3:
            _cru_hiword_update(self._cru, 0x12000 + 0x300 + 0x04, 14, 2, 0)
            _cru_ungate(self._cru, 0x12000 + 0x800 + 0x04, (1 << 12) | (1 << 13))
            _cru_hiword_update(self._cru, 0x12000 + 0xA00 + 0x04, 12, 2, 0)
        elif bus == 4:
            _cru_hiword_update(self._cru, 0x12000 + 0x300 + 0x08, 0, 2, 0)
            _cru_ungate(self._cru, 0x12000 + 0x800 + 0x04, (1 << 14) | (1 << 15))
            _cru_hiword_update(self._cru, 0x12000 + 0xA00 + 0x04, 14, 2, 0)

    def _calculate_i2c_timing(self):
        # RV1106 exposes the v1 Rockchip I2C timing fields in CON[15:8].
        clk_rate_khz = max(1, int((self.source_clock_hz + 999) // 1000))
        scl_rate_khz = max(1, int(self.freq_hz // 1000))
        min_total_div = max(4, int((clk_rate_khz + scl_rate_khz * 8 - 1) // (scl_rate_khz * 8)))
        div_low = max(2, min_total_div // 2)
        div_high = max(2, min_total_div - div_low)
        data_upd = 3
        start_setup = 1
        stop_setup = 1
        return (
            min(div_low - 1, 0xFFFF),
            min(div_high - 1, 0xFFFF),
            ((data_upd & 0xF) << 8) | ((start_setup & 0x3) << 12) | ((stop_setup & 0x3) << 14),
        )

    def _configure_register_bus(self):
        REG_CON = 0x00
        REG_CLKDIV = 0x04
        REG_IEN = 0x18
        REG_IPD = 0x1C
        REG_SCL_OE_DB = 0x24
        REG_CON1 = 0x228
        div_low, div_high, self._i2c_tuning = self._calculate_i2c_timing()
        self._reg.write32(REG_CON, self._i2c_tuning)
        self._reg.write32(REG_IEN, 0)
        self._reg.write32(REG_IPD, 0xFF)
        self._reg.write32(REG_CON1, 0)
        self._reg.write32(REG_CLKDIV, (div_high << 16) | div_low)
        period_ns = max(1, int(1_000_000_000 // max(1, int(self.source_clock_hz))))
        self._reg.write32(REG_SCL_OE_DB, max(0x20, int(100_000_000 // period_ns)))

    def _write_i2c_words(self, base, data):
        data = bytes(data)
        for offset in range(0, len(data), 4):
            word = 0
            for index, value in enumerate(data[offset : offset + 4]):
                word |= int(value) << (8 * index)
            self._reg.write32(base + offset, word)

    def _read_i2c_words(self, base, length):
        out = bytearray()
        for offset in range(0, length, 4):
            word = self._reg.read32(base + offset)
            for index in range(4):
                if len(out) >= length:
                    break
                out.append((word >> (8 * index)) & 0xFF)
        return bytes(out)

    def _wait_i2c(self, wanted, *, ignore_nak=False):
        REG_IPD = 0x1C
        INT_NAKRCV = 1 << 6
        deadline = time.monotonic() + max(0.001, float(self.timeout))
        while time.monotonic() < deadline:
            status = self._reg.read32(REG_IPD)
            if status & INT_NAKRCV and not ignore_nak:
                self._reg.write32(REG_IPD, status)
                raise OSError("I2C NAK")
            if status & wanted:
                self._reg.write32(REG_IPD, status)
                return status
        raise TimeoutError("register I2C transfer timed out")

    def _start_i2c_reg(self, mode, *, lastack=False):
        REG_CON = 0x00
        REG_CON_EN = 1 << 0
        REG_CON_START = 1 << 3
        REG_CON_LASTACK = 1 << 5
        REG_CON_ACTACK = 1 << 6

        tuning = self._i2c_tuning or (self._reg.read32(REG_CON) & 0xFF00)
        con = tuning | REG_CON_EN | (int(mode) << 1) | REG_CON_START | REG_CON_ACTACK
        if lastack:
            con |= REG_CON_LASTACK
        self._reg.write32(REG_CON, con)

    def _stop_i2c_reg(self, mode=0, *, lastack=False, raise_on_timeout=True):
        REG_CON = 0x00
        REG_IEN = 0x18
        REG_CON_EN = 1 << 0
        REG_CON_START = 1 << 3
        REG_CON_STOP = 1 << 4
        REG_CON_LASTACK = 1 << 5
        REG_CON_ACTACK = 1 << 6
        INT_STOP = 1 << 5

        con = self._reg.read32(REG_CON)
        if (con & REG_CON_EN) == 0:
            con = (self._i2c_tuning or (con & 0xFF00)) | REG_CON_EN | (int(mode) << 1) | REG_CON_ACTACK
        con |= REG_CON_STOP
        con &= ~REG_CON_START
        if lastack:
            con |= REG_CON_LASTACK
        self._reg.write32(REG_IEN, INT_STOP)
        self._reg.write32(REG_CON, con)
        try:
            self._wait_i2c(INT_STOP, ignore_nak=True)
        except TimeoutError:
            if raise_on_timeout:
                raise
        finally:
            self._reg.write32(REG_CON, self._reg.read32(REG_CON) & ~REG_CON_STOP)

    def _writeto_reg(self, addr, buf):
        with self._lock:
            irq_token = self._mask_kernel_irq()
            try:
                return self._writeto_reg_unmasked(addr, buf)
            finally:
                self._restore_kernel_irq(irq_token)

    def _writeto_reg_unmasked(self, addr, buf):
        REG_CON = 0x00
        REG_MTXCNT = 0x10
        REG_IEN = 0x18
        REG_IPD = 0x1C
        REG_CON1 = 0x228
        TXBUFFER_BASE = 0x100
        REG_CON_EN = 1 << 0
        REG_CON_MOD_TX = 0 << 1
        REG_CON_START = 1 << 3
        REG_CON_STOP = 1 << 4
        REG_CON_ACTACK = 1 << 6
        INT_MBTF = 1 << 2
        INT_STOP = 1 << 5
        INT_NAKRCV = 1 << 6

        data = bytes(buf)
        total = 0
        starts = range(0, len(data), 31) if data else (0,)
        for start in starts:
            chunk = data[start : start + 31] if data else b""
            payload = bytes([(int(addr) & 0x7F) << 1]) + chunk
            self._reg.write32(REG_CON, self._i2c_tuning)
            self._reg.write32(REG_IEN, 0)
            self._reg.write32(REG_IPD, 0xFF)
            self._reg.write32(REG_CON1, 0)
            try:
                self._write_i2c_words(TXBUFFER_BASE, payload)
                self._reg.write32(REG_IEN, INT_MBTF | INT_NAKRCV)
                self._start_i2c_reg(REG_CON_MOD_TX)
                self._reg.write32(REG_MTXCNT, len(payload))
                self._wait_i2c(INT_MBTF)
                self._stop_i2c_reg(REG_CON_MOD_TX)
            except Exception:
                self._stop_i2c_reg(REG_CON_MOD_TX, raise_on_timeout=False)
                raise
            finally:
                self._reg.write32(REG_IEN, 0)
                self._reg.write32(REG_CON, self._i2c_tuning)
            total += len(chunk)
        return total

    def _readfrom_reg(self, addr, nbytes, memaddr_bytes=b""):
        with self._lock:
            irq_token = self._mask_kernel_irq()
            try:
                return self._readfrom_reg_unmasked(addr, nbytes, memaddr_bytes)
            finally:
                self._restore_kernel_irq(irq_token)

    def _readfrom_reg_unmasked(self, addr, nbytes, memaddr_bytes=b""):
        REG_CON = 0x00
        REG_MRXADDR = 0x08
        REG_MRXRADDR = 0x0C
        REG_MRXCNT = 0x14
        REG_IEN = 0x18
        REG_IPD = 0x1C
        REG_CON1 = 0x228
        RXBUFFER_BASE = 0x200
        REG_CON_MOD_REGISTER_TX_ID = 1
        INT_MBRF = 1 << 3
        INT_NAKRCV = 1 << 6
        MRXADDR_VALID0 = 1 << 24

        remaining = int(nbytes)
        out = bytearray()
        first = True
        while remaining > 0:
            count = min(32, remaining)
            raddr = 0
            if first:
                for index, value in enumerate(bytes(memaddr_bytes)[:4]):
                    raddr |= int(value) << (8 * index)
                    raddr |= 1 << (24 + index)
            self._reg.write32(REG_CON, self._i2c_tuning)
            self._reg.write32(REG_IEN, 0)
            self._reg.write32(REG_IPD, 0xFF)
            self._reg.write32(REG_CON1, 0)
            slave_addr = (int(addr) & 0x7F) << 1
            if not memaddr_bytes or not first:
                slave_addr |= 1
            self._reg.write32(REG_MRXADDR, slave_addr | MRXADDR_VALID0)
            self._reg.write32(REG_MRXRADDR, raddr)
            try:
                self._reg.write32(REG_IEN, INT_MBRF | INT_NAKRCV)
                self._start_i2c_reg(REG_CON_MOD_REGISTER_TX_ID, lastack=True)
                self._reg.write32(REG_MRXCNT, count)
                self._wait_i2c(INT_MBRF)
                out += self._read_i2c_words(RXBUFFER_BASE, count)
                self._stop_i2c_reg(REG_CON_MOD_REGISTER_TX_ID, lastack=True)
            except Exception:
                self._stop_i2c_reg(REG_CON_MOD_REGISTER_TX_ID, lastack=True, raise_on_timeout=False)
                raise
            finally:
                self._reg.write32(REG_IEN, 0)
                self._reg.write32(REG_CON, self._i2c_tuning)
            remaining -= count
            first = False
        return bytes(out)

    def writeto(self, addr, buf, stop=True):
        del stop
        if self._fd is None and self._reg is not None:
            return self._writeto_reg(addr, buf)
        self._select_slave(addr)
        return os.write(self._fd, bytes(buf))

    def readfrom(self, addr, nbytes, stop=True):
        del stop
        if self._fd is None and self._reg is not None:
            return self._readfrom_reg(addr, nbytes)
        self._select_slave(addr)
        return os.read(self._fd, int(nbytes))

    def send(self, addr, data, stop=True):
        return self.writeto(addr, data, stop=stop)

    def read(self, addr, nbytes, stop=True):
        return self.readfrom(addr, nbytes, stop=stop)

    def recv(self, addr, nbytes, stop=True):
        return self.readfrom(addr, nbytes, stop=stop)

    def readfrom_into(self, addr, buf, stop=True):
        data = self.readfrom(addr, len(buf), stop=stop)
        buf[: len(data)] = data
        return None

    def writeto_mem(self, addr, memaddr, buf, *, addrsize=8):
        if addrsize not in (8, 16):
            raise ValueError("addrsize must be 8 or 16")
        prefix = bytes([memaddr & 0xFF]) if addrsize == 8 else bytes([(memaddr >> 8) & 0xFF, memaddr & 0xFF])
        return self.writeto(addr, prefix + bytes(buf))

    def readfrom_mem(self, addr, memaddr, nbytes, *, addrsize=8):
        if addrsize not in (8, 16):
            raise ValueError("addrsize must be 8 or 16")
        prefix = bytes([memaddr & 0xFF]) if addrsize == 8 else bytes([(memaddr >> 8) & 0xFF, memaddr & 0xFF])
        if self._fd is None and self._reg is not None:
            return self._readfrom_reg(addr, nbytes, prefix)
        self.writeto(addr, prefix)
        return self.readfrom(addr, nbytes)

    def readfrom_mem_into(self, addr, memaddr, buf, *, addrsize=8):
        data = self.readfrom_mem(addr, memaddr, len(buf), addrsize=addrsize)
        buf[: len(data)] = data
        return None

    def scan(self):
        found = []
        if self._fd is None and self._reg is not None:
            with self._lock:
                irq_token = self._mask_kernel_irq()
                old_timeout = self.timeout
                self.timeout = min(float(old_timeout), 0.005)
                try:
                    for addr in range(0x03, 0x78):
                        try:
                            self._readfrom_reg_unmasked(addr, 1)
                            found.append(addr)
                        except Exception:
                            pass
                    return found
                finally:
                    self.timeout = old_timeout
                    self._restore_kernel_irq(irq_token)
        if self._fd is None:
            return found
        import fcntl

        class _I2CSmbusData(ctypes.Union):
            _fields_ = [
                ("byte", ctypes.c_uint8),
                ("word", ctypes.c_uint16),
                ("block", ctypes.c_uint8 * 34),
            ]

        class _I2CSmbusIoctlData(ctypes.Structure):
            _fields_ = [
                ("read_write", ctypes.c_uint8),
                ("command", ctypes.c_uint8),
                ("size", ctypes.c_int),
                ("data", ctypes.POINTER(_I2CSmbusData)),
            ]

        for addr in range(0x03, 0x78):
            try:
                self._select_slave(addr)
                data = _I2CSmbusData()
                args = _I2CSmbusIoctlData(0, 0, 0, ctypes.pointer(data))
                fcntl.ioctl(self._fd, 0x0720, args)
                found.append(addr)
            except OSError:
                pass
        return found

    def mmio(self):
        return Reg(f"i2c{self.status.bus}")

    def deinit(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        if self._reg is not None:
            try:
                self._reg.write32(0x00, 0)
            except Exception:
                pass
            self._reg.close()
            self._reg = None
        if self._cru is not None:
            self._cru.close()
            self._cru = None
        if self._gicd is not None:
            self._gicd.close()
            self._gicd = None
        self._pinmux.close()

    close = deinit

    def __repr__(self):
        return f"I2C({self.id}, path='{self.path}', group='{self.status.group}')"


class PWM:
    def __init__(
        self,
        id=None,
        *,
        pin=None,
        freq=1000,
        duty=0,
        duty_u16=None,
        bind=False,
        backend="auto",
        source_clock_hz=24_000_000,
    ):
        self.id = id
        self._freq = int(freq)
        self._duty_u16 = 0
        self._enabled = False
        self.backend = _backend_name(backend)
        self.source_clock_hz = int(source_clock_hz)
        self._pinmux = PinMux()
        pin_arg = _pin_name(pin) if pin is not None else None
        pins = [] if pin_arg is None else [pin_arg]

        requested_channel = _parse_bus_id(id, "pwm")
        if pin_arg is not None:
            channel = _infer_pwm_from_pin(self._pinmux, pin_arg, requested_channel)
            self.id = channel
            self.status = self._pinmux.pwm(channel, pins, bind_driver=bind)
        elif requested_channel is not None:
            self.id = requested_channel
            self.status = _make_status(
                ok=False,
                channel=requested_channel,
                device=f"pwm{requested_channel}",
                dev_path="",
                group="",
                pins=[],
            )
        else:
            raise ValueError("PWM requires pin when id is omitted; use PWM(pin=...)")
        self.path = self.status.dev_path
        self._reg = None
        self._cru = None
        self._lock = _peripheral_lock(f"pwm{self.status.channel}")
        self._use_sysfs = self.backend in ("sysfs", "linux", "dev")
        self._pwm_path = self._find_sysfs_pwm() if self._use_sysfs else None
        if self.backend in ("auto", "reg", "register", "direct"):
            self._reg = Reg(f"pwm{self.status.channel}")
            self._enable_register_clock()
        elif self._use_sysfs:
            if self._pwm_path is None:
                raise RuntimeError(f"PWM sysfs backend is unavailable for {self.status.device}")
        else:
            raise ValueError("PWM backend must be auto, reg, register, direct, sysfs, linux, or dev")

        value = duty_u16 if duty_u16 is not None else int(max(0, min(1023, duty)) * 65535 / 1023)
        self.duty_u16(value)

    def _find_sysfs_pwm(self):
        root = "/sys/class/pwm"
        if not os.path.isdir(root):
            return None
        for name in os.listdir(root):
            if not name.startswith("pwmchip"):
                continue
            chip = os.path.join(root, name)
            device_link = os.path.join(chip, "device")
            try:
                device_name = os.path.basename(os.path.realpath(device_link))
            except OSError:
                device_name = ""
            if device_name != self.status.device:
                continue
            pwm_path = os.path.join(chip, "pwm0")
            if not os.path.isdir(pwm_path):
                try:
                    with open(os.path.join(chip, "export"), "w", encoding="ascii") as f:
                        f.write("0")
                except OSError:
                    pass
            return pwm_path if os.path.isdir(pwm_path) else None
        return None

    def _write_sysfs(self, name, value):
        if not self._use_sysfs or self._pwm_path is None:
            return False
        try:
            with open(os.path.join(self._pwm_path, name), "w", encoding="ascii") as f:
                f.write(str(value))
            return True
        except OSError:
            return False

    def _apply_sysfs(self):
        if not self._use_sysfs or self._pwm_path is None or self._freq <= 0:
            return
        period_ns = max(1, int(1_000_000_000 / self._freq))
        duty_ns = int(period_ns * self._duty_u16 / 65535)
        self._write_sysfs("enable", 0)
        self._write_sysfs("period", period_ns)
        self._write_sysfs("duty_cycle", min(duty_ns, period_ns))
        if self._enabled:
            self._write_sysfs("enable", 1)

    def _select_pwm_clock(self):
        requested = int(self.source_clock_hz)
        if requested >= 75_000_000:
            self.source_clock_hz = 100_000_000
            return 0
        if requested >= 37_000_000:
            self.source_clock_hz = 50_000_000
            return 1
        self.source_clock_hz = 24_000_000
        return 2

    def _enable_register_clock(self):
        self._cru = Reg("cru")
        mux = self._select_pwm_clock()
        channel = int(self.status.channel)
        if channel <= 3:
            _cru_hiword_update(self._cru, 0x12000 + 0x300 + 0x2C, 0, 2, mux)
            _cru_ungate(self._cru, 0x12000 + 0x800 + 0x1C, (1 << 3) | (1 << 4))
        elif channel <= 7:
            _cru_hiword_update(self._cru, 0x12000 + 0x300 + 0x18, 9, 2, mux)
            _cru_ungate(self._cru, 0x12000 + 0x800 + 0x0C, 1 << 15)
            _cru_ungate(self._cru, 0x12000 + 0x800 + 0x10, 1 << 0)
        else:
            _cru_hiword_update(self._cru, 0x12000 + 0x300 + 0x18, 11, 2, mux)
            _cru_ungate(self._cru, 0x12000 + 0x800 + 0x10, (1 << 2) | (1 << 3))

    def _apply_register(self):
        if self._reg is None or self._freq <= 0:
            return
        REG_CNTR = 0x00
        REG_PERIOD = 0x04
        REG_DUTY = 0x08
        REG_CTRL = 0x0C
        PWM_ENABLE = 1 << 0
        PWM_CONTINUOUS = 1 << 1
        PWM_DUTY_POSITIVE = 1 << 3
        PWM_LOCK_EN = 1 << 6
        period = max(2, int(self.source_clock_hz // self._freq))
        duty = min(period, int(period * self._duty_u16 / 65535))
        base_ctrl = PWM_CONTINUOUS | PWM_DUTY_POSITIVE
        self._reg.write32(REG_CTRL, base_ctrl | PWM_LOCK_EN)
        self._reg.write32(REG_CNTR, 0)
        self._reg.write32(REG_PERIOD, period)
        self._reg.write32(REG_DUTY, duty)
        self._reg.write32(REG_CTRL, base_ctrl | (PWM_ENABLE if self._enabled else 0))

    def freq(self, value=None):
        if value is None:
            return self._freq
        with self._lock:
            self._freq = int(value)
            self._apply_sysfs()
            self._apply_register()
            return self._freq

    def duty_u16(self, value=None):
        if value is None:
            return self._duty_u16
        with self._lock:
            self._duty_u16 = max(0, min(65535, int(value)))
            self._apply_sysfs()
            self._apply_register()
            return self._duty_u16

    def duty(self, value=None):
        if value is None:
            return int(self._duty_u16 * 1023 / 65535)
        return self.duty_u16(int(max(0, min(1023, int(value))) * 65535 / 1023))

    def enable(self, value=True):
        with self._lock:
            self._enabled = bool(value)
            self._write_sysfs("enable", 1 if self._enabled else 0)
            self._apply_register()

    def disable(self):
        self.enable(False)

    def mmio(self):
        return Reg(f"pwm{self.status.channel}")

    def deinit(self):
        self.disable()
        if self._reg is not None:
            try:
                self._reg.write32(0x0C, 0)
            except Exception:
                pass
            self._reg.close()
            self._reg = None
        if self._cru is not None:
            self._cru.close()
            self._cru = None
        self._pinmux.close()

    close = deinit

    def __repr__(self):
        backend = "reg" if self._reg is not None else "sysfs"
        return f"PWM({self.id}, group='{self.status.group}', backend='{backend}')"


MMIO = Reg
Register = Reg


__all__ = [n for n in globals() if not n.startswith("_")]

