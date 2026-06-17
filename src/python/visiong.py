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
_RTLD_LAZY = getattr(os, "RTLD_LAZY", 1)
_RTLD_GLOBAL = getattr(os, "RTLD_GLOBAL", 0x100)
_RTLD_MODE = _RTLD_NOW | _RTLD_GLOBAL
_PRELOAD_MODE = _RTLD_LAZY | _RTLD_GLOBAL
_LD_MARKER = "_VISIONG_LD_PATH_READY"


def _python_restart_argv():
    original = getattr(sys, "orig_argv", None)
    if original and len(original) > 1:
        return [sys.executable] + list(original[1:])
    return [sys.executable] + (sys.argv if sys.argv else [])


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
    if not sys.argv:
        return
    if sys.argv[0] == "":
        if sys.stdin is not None and sys.stdin.isatty() and exe:
            os.execv(exe, [exe, "-i", "-c", "import visiong"])
        return
    if sys.argv[0] == "-c":
        if exe:
            os.execv(exe, _python_restart_argv())
        return
    if sys.argv[0] == "-":
        return

    if exe:
        os.execv(exe, _python_restart_argv())


def _preload_vendor_libraries_global():
    search_dirs = (_MODULE_DIR, "/oem/usr/lib")
    libraries = (
        "librga.so",
        "librockchip_mpp.so.1",
        "librockit.so",
        "librockit_full.so",
        "librkaiq.so",
        "librockiva.so",
        "librknnmrt.so",
        "librve.so",
        "libivs.so",
    )
    for name in libraries:
        for directory in search_dirs:
            try:
                ctypes.CDLL(os.path.join(directory, name), mode=_PRELOAD_MODE)
                break
            except OSError:
                continue


_ensure_loader_library_path()

_old_flags = sys.getdlopenflags()
try:
    while _MODULE_DIR in sys.path:
        sys.path.remove(_MODULE_DIR)
    sys.path.insert(0, _MODULE_DIR)
    sys.setdlopenflags(_RTLD_MODE)
    _preload_vendor_libraries_global()
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


_NativeDisplaySPI = globals().get("DisplaySPI")
_NativeMppRecorder = globals().get("MppRecorder")
_SHORT_GPIO_PIN_PATTERN = re.compile(r"^(?:GPIO)?(?P<bank>\d+)_?(?P<group>[A-Da-d])(?P<index>[0-7])(?:_d)?$")
_GPIO_OFFSET_PIN_PATTERN = re.compile(r"^(?:GPIO)?(?P<bank>\d+)[-:](?P<pin>\d+)$", re.IGNORECASE)
_DEFAULT_DISPLAY_SPI_PINS = ("4A0", "4A1", "4A5", "4A7")
_DEFAULT_DISPLAY_SPI_DC_PIN = "GPIO1_C3"
_DEFAULT_DISPLAY_SPI_RESET_PIN = "GPIO1_C2"


def _video_codec_from_path(filepath, codec="auto"):
    codec_text = str(codec or "auto").strip().lower()
    if codec_text in ("h264", "avc"):
        return "h264"
    if codec_text in ("h265", "hevc"):
        return "h265"
    if codec_text != "auto":
        raise ValueError("codec must be 'auto', 'h264', or 'h265'")

    ext = os.path.splitext(str(filepath))[1].lower()
    return "h265" if ext in (".h265", ".hevc") else "h264"


def _video_container_from_path(filepath):
    ext = os.path.splitext(str(filepath))[1].lower()
    return "mp4" if ext == ".mp4" else "annexb"


def _friendly_video_error(exc):
    text = str(exc)
    for old, new in (
        ("MppRecorder", "video recorder"),
        ("MPP", "video encoder"),
        ("mpp", "video encoder"),
    ):
        text = text.replace(old, new)
    return text


def load_text_chars_from_dict(dict_path):
    dict_path = os.fspath(dict_path)
    chars = []
    seen = set()
    try:
        with open(dict_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                token = line.rstrip("\r\n")
                if line_no == 0:
                    token = token.lstrip("\ufeff")
                if not token:
                    continue
                for ch in token:
                    if ch not in seen:
                        seen.add(ch)
                        chars.append(ch)
    except OSError as exc:
        raise RuntimeError(f"Failed to load PPOCR dictionary chars from '{dict_path}': {exc}") from None

    if not chars:
        raise RuntimeError(f"PPOCR dictionary is empty: '{dict_path}'")
    return "".join(chars)


def set_text_font_for_dict(font_path="", dict_path="", glyph_budget=0):
    if not hasattr(ImageBuffer, "set_text_font"):
        raise RuntimeError("ImageBuffer.set_text_font is not available in this build")

    dict_chars = ""
    if dict_path:
        dict_chars = load_text_chars_from_dict(dict_path)

    ImageBuffer.set_text_font(font_path, dict_chars, int(max(0, glyph_budget)))
    return dict_chars


class VideoRecorder:
    """Simple video writer. Construct once, call write(frame), then close()."""

    def __init__(self, filepath, quality=75, fps=30, codec="auto"):
        if _NativeMppRecorder is None:
            raise RuntimeError("Video recording is not available in this build")
        self.filepath = str(filepath)
        self.codec = _video_codec_from_path(self.filepath, codec)
        self.container = _video_container_from_path(self.filepath)
        try:
            self._recorder = _NativeMppRecorder(
                self.filepath,
                codec=self.codec,
                container=self.container,
                quality=int(quality),
                fps=int(fps),
            )
        except Exception as exc:
            raise RuntimeError(f"VideoRecorder: failed to create recorder. {_friendly_video_error(exc)}") from None

    def write(self, img):
        recorder = getattr(self, "_recorder", None)
        if recorder is None:
            raise RuntimeError("VideoRecorder: recorder is closed")
        try:
            recorder.write(img)
        except Exception as exc:
            raise RuntimeError(f"VideoRecorder: failed to write frame. {_friendly_video_error(exc)}") from None

    def close(self):
        recorder = getattr(self, "_recorder", None)
        if recorder is not None:
            try:
                recorder.close()
            except Exception as exc:
                raise RuntimeError(f"VideoRecorder: failed to close video. {_friendly_video_error(exc)}") from None
            finally:
                self._recorder = None

    def is_open(self):
        return getattr(self, "_recorder", None) is not None

    def is_started(self):
        recorder = getattr(self, "_recorder", None)
        return bool(recorder is not None and recorder.is_open())

    def path(self):
        return self.filepath

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


def _gpio_offset_to_name(bank, pin):
    bank = int(bank)
    pin = int(pin)
    if pin < 0 or pin > 31:
        raise ValueError(f"invalid GPIO pin offset: {pin}")
    return f"GPIO{bank}_{chr(ord('A') + pin // 8)}{pin % 8}"


def _normalize_pin_name(pin):
    text = str(getattr(pin, "id", pin)).strip()
    match = _SHORT_GPIO_PIN_PATTERN.match(text)
    if match:
        return f"GPIO{int(match.group('bank'))}_{match.group('group').upper()}{int(match.group('index'))}"
    match = _GPIO_OFFSET_PIN_PATTERN.match(text)
    if match:
        return _gpio_offset_to_name(match.group("bank"), match.group("pin"))
    return text


def _pin_name(pin):
    return _normalize_pin_name(pin)


def _looks_like_pin(value):
    if value is None:
        return False
    text = str(getattr(value, "id", value)).strip()
    return bool(_SHORT_GPIO_PIN_PATTERN.match(text) or _GPIO_OFFSET_PIN_PATTERN.match(text))


def _is_int_like(value):
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    text = str(value).strip().replace("_", "")
    return bool(re.match(r"^[+-]?\d+$", text))


def _int_like_value(value):
    return int(str(value).strip().replace("_", ""))


def _split_trailing_int(values):
    values = list(values or [])
    if values and not _looks_like_pin(values[-1]) and _is_int_like(values[-1]):
        return values[:-1], _int_like_value(values[-1])
    return values, None


def _collect_positional_values(id_value, args):
    if _looks_like_pin(id_value):
        return None, [id_value] + list(args)
    return id_value, list(args)


def _looks_like_spi_bus(value):
    if value is None:
        return False
    text = str(value).strip().lower()
    return bool(
        text.startswith("/dev/spidev")
        or re.match(r"^(?:/dev/)?spidev\d+\.\d+$", text)
        or re.match(r"^spi\d+\.\d+$", text)
    )


def _normalize_spi_bus(value):
    if value is None:
        return None
    text = str(value).strip()
    lower = text.lower()
    if lower.startswith("/dev/"):
        return text
    if lower.startswith("spidev"):
        return "/dev/" + text
    if lower.startswith("spi") and "." in lower:
        return "/dev/spidev" + text[3:]
    return text


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


def _append_pin_with_role(out, roles, value, role=None):
    before = len(out)
    _append_pin_names(out, value)
    for _ in range(len(out) - before):
        roles.append(role)


def _backend_name(value):
    text = str(value or "auto").strip().lower()
    return text or "auto"


def _displayfb_is_any_active():
    displayfb = globals().get("DisplayFB")
    check = getattr(displayfb, "is_any_active", None)
    if check is None:
        return False
    try:
        return bool(check())
    except Exception:
        return False


def _rect_tuple(value, name="rectangle"):
    try:
        items = tuple(value)
    except TypeError:
        raise ValueError(f"{name} must be a 4-tuple") from None
    if len(items) != 4:
        raise ValueError(f"{name} must be a 4-tuple")
    return tuple(int(item) for item in items)


def _rgb_component(value):
    if isinstance(value, float) and 0.0 <= value <= 1.0:
        value = value * 255.0
    return max(0, min(255, int(round(value))))


def _color_rgb565(color):
    if isinstance(color, bool):
        raise ValueError("color must be RGB565 int or an RGB tuple")
    if isinstance(color, int):
        if color < 0 or color > 0xFFFF:
            raise ValueError("RGB565 color integer must be in 0..0xffff")
        return int(color)
    try:
        values = tuple(color)
    except TypeError:
        raise ValueError("color must be RGB565 int or an RGB tuple") from None
    if len(values) < 3:
        raise ValueError("RGB color tuple must contain at least r, g, b")
    r, g, b = (_rgb_component(values[0]), _rgb_component(values[1]), _rgb_component(values[2]))
    return ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)


_HW_LOAD_WARNED = set()


def _try_load_hw_accel(context):
    try:
        if "HW" not in globals():
            return False
        if HW.is_available():
            return True
        loaded = bool(HW.load())
        if loaded or HW.is_available():
            return True
        message = "visiong_hw.ko is unavailable; falling back to native PIO register transfer."
    except Exception as exc:
        message = f"visiong_hw.ko load failed ({exc}); falling back to native PIO register transfer."

    if context not in _HW_LOAD_WARNED:
        print(f"[{context}] Warning: {message}", file=sys.stderr)
        _HW_LOAD_WARNED.add(context)
    return False


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


def _native_role_hints(role_hints, count, peripheral):
    hints = list(role_hints or [None] * count)
    if len(hints) != count:
        raise ValueError(f"{peripheral} role_hints length must match pins length")
    return ["" if role is None else str(role) for role in hints]


def _infer_spi_from_pins(pinmux, pins, requested_bus=None, requested_cs=None, role_hints=None):
    if len(pins) < 3:
        raise ValueError("SPI pins must include clk/sck, cs, and at least one of mosi/miso")
    native = getattr(pinmux, "_infer_spi_from_pins_native", None)
    if native is None:
        raise RuntimeError("native PinMux SPI inference is unavailable; update _visiong.so")
    bus, chip_select = native(
        pins,
        -1 if requested_bus is None else int(requested_bus),
        -1 if requested_cs is None else int(requested_cs),
        _native_role_hints(role_hints, len(pins), "SPI"),
    )
    return int(bus), int(chip_select)


def _infer_i2c_from_pins(pinmux, pins, requested_bus=None, role_hints=None):
    if len(pins) < 2:
        raise ValueError("I2C pins must include scl and sda")
    native = getattr(pinmux, "_infer_i2c_from_pins_native", None)
    if native is None:
        raise RuntimeError("native PinMux I2C inference is unavailable; update _visiong.so")
    (bus,) = native(
        pins,
        -1 if requested_bus is None else int(requested_bus),
        _native_role_hints(role_hints, len(pins), "I2C"),
    )
    return int(bus)


def _infer_uart_from_pins(pinmux, pins, requested_bus=None, role_hints=None):
    if not pins:
        raise ValueError("UART pins must include at least tx or rx")
    native = getattr(pinmux, "_infer_uart_from_pins_native", None)
    if native is None:
        raise RuntimeError("native PinMux UART inference is unavailable; update _visiong.so")
    (bus,) = native(
        pins,
        -1 if requested_bus is None else int(requested_bus),
        _native_role_hints(role_hints, len(pins), "UART"),
    )
    return int(bus)


def _infer_pwm_from_pin(pinmux, pin, requested_channel=None):
    native = getattr(pinmux, "_infer_pwm_from_pin_native", None)
    if native is None:
        raise RuntimeError("native PinMux PWM inference is unavailable; update _visiong.so")
    (channel,) = native(_pin_name(pin), -1 if requested_channel is None else int(requested_channel))
    return int(channel)


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
        if not os.path.islink(link_path):
            return ""
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
        *args,
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
        self._hw_reg = None
        self._hw_reg_unavailable = False
        self._reg_backend = False
        self._released_spi_child = ""
        self._released_spi_driver = ""
        self._released_spi_driver_path = ""
        self._released_platform_device = ""
        self._released_platform_driver = ""
        self._released_platform_driver_path = ""

        id, positional_pins = _collect_positional_values(id, args)
        positional_pins, positional_baudrate = _split_trailing_int(positional_pins)
        if positional_baudrate is not None:
            baudrate = positional_baudrate
        if positional_pins:
            if len(positional_pins) not in (3, 4):
                raise ValueError("SPI positional pins must contain clk/sck, cs, and mosi or miso")
        self.baudrate = int(baudrate)

        pin_list = []
        role_hints = []
        _append_pin_with_role(pin_list, role_hints, pins, None)
        _append_pin_with_role(pin_list, role_hints, positional_pins, None)
        for pin, role in (
            (clk if clk is not None else sck, "clk"),
            (mosi, "mosi"),
            (miso, "miso"),
            (cs, "cs"),
        ):
            _append_pin_with_role(pin_list, role_hints, pin, role)

        requested_bus = _parse_bus_id(id, "spi")
        if pin_list:
            bus, inferred_cs = _infer_spi_from_pins(self._pinmux, pin_list, requested_bus, chip_select, role_hints)
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

    def write(self, data):
        if self._fd is None and self._reg_backend:
            with self._lock:
                return self._transfer_reg(bytes(data), 0, tx_only=True)
        if self._fd is None:
            raise RuntimeError(f"{self.path} is not open; use backend='reg' or bind=True for SPI.write()")
        return os.write(self._fd, bytes(data))

    def _transfer_reg_fast(self, tx_data, rx_len=0, *, tx_only=False):
        if self._hw_reg_unavailable:
            return None
        try:
            if self._hw_reg is None:
                _try_load_hw_accel("SPI")
                self._hw_reg = HW(required=False, autoload=False)
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
        mode = (1 if self.phase else 0) | (2 if self.polarity else 0)
        return _spi_reg_pio_transfer_native(
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
        kwargs.setdefault("spi", self)
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
        self._pinmux.close()

    close = deinit

    def __repr__(self):
        return f"SPI({self.id}, path='{self.path}', group='{self.status.group}')"


class DisplaySPI:
    __doc__ = """SPI display output for ST7789-compatible panels.

    Accepts either an existing SPI object, SPI pins, or the legacy spi_bus
    argument. Pin based construction is preferred for RV1103/RV1106 because
    the bus and chip-select can be inferred from the pinmux table.
    """

    def __init__(
        self,
        *args,
        chip_model="ST7789",
        spi=None,
        spi_bus=None,
        width=240,
        height=320,
        rotation_degrees=90,
        dc_pin=_DEFAULT_DISPLAY_SPI_DC_PIN,
        reset_pin=_DEFAULT_DISPLAY_SPI_RESET_PIN,
        backlight_pin="",
        speed_hz=50_000_000,
        baudrate=None,
        x_offset=0,
        y_offset=0,
        bgr=False,
        invert=False,
        spi_mode=0,
        bits_per_word=8,
        transfer_chunk_size=4096,
        multi_buffering=True,
        buffer_count=3,
        backend="auto",
        source_clock_hz=200_000_000,
        clk=None,
        sck=None,
        mosi=None,
        miso=None,
        cs=None,
        pins=None,
        chip_select=None,
        bind=False,
        polarity=0,
        phase=0,
        dummy=0xFF,
    ):
        if _NativeDisplaySPI is None:
            raise RuntimeError("native DisplaySPI backend is unavailable")
        if _displayfb_is_any_active() and _backend_name(backend) in ("reg", "register", "direct"):
            raise RuntimeError("DisplaySPI requires DisplayFB.release() first when using the register/direct SPI backend")
        if baudrate is not None:
            speed_hz = baudrate

        values = list(args)
        if values and not _looks_like_pin(values[0]) and not _looks_like_spi_bus(values[0]):
            chip_model = values.pop(0)
        if values and _looks_like_spi_bus(values[0]):
            spi_bus = values.pop(0)
        values, positional_speed = _split_trailing_int(values)
        if positional_speed is not None:
            speed_hz = positional_speed
        if values and len(values) not in (3, 4):
            raise ValueError("DisplaySPI positional SPI pins must contain clk/sck, cs, and mosi or miso")

        self._spi = None
        self._owns_spi = False
        self._impl = None

        if spi is not None and values:
            raise ValueError("DisplaySPI accepts either spi= or positional SPI pins, not both")
        if spi is not None and any(value is not None for value in (clk, sck, mosi, miso, cs, pins)):
            raise ValueError("DisplaySPI accepts either spi= or SPI pin arguments, not both")

        pin_args_supplied = values or pins is not None or any(value is not None for value in (clk, sck, mosi, miso, cs))
        if spi is None and spi_bus is None and not pin_args_supplied:
            values = list(_DEFAULT_DISPLAY_SPI_PINS)
            pin_args_supplied = True

        if spi is None and pin_args_supplied:
            spi_args = values
            self._spi = SPI(
                *spi_args,
                baudrate=speed_hz,
                polarity=polarity,
                phase=phase,
                bits=bits_per_word,
                clk=clk,
                sck=sck,
                mosi=mosi,
                miso=miso,
                cs=cs,
                pins=pins,
                chip_select=chip_select,
                bind=False,
                backend="reg",
                source_clock_hz=source_clock_hz,
                dummy=dummy,
            )
            self._owns_spi = True
        elif spi is not None:
            self._spi = spi

        if self._spi is not None:
            spi_bus = getattr(self._spi, "path", spi_bus)
            speed_hz = int(getattr(self._spi, "baudrate", speed_hz))
            source_clock_hz = int(getattr(self._spi, "source_clock_hz", source_clock_hz))
            if backend == "auto" and getattr(self._spi, "_reg_backend", False):
                backend = "reg"

        spi_bus = _normalize_spi_bus(spi_bus)
        if spi_bus is None:
            raise ValueError("DisplaySPI requires spi=, SPI pins, or spi_bus")
        dc_pin = _pin_name(dc_pin) if dc_pin else ""
        reset_pin = _pin_name(reset_pin) if reset_pin else ""
        backlight_pin = _pin_name(backlight_pin) if backlight_pin else ""

        try:
            if _backend_name(backend) in ("auto", "reg", "register", "direct"):
                _try_load_hw_accel("DisplaySPI")
            self._impl = _NativeDisplaySPI(
                chip_model=chip_model,
                spi_bus=spi_bus,
                width=width,
                height=height,
                rotation_degrees=rotation_degrees,
                dc_pin=dc_pin,
                reset_pin=reset_pin,
                backlight_pin=backlight_pin,
                speed_hz=speed_hz,
                x_offset=x_offset,
                y_offset=y_offset,
                bgr=bgr,
                invert=invert,
                spi_mode=spi_mode,
                bits_per_word=bits_per_word,
                transfer_chunk_size=transfer_chunk_size,
                multi_buffering=multi_buffering,
                buffer_count=buffer_count,
                backend=backend,
                source_clock_hz=source_clock_hz,
            )
        except Exception:
            if self._owns_spi and self._spi is not None:
                self._spi.deinit()
                self._spi = None
                self._owns_spi = False
            raise

    def __getattr__(self, name):
        impl = self.__dict__.get("_impl")
        if impl is None:
            raise AttributeError(name)
        return getattr(impl, name)

    def display(self, frame, roi=None):
        if roi is None:
            return self._impl.display(frame)
        return self._impl.display(frame, _rect_tuple(roi, "roi"))

    def display_area(self, frame, x=0, y=0, roi=None):
        if roi is None:
            return self._impl.display_area(frame, int(x), int(y))
        return self._impl.display_area(frame, int(x), int(y), _rect_tuple(roi, "roi"))

    def draw_rgb565(self, x, y, w, h, data, stride_bytes=0, source_is_native_endian=True):
        return self._impl.draw_rgb565(
            int(x),
            int(y),
            int(w),
            int(h),
            data,
            0 if stride_bytes is None else int(stride_bytes),
            bool(source_is_native_endian),
        )

    def draw_pixel(self, x, y, color=0xFFFF):
        return self._impl.draw_pixel(int(x), int(y), _color_rgb565(color))

    def draw_line(self, x0, y0, x1, y1, color=0xFFFF, thickness=1):
        return self._impl.draw_line(int(x0), int(y0), int(x1), int(y1), _color_rgb565(color), int(thickness))

    def draw_rectangle(self, x, y=None, w=None, h=None, color=0xFFFF, thickness=1, fill=False):
        if isinstance(x, (list, tuple)):
            rect = _rect_tuple(x, "rectangle")
            if y is not None:
                color = y
            if w is not None:
                thickness = w
            if h is not None:
                fill = h
            x, y, w, h = rect
        elif y is None or w is None or h is None:
            raise TypeError("draw_rectangle() requires x, y, w, h or a rectangle tuple")
        thickness = int(thickness)
        if thickness < 0:
            fill = True
            thickness = 1
        return self._impl.draw_rectangle(int(x), int(y), int(w), int(h), _color_rgb565(color), thickness, bool(fill))

    def draw_circle(self, cx, cy, radius, color=0xFFFF, thickness=1, fill=False):
        thickness = int(thickness)
        if thickness < 0:
            fill = True
            thickness = 1
        return self._impl.draw_circle(int(cx), int(cy), int(radius), _color_rgb565(color), thickness, bool(fill))

    def draw_cross(self, cx, cy, color=0xFFFF, size=5, thickness=1):
        return self._impl.draw_cross(int(cx), int(cy), _color_rgb565(color), int(size), int(thickness))

    def clear(self, color=0):
        return self._impl.clear(_color_rgb565(color))

    def draw(self, *args, **kwargs):
        if not args:
            raise TypeError("draw() requires an ImageBuffer or a primitive name")
        primitive = args[0]
        if not isinstance(primitive, str):
            return self.display_area(*args, **kwargs)
        name = primitive.strip().lower().replace("-", "_")
        rest = args[1:]
        if name in ("image", "frame", "buffer", "area", "display_area"):
            return self.display_area(*rest, **kwargs)
        if name in ("rgb565", "bitmap", "raw"):
            return self.draw_rgb565(*rest, **kwargs)
        if name in ("pixel", "point"):
            return self.draw_pixel(*rest, **kwargs)
        if name in ("line",):
            return self.draw_line(*rest, **kwargs)
        if name in ("rect", "rectangle", "box"):
            return self.draw_rectangle(*rest, **kwargs)
        if name in ("circle",):
            return self.draw_circle(*rest, **kwargs)
        if name in ("cross",):
            return self.draw_cross(*rest, **kwargs)
        raise ValueError("unknown DisplaySPI draw primitive: " + primitive)

    def release(self):
        impl = self._impl
        self._impl = None
        try:
            if impl is not None:
                release = getattr(impl, "release", None)
                if release is not None:
                    release()
                else:
                    close = getattr(impl, "close", None)
                    if close is not None:
                        close()
        finally:
            if self._owns_spi and self._spi is not None:
                self._spi.deinit()
                self._spi = None
                self._owns_spi = False

    close = release
    deinit = release

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass

    def __repr__(self):
        return f"DisplaySPI(spi={self._spi!r})"


class UART:
    def __init__(
        self,
        id=None,
        *args,
        baudrate=115200,
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

        id, positional_pins = _collect_positional_values(id, args)
        positional_pins, positional_baudrate = _split_trailing_int(positional_pins)
        if positional_baudrate is not None:
            baudrate = positional_baudrate
        if positional_pins:
            if len(positional_pins) > 4:
                raise ValueError("UART positional pins must include up to four UART pins")

        self.baudrate = int(baudrate)

        pin_list = []
        role_hints = []
        _append_pin_with_role(pin_list, role_hints, pins, None)
        _append_pin_with_role(pin_list, role_hints, positional_pins, None)
        for pin, role in ((tx, "tx"), (rx, "rx"), (rts, "rts"), (cts, "cts")):
            _append_pin_with_role(pin_list, role_hints, pin, role)

        requested_bus = _parse_bus_id(id, "uart")
        if pin_list:
            bus = _infer_uart_from_pins(self._pinmux, pin_list, requested_bus, role_hints)
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
        return _uart_reg_write_native(int(self.status.bus), bytes(data), float(self.timeout))

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
        return _uart_reg_read_native(int(self.status.bus), int(nbytes), float(self.timeout))

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
                return _uart_reg_any_native(int(self.status.bus))
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
        *args,
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

        id, positional_pins = _collect_positional_values(id, args)
        positional_pins, positional_freq = _split_trailing_int(positional_pins)
        if positional_freq is not None:
            freq = positional_freq
            self.freq_hz = freq
        if positional_pins:
            if len(positional_pins) != 2:
                raise ValueError("I2C positional pins must be (scl, sda)")

        pin_list = []
        role_hints = []
        _append_pin_with_role(pin_list, role_hints, pins, None)
        _append_pin_with_role(pin_list, role_hints, positional_pins, None)
        for pin, role in ((scl, "scl"), (sda, "sda")):
            _append_pin_with_role(pin_list, role_hints, pin, role)

        requested_bus = _parse_bus_id(id, "i2c")
        if pin_list:
            bus = _infer_i2c_from_pins(self._pinmux, pin_list, requested_bus, role_hints)
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
        return _i2c_reg_writeto_native(
            int(self.status.bus),
            int(addr),
            bytes(buf),
            int(self._i2c_tuning),
            float(self.timeout),
        )

    def _readfrom_reg(self, addr, nbytes, memaddr_bytes=b""):
        with self._lock:
            irq_token = self._mask_kernel_irq()
            try:
                return self._readfrom_reg_unmasked(addr, nbytes, memaddr_bytes)
            finally:
                self._restore_kernel_irq(irq_token)

    def _readfrom_reg_unmasked(self, addr, nbytes, memaddr_bytes=b""):
        return _i2c_reg_readfrom_native(
            int(self.status.bus),
            int(addr),
            int(nbytes),
            bytes(memaddr_bytes),
            int(self._i2c_tuning),
            float(self.timeout),
        )

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
        *args,
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
        id, positional_values = _collect_positional_values(id, args)
        positional_values, positional_freq = _split_trailing_int(positional_values)
        if positional_freq is not None:
            freq = positional_freq
            self._freq = int(freq)
        if positional_values:
            if len(positional_values) != 1:
                raise ValueError("PWM positional form is PWM(pin), PWM(pin, freq), or PWM(channel, pin, freq)")
            if pin is None:
                pin = positional_values[0]
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
