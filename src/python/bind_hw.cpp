// SPDX-License-Identifier: LGPL-3.0-or-later
#include "internal/bindings_common.h"
#include "visiong/uapi/visiong_hw.h"

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr const char* kDefaultHwPath = "/dev/visiong-hw";
constexpr uintptr_t kRv1106Spi0Base = 0xff500000;
constexpr uintptr_t kRv1106Spi1Base = 0xff510000;
constexpr uintptr_t kRv1106CruBase = 0xff3a0000;
constexpr size_t kRegisterMapSize = 0x1000;
constexpr size_t kCruMapSize = 0x20000;

constexpr uint32_t kSpiCtrlr0 = 0x0000;
constexpr uint32_t kSpiCtrlr1 = 0x0004;
constexpr uint32_t kSpiSsienr = 0x0008;
constexpr uint32_t kSpiSer = 0x000c;
constexpr uint32_t kSpiBaudr = 0x0010;
constexpr uint32_t kSpiTxftlr = 0x0014;
constexpr uint32_t kSpiRxftlr = 0x0018;
constexpr uint32_t kSpiTxflr = 0x001c;
constexpr uint32_t kSpiRxflr = 0x0020;
constexpr uint32_t kSpiSr = 0x0024;
constexpr uint32_t kSpiImr = 0x002c;
constexpr uint32_t kSpiIcr = 0x0038;
constexpr uint32_t kSpiDmacr = 0x003c;
constexpr uint32_t kSpiVersion = 0x0048;
constexpr uint32_t kSpiTxdr = 0x0400;
constexpr uint32_t kSpiRxdr = 0x0800;
constexpr uint32_t kSpiMaxTransferLen = 0xffff;
constexpr uint32_t kSpiVer2Type1 = 0x05ec0002;
constexpr uint32_t kSpiVer2Type2 = 0x00110002;

constexpr uintptr_t kRv1106UartBase[] = {
    0xff4a0000,
    0xff4b0000,
    0xff4c0000,
    0xff4d0000,
    0xff4e0000,
    0xff4f0000,
};

constexpr uintptr_t kRv1106I2cBase[] = {
    0xff310000,
    0xff320000,
    0xff450000,
    0xff460000,
    0xff470000,
};

[[noreturn]] void throw_os_error(const std::string& message) {
    PyErr_SetString(PyExc_OSError, message.c_str());
    throw py::error_already_set();
}

[[noreturn]] void throw_timeout_error(const std::string& message) {
    PyErr_SetString(PyExc_TimeoutError, message.c_str());
    throw py::error_already_set();
}

bool path_exists(const std::string& path) {
    return !path.empty() && ::access(path.c_str(), F_OK) == 0;
}

std::string shell_quote(const std::string& input) {
    std::string out = "'";
    for (char c : input) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out += c;
        }
    }
    out += "'";
    return out;
}

int fd_from_object(const py::object& object) {
    if (py::isinstance<py::int_>(object)) {
        return object.cast<int>();
    }
    if (py::hasattr(object, "fd")) {
        return py::getattr(object, "fd").cast<int>();
    }
    throw std::invalid_argument("object must be an fd integer or expose .fd");
}

int size_from_object(const py::object& object) {
    if (py::hasattr(object, "size")) {
        return py::getattr(object, "size").cast<int>();
    }
    return -1;
}

std::vector<uint8_t> bytes_from_object(const py::object& object) {
    if (object.is_none()) {
        return {};
    }
    if (py::isinstance<py::bytes>(object)) {
        const std::string data = object.cast<std::string>();
        return std::vector<uint8_t>(data.begin(), data.end());
    }
    py::buffer buffer = py::reinterpret_borrow<py::buffer>(object);
    py::buffer_info info = buffer.request();
    if (info.ndim < 1 || info.itemsize <= 0 || info.size < 0) {
        throw std::invalid_argument("buffer object is invalid");
    }
    const auto* ptr = static_cast<const uint8_t*>(info.ptr);
    return std::vector<uint8_t>(ptr, ptr + (info.size * info.itemsize));
}

uintptr_t spi_base_for_bus(int bus) {
    switch (bus) {
        case 0:
            return kRv1106Spi0Base;
        case 1:
            return kRv1106Spi1Base;
        default:
            throw std::invalid_argument("register SPI PIO backend currently supports spi0 and spi1 on RV1103/RV1106");
    }
}

uintptr_t uart_base_for_bus(int bus) {
    if (bus < 0 || bus >= static_cast<int>(sizeof(kRv1106UartBase) / sizeof(kRv1106UartBase[0]))) {
        throw std::invalid_argument("register UART backend currently supports uart0..uart5 on RV1103/RV1106");
    }
    return kRv1106UartBase[bus];
}

uintptr_t i2c_base_for_bus(int bus) {
    if (bus < 0 || bus >= static_cast<int>(sizeof(kRv1106I2cBase) / sizeof(kRv1106I2cBase[0]))) {
        throw std::invalid_argument("register I2C backend currently supports i2c0..i2c4 on RV1103/RV1106");
    }
    return kRv1106I2cBase[bus];
}

uint32_t mmio_read32(volatile uint8_t* base, uint32_t offset) {
    return *reinterpret_cast<volatile uint32_t*>(const_cast<uint8_t*>(base) + offset);
}

void mmio_write32(volatile uint8_t* base, uint32_t offset, uint32_t value) {
    *reinterpret_cast<volatile uint32_t*>(const_cast<uint8_t*>(base) + offset) = value;
    __sync_synchronize();
}

void mmio_write8(volatile uint8_t* base, uint32_t offset, uint8_t value) {
    *(base + offset) = value;
}

uint8_t mmio_read8(volatile uint8_t* base, uint32_t offset) {
    return *(base + offset);
}

void cru_hiword_update_raw(volatile uint8_t* cru, uint32_t offset, uint32_t shift, uint32_t width, uint32_t value) {
    const uint32_t mask = ((1u << width) - 1u) << shift;
    mmio_write32(cru, offset, (mask << 16) | ((value << shift) & mask));
}

void cru_ungate_raw(volatile uint8_t* cru, uint32_t offset, uint32_t bits) {
    mmio_write32(cru, offset, bits << 16);
}

class MmioMap {
public:
    explicit MmioMap(uintptr_t base, size_t size = kRegisterMapSize) : size_(size) {
        mem_fd_ = ::open("/dev/mem", O_RDWR | O_SYNC | O_CLOEXEC);
        if (mem_fd_ < 0) {
            throw_os_error("failed to open /dev/mem for register backend");
        }
        void* map = ::mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd_, base);
        if (map == MAP_FAILED) {
            const std::string err = std::strerror(errno);
            close();
            throw std::runtime_error("failed to mmap peripheral registers: " + err);
        }
        base_ = static_cast<volatile uint8_t*>(map);
    }

    ~MmioMap() { close(); }

    MmioMap(const MmioMap&) = delete;
    MmioMap& operator=(const MmioMap&) = delete;

    uint32_t read32(uint32_t offset) const { return mmio_read32(base_, offset); }
    void write32(uint32_t offset, uint32_t value) const { mmio_write32(base_, offset, value); }
    uint8_t read8(uint32_t offset) const { return mmio_read8(base_, offset); }
    void write8(uint32_t offset, uint8_t value) const { mmio_write8(base_, offset, value); }

private:
    void close() {
        if (base_) {
            ::munmap(const_cast<uint8_t*>(base_), size_);
            base_ = nullptr;
        }
        if (mem_fd_ >= 0) {
            ::close(mem_fd_);
            mem_fd_ = -1;
        }
    }

    int mem_fd_ = -1;
    volatile uint8_t* base_ = nullptr;
    size_t size_ = 0;
};

class SpiPioMap {
public:
    explicit SpiPioMap(int bus) : bus_(bus) {
        mem_fd_ = ::open("/dev/mem", O_RDWR | O_SYNC | O_CLOEXEC);
        if (mem_fd_ < 0) {
            throw_os_error("failed to open /dev/mem for SPI PIO register backend");
        }
        void* cru_map = ::mmap(nullptr, kCruMapSize, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd_, kRv1106CruBase);
        if (cru_map == MAP_FAILED) {
            const std::string err = std::strerror(errno);
            close();
            throw std::runtime_error("failed to mmap RV1106 CRU registers: " + err);
        }
        cru_ = static_cast<volatile uint8_t*>(cru_map);
        enable_clocks();

        void* spi_map = ::mmap(nullptr, kRegisterMapSize, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd_, spi_base_for_bus(bus_));
        if (spi_map == MAP_FAILED) {
            const std::string err = std::strerror(errno);
            close();
            throw std::runtime_error("failed to mmap SPI registers: " + err);
        }
        spi_ = static_cast<volatile uint8_t*>(spi_map);
        const uint32_t version = read(kSpiVersion);
        fifo_len_ = (version == kSpiVer2Type1 || version == kSpiVer2Type2) ? 64 : 32;
        write(kSpiSsienr, 0);
        write(kSpiImr, 0);
        write(kSpiIcr, 0xffffffff);
        write(kSpiDmacr, 0);
    }

    ~SpiPioMap() { close(); }

    uint32_t read(uint32_t offset) const { return mmio_read32(spi_, offset); }
    void write(uint32_t offset, uint32_t value) const { mmio_write32(spi_, offset, value); }
    uint8_t read8(uint32_t offset) const { return mmio_read8(spi_, offset); }
    void write8(uint32_t offset, uint8_t value) const { mmio_write8(spi_, offset, value); }
    int fifo_len() const { return fifo_len_; }

private:
    void enable_clocks() {
        if (bus_ == 0) {
            cru_hiword_update_raw(cru_, 0x1a000 + 0x300, 12, 2, 0);
            cru_ungate_raw(cru_, 0x1a000 + 0x800 + 0x04, (1u << 2) | (1u << 3) | (1u << 4));
        } else if (bus_ == 1) {
            cru_hiword_update_raw(cru_, 0x12000 + 0x300 + 0x18, 3, 2, 0);
            cru_ungate_raw(cru_, 0x12000 + 0x800 + 0x0c, (1u << 6) | (1u << 7));
        }
    }

    void close() {
        if (spi_) {
            mmio_write32(spi_, kSpiSsienr, 0);
            mmio_write32(spi_, kSpiSer, 0);
            ::munmap(const_cast<uint8_t*>(spi_), kRegisterMapSize);
            spi_ = nullptr;
        }
        if (cru_) {
            ::munmap(const_cast<uint8_t*>(cru_), kCruMapSize);
            cru_ = nullptr;
        }
        if (mem_fd_ >= 0) {
            ::close(mem_fd_);
            mem_fd_ = -1;
        }
    }

    int bus_ = 0;
    int mem_fd_ = -1;
    volatile uint8_t* cru_ = nullptr;
    volatile uint8_t* spi_ = nullptr;
    int fifo_len_ = 64;
};

py::object spi_reg_pio_transfer_native(int bus,
                                       int chip_select,
                                       py::object tx_data_object,
                                       int rx_len,
                                       bool tx_only,
                                       int speed_hz,
                                       int source_clock_hz,
                                       int mode,
                                       int bits_per_word,
                                       int dummy) {
    if (bits_per_word != 8) {
        throw std::invalid_argument("register SPI PIO backend currently supports bits=8 only");
    }
    if (chip_select < 0 || chip_select > 1) {
        throw std::invalid_argument("register SPI PIO backend supports chip_select 0 or 1");
    }
    if (rx_len < 0) {
        throw std::invalid_argument("rx_len must be >= 0");
    }
    std::vector<uint8_t> tx = bytes_from_object(tx_data_object);
    if (tx.empty() && rx_len == 0) {
        if (tx_only) {
            return py::int_(0);
        }
        return py::bytes("");
    }

    const size_t total_frames = tx_only ? tx.size() : std::max(tx.size(), static_cast<size_t>(rx_len));
    if (!tx_only && tx.size() < total_frames) {
        tx.resize(total_frames, static_cast<uint8_t>(dummy & 0xff));
    }

    SpiPioMap regs(bus);
    const uint32_t div_raw = std::max(2, (source_clock_hz + speed_hz - 1) / std::max(1, speed_hz));
    const uint32_t div = (div_raw & 1u) ? div_raw + 1u : div_raw;
    const uint32_t cr0_base = 0x1u | (1u << 10) | (1u << 11) | (1u << 13);
    const uint32_t xfm = (tx_only && rx_len == 0) ? (1u << 18) : 0u;
    const uint32_t mode_bits = (static_cast<uint32_t>(mode) & 0x3u) << 6;
    const int fifo = regs.fifo_len();

    size_t written = 0;
    std::vector<uint8_t> received;
    received.reserve(static_cast<size_t>(rx_len));

    {
    py::gil_scoped_release release;
    for (size_t start = 0; start < total_frames; start += kSpiMaxTransferLen) {
        const size_t count = std::min<size_t>(kSpiMaxTransferLen, total_frames - start);
        regs.write(kSpiSsienr, 0);
        regs.write(kSpiImr, 0);
        regs.write(kSpiIcr, 0xffffffff);
        regs.write(kSpiDmacr, 0);
        regs.write(kSpiCtrlr0, cr0_base | mode_bits | xfm);
        regs.write(kSpiCtrlr1, static_cast<uint32_t>(count - 1));
        regs.write(kSpiTxftlr, static_cast<uint32_t>(std::max(1, fifo / 2)));
        regs.write(kSpiRxftlr, 0);
        regs.write(kSpiBaudr, div);
        regs.write(kSpiSer, 1u << chip_select);
        regs.write(kSpiSsienr, 1);

        size_t tx_pos = 0;
        const size_t rx_target = tx_only ? 0 : count;
        size_t rx_pos = 0;
        size_t idle_guard = 0;
        while (tx_pos < count || rx_pos < rx_target) {
            bool progressed = false;
            const uint32_t level = regs.read(kSpiTxflr);
            if (tx_pos < count && level < static_cast<uint32_t>(fifo)) {
                const size_t writable = std::min<size_t>(static_cast<size_t>(fifo - level), count - tx_pos);
                for (size_t i = 0; i < writable; ++i) {
                    regs.write8(kSpiTxdr, tx[start + tx_pos + i]);
                }
                tx_pos += writable;
                written += writable;
                progressed = true;
            }

            if (rx_pos < rx_target) {
                const size_t readable = std::min<size_t>(regs.read(kSpiRxflr), rx_target - rx_pos);
                if (readable) {
                    const size_t remaining = rx_len > static_cast<int>(received.size())
                                                 ? static_cast<size_t>(rx_len) - received.size()
                                                 : 0;
                    for (size_t i = 0; i < readable; ++i) {
                        const uint8_t value = regs.read8(kSpiRxdr);
                        if (i < remaining) {
                            received.push_back(value);
                        }
                    }
                    rx_pos += readable;
                    progressed = true;
                }
            }

            if (progressed) {
                idle_guard = 0;
            } else if (++idle_guard > 5000000) {
                regs.write(kSpiSsienr, 0);
                regs.write(kSpiSer, 0);
                throw std::runtime_error("register SPI PIO transfer timed out");
            }
        }

        idle_guard = 0;
        while (regs.read(kSpiSr) & 1u) {
            if (++idle_guard > 5000000) {
                regs.write(kSpiSsienr, 0);
                regs.write(kSpiSer, 0);
                throw std::runtime_error("register SPI PIO transfer did not become idle");
            }
        }
        regs.write(kSpiSsienr, 0);
        regs.write(kSpiSer, 0);
    }
    }

    if (tx_only) {
        return py::int_(written);
    }
    return py::bytes(reinterpret_cast<const char*>(received.data()), std::min<size_t>(received.size(), rx_len));
}

int uart_reg_write_native(int bus, py::object data_object, double timeout_seconds) {
    constexpr uint32_t kUartThr = 0x00;
    constexpr uint32_t kUartLsr = 0x14;
    constexpr uint32_t kUartTfl = 0x80;
    constexpr uint32_t kUartFifoDepth = 64;
    constexpr uint32_t kUartLsrThre = 1u << 5;

    const std::vector<uint8_t> payload = bytes_from_object(data_object);
    if (payload.empty()) {
        return 0;
    }
    MmioMap regs(uart_base_for_bus(bus), 0x1000);
    const auto timeout = std::chrono::duration<double>(std::max(0.01, timeout_seconds > 0.0 ? timeout_seconds : 1.0));
    auto deadline = std::chrono::steady_clock::now() + timeout;
    size_t written = 0;
    while (written < payload.size()) {
        const uint32_t level = std::min(kUartFifoDepth, regs.read32(kUartTfl));
        uint32_t space = kUartFifoDepth - level;
        if (space == 0 && (regs.read32(kUartLsr) & kUartLsrThre)) {
            space = kUartFifoDepth;
        }
        if (space == 0) {
            if (std::chrono::steady_clock::now() > deadline) {
                throw_timeout_error("register UART TX FIFO did not drain");
            }
            continue;
        }
        const size_t count = std::min<size_t>(space, payload.size() - written);
        for (size_t i = 0; i < count; ++i) {
            regs.write8(kUartThr, payload[written + i]);
        }
        written += count;
        deadline = std::chrono::steady_clock::now() + timeout;
    }
    return static_cast<int>(written);
}

py::bytes uart_reg_read_native(int bus, int nbytes, double timeout_seconds) {
    constexpr uint32_t kUartRbr = 0x00;
    constexpr uint32_t kUartLsr = 0x14;
    constexpr uint32_t kUartRfl = 0x84;
    constexpr uint32_t kUartLsrDr = 1u << 0;

    if (nbytes <= 0) {
        return py::bytes("");
    }
    MmioMap regs(uart_base_for_bus(bus), 0x1000);
    std::vector<uint8_t> out;
    out.reserve(static_cast<size_t>(nbytes));
    const auto timeout = std::chrono::duration<double>(std::max(0.0, timeout_seconds));
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (out.size() < static_cast<size_t>(nbytes)) {
        uint32_t level = regs.read32(kUartRfl);
        if (level == 0 && (regs.read32(kUartLsr) & kUartLsrDr)) {
            level = 1;
        }
        if (level) {
            const size_t count = std::min<size_t>(level, static_cast<size_t>(nbytes) - out.size());
            for (size_t i = 0; i < count; ++i) {
                out.push_back(regs.read8(kUartRbr));
            }
            continue;
        }
        if (timeout_seconds <= 0.0 || std::chrono::steady_clock::now() >= deadline) {
            break;
        }
    }
    return py::bytes(reinterpret_cast<const char*>(out.data()), out.size());
}

int uart_reg_any_native(int bus) {
    constexpr uint32_t kUartRfl = 0x84;
    MmioMap regs(uart_base_for_bus(bus), 0x1000);
    return static_cast<int>(regs.read32(kUartRfl));
}

void i2c_write_words(MmioMap& regs, uint32_t base, const std::vector<uint8_t>& data) {
    for (size_t offset = 0; offset < data.size(); offset += 4) {
        uint32_t word = 0;
        const size_t count = std::min<size_t>(4, data.size() - offset);
        for (size_t index = 0; index < count; ++index) {
            word |= static_cast<uint32_t>(data[offset + index]) << (8 * index);
        }
        regs.write32(base + static_cast<uint32_t>(offset), word);
    }
}

std::vector<uint8_t> i2c_read_words(MmioMap& regs, uint32_t base, size_t length) {
    std::vector<uint8_t> out;
    out.reserve(length);
    for (size_t offset = 0; offset < length; offset += 4) {
        const uint32_t word = regs.read32(base + static_cast<uint32_t>(offset));
        for (size_t index = 0; index < 4 && out.size() < length; ++index) {
            out.push_back(static_cast<uint8_t>((word >> (8 * index)) & 0xff));
        }
    }
    return out;
}

uint32_t wait_i2c_native(MmioMap& regs, uint32_t wanted, double timeout_seconds, bool ignore_nak) {
    constexpr uint32_t kI2cIpd = 0x1c;
    constexpr uint32_t kI2cIntNakRcv = 1u << 6;

    const auto timeout = std::chrono::duration<double>(std::max(0.001, timeout_seconds));
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        const uint32_t status = regs.read32(kI2cIpd);
        if ((status & kI2cIntNakRcv) && !ignore_nak) {
            regs.write32(kI2cIpd, status);
            throw_os_error("I2C NAK");
        }
        if (status & wanted) {
            regs.write32(kI2cIpd, status);
            return status;
        }
    }
    throw_timeout_error("register I2C transfer timed out");
}

void start_i2c_native(MmioMap& regs, uint32_t tuning, uint32_t mode, bool lastack) {
    constexpr uint32_t kI2cCon = 0x00;
    constexpr uint32_t kI2cConEn = 1u << 0;
    constexpr uint32_t kI2cConStart = 1u << 3;
    constexpr uint32_t kI2cConLastAck = 1u << 5;
    constexpr uint32_t kI2cConActAck = 1u << 6;

    uint32_t con = tuning | kI2cConEn | (mode << 1) | kI2cConStart | kI2cConActAck;
    if (lastack) {
        con |= kI2cConLastAck;
    }
    regs.write32(kI2cCon, con);
}

void stop_i2c_native(MmioMap& regs, uint32_t tuning, uint32_t mode, bool lastack, double timeout_seconds, bool raise_on_timeout) {
    constexpr uint32_t kI2cCon = 0x00;
    constexpr uint32_t kI2cIen = 0x18;
    constexpr uint32_t kI2cConEn = 1u << 0;
    constexpr uint32_t kI2cConStart = 1u << 3;
    constexpr uint32_t kI2cConStop = 1u << 4;
    constexpr uint32_t kI2cConLastAck = 1u << 5;
    constexpr uint32_t kI2cConActAck = 1u << 6;
    constexpr uint32_t kI2cIntStop = 1u << 5;

    uint32_t con = regs.read32(kI2cCon);
    if ((con & kI2cConEn) == 0) {
        con = tuning | kI2cConEn | (mode << 1) | kI2cConActAck;
    }
    con |= kI2cConStop;
    con &= ~kI2cConStart;
    if (lastack) {
        con |= kI2cConLastAck;
    }
    regs.write32(kI2cIen, kI2cIntStop);
    regs.write32(kI2cCon, con);
    try {
        wait_i2c_native(regs, kI2cIntStop, timeout_seconds, true);
    } catch (const py::error_already_set&) {
        if (raise_on_timeout) {
            throw;
        }
        PyErr_Clear();
    }
    regs.write32(kI2cCon, regs.read32(kI2cCon) & ~kI2cConStop);
}

int i2c_reg_writeto_native(int bus, int addr, py::object data_object, uint32_t tuning, double timeout_seconds) {
    constexpr uint32_t kI2cCon = 0x00;
    constexpr uint32_t kI2cMtxCnt = 0x10;
    constexpr uint32_t kI2cIen = 0x18;
    constexpr uint32_t kI2cIpd = 0x1c;
    constexpr uint32_t kI2cCon1 = 0x228;
    constexpr uint32_t kI2cTxBufferBase = 0x100;
    constexpr uint32_t kI2cModTx = 0u;
    constexpr uint32_t kI2cIntMbtf = 1u << 2;
    constexpr uint32_t kI2cIntNakRcv = 1u << 6;

    const std::vector<uint8_t> data = bytes_from_object(data_object);
    MmioMap regs(i2c_base_for_bus(bus), 0x1000);
    size_t total = 0;
    const size_t iterations = data.empty() ? 1 : ((data.size() + 30) / 31);
    for (size_t part = 0; part < iterations; ++part) {
        const size_t start = part * 31;
        const size_t chunk_len = data.empty() ? 0 : std::min<size_t>(31, data.size() - start);
        std::vector<uint8_t> payload;
        payload.reserve(chunk_len + 1);
        payload.push_back(static_cast<uint8_t>((addr & 0x7f) << 1));
        payload.insert(payload.end(), data.begin() + static_cast<std::vector<uint8_t>::difference_type>(start),
                       data.begin() + static_cast<std::vector<uint8_t>::difference_type>(start + chunk_len));

        regs.write32(kI2cCon, tuning);
        regs.write32(kI2cIen, 0);
        regs.write32(kI2cIpd, 0xff);
        regs.write32(kI2cCon1, 0);
        try {
            i2c_write_words(regs, kI2cTxBufferBase, payload);
            regs.write32(kI2cIen, kI2cIntMbtf | kI2cIntNakRcv);
            start_i2c_native(regs, tuning, kI2cModTx, false);
            regs.write32(kI2cMtxCnt, static_cast<uint32_t>(payload.size()));
            wait_i2c_native(regs, kI2cIntMbtf, timeout_seconds, false);
            stop_i2c_native(regs, tuning, kI2cModTx, false, timeout_seconds, true);
        } catch (...) {
            try {
                stop_i2c_native(regs, tuning, kI2cModTx, false, timeout_seconds, false);
            } catch (...) {
            }
            regs.write32(kI2cIen, 0);
            regs.write32(kI2cCon, tuning);
            throw;
        }
        regs.write32(kI2cIen, 0);
        regs.write32(kI2cCon, tuning);
        total += chunk_len;
    }
    return static_cast<int>(total);
}

py::bytes i2c_reg_readfrom_native(int bus,
                                  int addr,
                                  int nbytes,
                                  py::object memaddr_object,
                                  uint32_t tuning,
                                  double timeout_seconds) {
    constexpr uint32_t kI2cCon = 0x00;
    constexpr uint32_t kI2cMrxAddr = 0x08;
    constexpr uint32_t kI2cMrxRAddr = 0x0c;
    constexpr uint32_t kI2cMrxCnt = 0x14;
    constexpr uint32_t kI2cIen = 0x18;
    constexpr uint32_t kI2cIpd = 0x1c;
    constexpr uint32_t kI2cCon1 = 0x228;
    constexpr uint32_t kI2cRxBufferBase = 0x200;
    constexpr uint32_t kI2cModRegisterTxId = 1u;
    constexpr uint32_t kI2cIntMbrf = 1u << 3;
    constexpr uint32_t kI2cIntNakRcv = 1u << 6;
    constexpr uint32_t kI2cMrxAddrValid0 = 1u << 24;

    if (nbytes <= 0) {
        return py::bytes("");
    }
    const std::vector<uint8_t> memaddr = bytes_from_object(memaddr_object);
    MmioMap regs(i2c_base_for_bus(bus), 0x1000);
    std::vector<uint8_t> out;
    out.reserve(static_cast<size_t>(nbytes));
    int remaining = nbytes;
    bool first = true;
    while (remaining > 0) {
        const int count = std::min(32, remaining);
        uint32_t raddr = 0;
        if (first) {
            const size_t prefix_len = std::min<size_t>(4, memaddr.size());
            for (size_t index = 0; index < prefix_len; ++index) {
                raddr |= static_cast<uint32_t>(memaddr[index]) << (8 * index);
                raddr |= 1u << (24 + index);
            }
        }
        regs.write32(kI2cCon, tuning);
        regs.write32(kI2cIen, 0);
        regs.write32(kI2cIpd, 0xff);
        regs.write32(kI2cCon1, 0);
        uint32_t slave_addr = static_cast<uint32_t>((addr & 0x7f) << 1);
        if (memaddr.empty() || !first) {
            slave_addr |= 1u;
        }
        regs.write32(kI2cMrxAddr, slave_addr | kI2cMrxAddrValid0);
        regs.write32(kI2cMrxRAddr, raddr);
        try {
            regs.write32(kI2cIen, kI2cIntMbrf | kI2cIntNakRcv);
            start_i2c_native(regs, tuning, kI2cModRegisterTxId, true);
            regs.write32(kI2cMrxCnt, static_cast<uint32_t>(count));
            wait_i2c_native(regs, kI2cIntMbrf, timeout_seconds, false);
            std::vector<uint8_t> chunk = i2c_read_words(regs, kI2cRxBufferBase, static_cast<size_t>(count));
            out.insert(out.end(), chunk.begin(), chunk.end());
            stop_i2c_native(regs, tuning, kI2cModRegisterTxId, true, timeout_seconds, true);
        } catch (...) {
            try {
                stop_i2c_native(regs, tuning, kI2cModRegisterTxId, true, timeout_seconds, false);
            } catch (...) {
            }
            regs.write32(kI2cIen, 0);
            regs.write32(kI2cCon, tuning);
            throw;
        }
        regs.write32(kI2cIen, 0);
        regs.write32(kI2cCon, tuning);
        remaining -= count;
        first = false;
    }
    return py::bytes(reinterpret_cast<const char*>(out.data()), out.size());
}

std::pair<int, int> parse_block_offset(const py::object& block) {
    static const std::map<std::string, int> blocks = {
        {"ioc", VISIONG_HW_REG_BLOCK_IOC},
        {"gpio", VISIONG_HW_REG_BLOCK_IOC},
        {"pinctrl", VISIONG_HW_REG_BLOCK_IOC},
        {"pmuioc", VISIONG_HW_REG_BLOCK_PMUIOC},
        {"cru", VISIONG_HW_REG_BLOCK_CRU},
        {"clock", VISIONG_HW_REG_BLOCK_CRU},
        {"gpio0", VISIONG_HW_REG_BLOCK_GPIO0},
        {"gpio1", VISIONG_HW_REG_BLOCK_GPIO1},
        {"gpio2", VISIONG_HW_REG_BLOCK_GPIO2},
        {"gpio3", VISIONG_HW_REG_BLOCK_GPIO3},
        {"gpio4", VISIONG_HW_REG_BLOCK_GPIO4},
        {"spi0", VISIONG_HW_REG_BLOCK_SPI0},
        {"spi1", VISIONG_HW_REG_BLOCK_SPI1},
        {"i2c0", VISIONG_HW_REG_BLOCK_I2C0},
        {"i2c1", VISIONG_HW_REG_BLOCK_I2C1},
        {"i2c2", VISIONG_HW_REG_BLOCK_I2C2},
        {"i2c3", VISIONG_HW_REG_BLOCK_I2C3},
        {"i2c4", VISIONG_HW_REG_BLOCK_I2C4},
        {"uart0", VISIONG_HW_REG_BLOCK_UART0},
        {"serial0", VISIONG_HW_REG_BLOCK_UART0},
        {"uart1", VISIONG_HW_REG_BLOCK_UART1},
        {"serial1", VISIONG_HW_REG_BLOCK_UART1},
        {"uart2", VISIONG_HW_REG_BLOCK_UART2},
        {"serial2", VISIONG_HW_REG_BLOCK_UART2},
        {"uart3", VISIONG_HW_REG_BLOCK_UART3},
        {"serial3", VISIONG_HW_REG_BLOCK_UART3},
        {"uart4", VISIONG_HW_REG_BLOCK_UART4},
        {"serial4", VISIONG_HW_REG_BLOCK_UART4},
        {"uart5", VISIONG_HW_REG_BLOCK_UART5},
        {"serial5", VISIONG_HW_REG_BLOCK_UART5},
        {"pwm0_3", VISIONG_HW_REG_BLOCK_PWM0_3},
        {"pwm4_7", VISIONG_HW_REG_BLOCK_PWM4_7},
        {"pwm8_11", VISIONG_HW_REG_BLOCK_PWM8_11},
        {"dmac", VISIONG_HW_REG_BLOCK_DMAC},
        {"dma", VISIONG_HW_REG_BLOCK_DMAC},
        {"gicd", VISIONG_HW_REG_BLOCK_GICD},
        {"gic", VISIONG_HW_REG_BLOCK_GICD},
    };

    if (py::isinstance<py::int_>(block)) {
        return {block.cast<int>(), 0};
    }
    std::string token = py::str(block).cast<std::string>();
    std::transform(token.begin(), token.end(), token.begin(), [](unsigned char c) {
        if (c == '-') {
            return '_';
        }
        return static_cast<char>(std::tolower(c));
    });
    std::smatch match;
    static const std::regex pwm_pattern(R"(^pwm([0-9]+)$)");
    if (std::regex_match(token, match, pwm_pattern)) {
        const int channel = std::stoi(match[1].str());
        if (0 <= channel && channel <= 3) {
            return {VISIONG_HW_REG_BLOCK_PWM0_3, channel * 0x10};
        }
        if (4 <= channel && channel <= 7) {
            return {VISIONG_HW_REG_BLOCK_PWM4_7, (channel - 4) * 0x10};
        }
        if (8 <= channel && channel <= 11) {
            return {VISIONG_HW_REG_BLOCK_PWM8_11, (channel - 8) * 0x10};
        }
    }
    const auto it = blocks.find(token);
    if (it == blocks.end()) {
        throw std::invalid_argument("unknown visiong-hw register block: " + py::str(block).cast<std::string>());
    }
    return {it->second, 0};
}

int dma_direction(const py::object& direction) {
    if (py::isinstance<py::int_>(direction)) {
        return direction.cast<int>();
    }
    std::string text = direction.is_none() ? std::string("bidirectional") : py::str(direction).cast<std::string>();
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        return c == '-' ? '_' : static_cast<char>(std::tolower(c));
    });
    if (text == "to_device" || text == "cpu_to_device" || text == "write" || text == "out" || text == "tx") {
        return VISIONG_HW_DMA_SYNC_TO_DEVICE;
    }
    if (text == "from_device" || text == "device_to_cpu" || text == "read" || text == "in" || text == "rx") {
        return VISIONG_HW_DMA_SYNC_FROM_DEVICE;
    }
    if (text == "bidirectional" || text == "bidir" || text == "both") {
        return VISIONG_HW_DMA_SYNC_BIDIRECTIONAL;
    }
    throw std::invalid_argument("DMA direction must be to_device, from_device, or bidirectional");
}

int irq_edge(const py::object& edge) {
    if (py::isinstance<py::int_>(edge)) {
        return edge.cast<int>();
    }
    std::string text = edge.is_none() ? std::string("both") : py::str(edge).cast<std::string>();
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (text == "rising" || text == "rise") {
        return VISIONG_HW_IRQ_EDGE_RISING;
    }
    if (text == "falling" || text == "fall") {
        return VISIONG_HW_IRQ_EDGE_FALLING;
    }
    if (text == "both" || text == "any") {
        return VISIONG_HW_IRQ_EDGE_BOTH;
    }
    throw std::invalid_argument("IRQ edge must be rising, falling, or both");
}

std::pair<int, int> parse_spi_bus(const py::object& bus, const py::object& chip_select) {
    int cs = chip_select.is_none() ? 0 : chip_select.cast<int>();
    if (py::isinstance<py::str>(bus)) {
        const std::string text = py::str(bus).cast<std::string>();
        static const std::regex pattern(R"(^spi([0-9]+)(?:\.([0-9]+))?$)");
        std::smatch match;
        std::string lower = text;
        std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        if (!std::regex_match(lower, match, pattern)) {
            throw std::invalid_argument("spi_open bus must be like 'spi0.0' or an integer bus id");
        }
        if (chip_select.is_none() && match[2].matched) {
            cs = std::stoi(match[2].str());
        }
        return {std::stoi(match[1].str()), cs};
    }
    return {bus.cast<int>(), cs};
}

class PyHW;

class PyHWReg {
public:
    PyHWReg(std::shared_ptr<PyHW> hw, int block, int base_offset)
        : hw_(std::move(hw)), block_(block), base_offset_(base_offset) {}

    uint32_t read32(uint32_t offset) const;
    void write32(uint32_t offset, uint32_t value, uint32_t mask = 0xFFFFFFFFU, bool hiword = false);
    void update32(uint32_t offset, uint32_t mask, uint32_t value) {
        write32(offset, value, mask, false);
    }

    int block() const { return block_; }
    int base_offset() const { return base_offset_; }

private:
    std::shared_ptr<PyHW> hw_;
    int block_ = 0;
    int base_offset_ = 0;
};

class PyHWDmaCopy;
class PyHWSPI;
class PyHWSPITransfer;
class PyHWIRQ;

class PyHWDmaBuffer {
public:
    PyHWDmaBuffer(std::shared_ptr<PyHW> hw, int size, int fd)
        : hw(std::move(hw)), size(size), fd(fd) {}
    ~PyHWDmaBuffer() { close(); }

    py::object mmap(py::object access);
    void sync_for_cpu(py::object direction, int offset, int size_arg);
    void sync_for_device(py::object direction, int offset, int size_arg);
    int fill(uint32_t value, py::object size_arg, int offset);
    void close() {
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
    }

    std::shared_ptr<PyHW> hw;
    int size = 0;
    int fd = -1;
};

class PyHW : public std::enable_shared_from_this<PyHW> {
public:
    explicit PyHW(const std::string& path, bool required, bool autoload)
        : path(path.empty() ? kDefaultHwPath : path) {
        if (autoload && !path_exists(this->path)) {
            load(py::none());
        }
        fd_ = ::open(this->path.c_str(), O_RDWR | O_CLOEXEC);
        if (fd_ < 0 && required) {
            throw_os_error(this->path + " is unavailable; build/load visiong_hw.ko or use existing fallbacks");
        }
    }

    ~PyHW() { close(); }

    static bool is_available(const std::string& path = kDefaultHwPath) {
        const int fd = ::open(path.empty() ? kDefaultHwPath : path.c_str(), O_RDWR | O_CLOEXEC);
        if (fd < 0) {
            return false;
        }
        ::close(fd);
        return true;
    }

    static std::vector<std::string> module_candidates(const py::object& module_path) {
        if (!module_path.is_none()) {
            return {py::str(module_path).cast<std::string>()};
        }
        return {
            "/usr/lib/python3.11/site-packages/visiong_hw.ko",
            "/mnt/sdcard/usr/lib/python3.11/site-packages/visiong_hw.ko",
            "/oem/usr/lib/python3.11/site-packages/visiong_hw.ko",
            "./visiong_hw.ko",
        };
    }

    static bool load(const py::object& module_path) {
        if (is_available()) {
            return true;
        }
        for (const std::string& candidate : module_candidates(module_path)) {
            if (!path_exists(candidate)) {
                continue;
            }
            const std::string command = "insmod " + shell_quote(candidate) + " >/dev/null 2>&1";
            const int rc = std::system(command.c_str());
            if (rc == 0 || is_available()) {
                return true;
            }
        }
        return is_available();
    }

    bool available() const { return fd_ >= 0; }
    int require_fd() const {
        if (fd_ < 0) {
            throw std::runtime_error(path + " is unavailable; build/load visiong_hw.ko or use existing fallbacks");
        }
        return fd_;
    }

    py::dict caps() const {
        visiong_hw_caps caps{};
        caps.size = sizeof(caps);
        if (::ioctl(require_fd(), VISIONG_HW_GET_CAPS, &caps) < 0) {
            throw_os_error("VISIONG_HW_GET_CAPS failed");
        }
        py::dict out;
        out["size"] = caps.size;
        out["abi_version"] = caps.abi_version;
        out["driver_version"] = caps.driver_version;
        out["feature_flags"] = caps.feature_flags;
        out["reg_access"] = static_cast<bool>(caps.feature_flags & VISIONG_HW_FEATURE_REG_ACCESS);
        out["pin_session"] = static_cast<bool>(caps.feature_flags & VISIONG_HW_FEATURE_PIN_SESSION);
        out["gpio_irq"] = static_cast<bool>(caps.feature_flags & VISIONG_HW_FEATURE_GPIO_IRQ);
        out["dma_buffer"] = static_cast<bool>(caps.feature_flags & VISIONG_HW_FEATURE_DMA_BUFFER);
        out["dma_memcpy"] = static_cast<bool>(caps.feature_flags & VISIONG_HW_FEATURE_DMA_MEMCPY);
        out["spi_display"] = static_cast<bool>(caps.feature_flags & VISIONG_HW_FEATURE_SPI_DISPLAY);
        out["dma_fill"] = static_cast<bool>(caps.feature_flags & VISIONG_HW_FEATURE_DMA_FILL);
        out["spi_reg"] = static_cast<bool>(caps.feature_flags & VISIONG_HW_FEATURE_SPI_REG);
        out["chip_id"] = caps.chip_id;
        out["max_dma_bytes"] = caps.max_dma_bytes;
        out["max_transfer_bytes"] = caps.max_transfer_bytes;
        return out;
    }

    uint32_t read32(const py::object& block, uint32_t offset) const {
        const auto parsed = parse_block_offset(block);
        return read32_block(parsed.first, parsed.second + static_cast<int>(offset));
    }

    uint32_t read32_block(int block, int offset) const {
        visiong_hw_reg_access req{};
        req.size = sizeof(req);
        req.block = static_cast<uint32_t>(block);
        req.offset = static_cast<uint32_t>(offset);
        if (::ioctl(require_fd(), VISIONG_HW_REG_READ, &req) < 0) {
            throw_os_error("VISIONG_HW_REG_READ failed");
        }
        return req.value;
    }

    void write32(const py::object& block, uint32_t offset, uint32_t value, uint32_t mask, bool hiword) {
        const auto parsed = parse_block_offset(block);
        write32_block(parsed.first, parsed.second + static_cast<int>(offset), value, mask, hiword);
    }

    void write32_block(int block, int offset, uint32_t value, uint32_t mask, bool hiword) {
        visiong_hw_reg_access req{};
        req.size = sizeof(req);
        req.block = static_cast<uint32_t>(block);
        req.offset = static_cast<uint32_t>(offset);
        req.value = value;
        req.mask = mask;
        req.flags = hiword ? VISIONG_HW_REG_FLAG_HIWORD_UPDATE : 0U;
        if (::ioctl(require_fd(), VISIONG_HW_REG_WRITE, &req) < 0) {
            throw_os_error("VISIONG_HW_REG_WRITE failed");
        }
    }

    void update32(const py::object& block, uint32_t offset, uint32_t mask, uint32_t value) {
        write32(block, offset, value, mask, false);
    }

    std::shared_ptr<PyHWReg> reg(const py::object& block) {
        const auto parsed = parse_block_offset(block);
        return std::make_shared<PyHWReg>(shared_from_this(), parsed.first, parsed.second);
    }

    std::shared_ptr<PyHWDmaBuffer> dma_alloc(int size, bool write_combine);
    void dma_sync(const py::object& buffer, py::object direction, uint32_t flags, int offset, int size_arg) const;
    int dma_fill(const py::object& buffer, uint32_t value, py::object size_arg, int offset) const;
    py::object dma_memcpy(const py::object& dst,
                          const py::object& src,
                          py::object size_arg,
                          int dst_offset,
                          int src_offset,
                          bool wait);
    std::shared_ptr<PyHWIRQ> irq(py::object pin, py::object bank, py::object pin_index, py::object edge);
    std::shared_ptr<PyHWSPI> spi_open(py::object bus,
                                      py::object chip_select,
                                      int speed_hz,
                                      int width,
                                      int height,
                                      int rotation);
    py::object spi_reg_transfer(int bus,
                                int chip_select,
                                py::object tx_data,
                                int rx_len,
                                bool tx_only,
                                int speed_hz,
                                int source_clock_hz,
                                int mode,
                                int bits_per_word,
                                int dummy);
    void spi_reg_release(int bus) const {
        visiong_hw_spi_reg_release req{};
        req.size = sizeof(req);
        req.bus = static_cast<uint32_t>(bus);
        if (::ioctl(require_fd(), VISIONG_HW_SPI_REG_RELEASE, &req) < 0) {
            throw_os_error("VISIONG_HW_SPI_REG_RELEASE failed");
        }
    }

    void close() {
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

    std::string repr() const {
        std::ostringstream oss;
        oss << "HW(path='" << path << "', state='" << (fd_ >= 0 ? "open" : "unavailable") << "')";
        return oss.str();
    }

    std::string path;

private:
    int fd_ = -1;
};

uint32_t PyHWReg::read32(uint32_t offset) const {
    return hw_->read32_block(block_, base_offset_ + static_cast<int>(offset));
}

void PyHWReg::write32(uint32_t offset, uint32_t value, uint32_t mask, bool hiword) {
    hw_->write32_block(block_, base_offset_ + static_cast<int>(offset), value, mask, hiword);
}

class PyHWDmaCopy {
public:
    PyHWDmaCopy(std::shared_ptr<PyHW> hw, uint32_t handle, int size)
        : hw(std::move(hw)), handle(handle), size(size) {}

    bool wait(int timeout_ms) {
        if (done_) {
            return true;
        }
        visiong_hw_wait req{};
        req.size = sizeof(req);
        req.handle = handle;
        req.timeout_ms = timeout_ms;
        if (::ioctl(hw->require_fd(), VISIONG_HW_DMA_WAIT, &req) < 0) {
            throw_os_error("VISIONG_HW_DMA_WAIT failed");
        }
        timestamp_ns = (static_cast<uint64_t>(req.timestamp_ns_hi) << 32U) | req.timestamp_ns_lo;
        if (req.status == VISIONG_HW_DMA_MEMCPY_STATUS_TIMEOUT) {
            return false;
        }
        if (req.status != VISIONG_HW_DMA_MEMCPY_STATUS_DONE) {
            throw std::runtime_error("visiong-hw DMA memcpy wait failed with status " + std::to_string(req.status));
        }
        done_ = true;
        return true;
    }

    bool done() const { return done_; }

    std::shared_ptr<PyHW> hw;
    uint32_t handle = 0;
    int size = 0;
    uint64_t timestamp_ns = 0;

private:
    bool done_ = false;
};

std::shared_ptr<PyHWDmaBuffer> PyHW::dma_alloc(int size, bool write_combine) {
    visiong_hw_dma_alloc req{};
    req.size = sizeof(req);
    req.bytes = static_cast<uint32_t>(size);
    req.flags = write_combine ? VISIONG_HW_DMA_ALLOC_WRITE_COMBINE : 0U;
    req.fd = -1;
    if (::ioctl(require_fd(), VISIONG_HW_DMA_ALLOC, &req) < 0) {
        throw_os_error("VISIONG_HW_DMA_ALLOC failed");
    }
    if (req.fd < 0) {
        throw std::runtime_error("visiong-hw DMA allocation did not return a valid fd");
    }
    return std::make_shared<PyHWDmaBuffer>(shared_from_this(), static_cast<int>(req.bytes), req.fd);
}

void PyHW::dma_sync(const py::object& buffer, py::object direction, uint32_t flags, int offset, int size_arg) const {
    visiong_hw_dma_sync req{};
    req.size = sizeof(req);
    req.fd = fd_from_object(buffer);
    req.direction = static_cast<uint32_t>(dma_direction(direction));
    req.flags = flags;
    req.offset = static_cast<uint32_t>(offset);
    req.bytes = static_cast<uint32_t>(size_arg);
    if (::ioctl(require_fd(), VISIONG_HW_DMA_SYNC, &req) < 0) {
        throw_os_error("VISIONG_HW_DMA_SYNC failed");
    }
}

int PyHW::dma_fill(const py::object& buffer, uint32_t value, py::object size_arg, int offset) const {
    int bytes = size_arg.is_none() ? -1 : size_arg.cast<int>();
    if (bytes < 0) {
        const int object_size = size_from_object(buffer);
        if (object_size < 0) {
            throw std::invalid_argument("dma_fill size is required when buffer is a raw fd");
        }
        bytes = object_size - offset;
    }
    if (bytes <= 0) {
        throw std::invalid_argument("dma_fill size must be positive");
    }
    visiong_hw_dma_fill req{};
    req.size = sizeof(req);
    req.fd = fd_from_object(buffer);
    req.offset = static_cast<uint32_t>(offset);
    req.bytes = static_cast<uint32_t>(bytes);
    req.value = value & 0xFFU;
    if (::ioctl(require_fd(), VISIONG_HW_DMA_FILL, &req) < 0) {
        throw_os_error("VISIONG_HW_DMA_FILL failed");
    }
    if (req.status != VISIONG_HW_DMA_MEMCPY_STATUS_DONE) {
        throw std::runtime_error("visiong-hw DMA fill failed with status " + std::to_string(req.status));
    }
    return bytes;
}

py::object PyHW::dma_memcpy(const py::object& dst,
                            const py::object& src,
                            py::object size_arg,
                            int dst_offset,
                            int src_offset,
                            bool wait) {
    int bytes = size_arg.is_none() ? -1 : size_arg.cast<int>();
    if (bytes < 0) {
        const int dst_size = size_from_object(dst);
        const int src_size = size_from_object(src);
        if (dst_size < 0 || src_size < 0) {
            throw std::invalid_argument("dma_memcpy size is required when dst/src are raw fds");
        }
        bytes = std::min(dst_size - dst_offset, src_size - src_offset);
    }
    if (bytes <= 0) {
        throw std::invalid_argument("dma_memcpy size must be positive");
    }
    visiong_hw_dma_memcpy req{};
    req.size = sizeof(req);
    req.dst_fd = fd_from_object(dst);
    req.src_fd = fd_from_object(src);
    req.dst_offset = static_cast<uint32_t>(dst_offset);
    req.src_offset = static_cast<uint32_t>(src_offset);
    req.bytes = static_cast<uint32_t>(bytes);
    req.flags = wait ? 0U : VISIONG_HW_DMA_MEMCPY_ASYNC;
    if (::ioctl(require_fd(), VISIONG_HW_DMA_MEMCPY, &req) < 0) {
        throw_os_error("VISIONG_HW_DMA_MEMCPY failed");
    }
    if (wait) {
        if (req.status != VISIONG_HW_DMA_MEMCPY_STATUS_DONE) {
            throw std::runtime_error("visiong-hw DMA memcpy failed with status " + std::to_string(req.status));
        }
        return py::int_(bytes);
    }
    if (req.status != VISIONG_HW_DMA_MEMCPY_STATUS_PENDING || req.handle == 0) {
        throw std::runtime_error("visiong-hw DMA memcpy failed with status " + std::to_string(req.status));
    }
    return py::cast(std::make_shared<PyHWDmaCopy>(shared_from_this(), req.handle, bytes));
}

py::object PyHWDmaBuffer::mmap(py::object access) {
    py::module_ mmap_module = py::module_::import("mmap");
    if (access.is_none()) {
        access = mmap_module.attr("ACCESS_WRITE");
    }
    return mmap_module.attr("mmap")(fd, size, py::arg("access") = access);
}

void PyHWDmaBuffer::sync_for_cpu(py::object direction, int offset, int size_arg) {
    hw->dma_sync(py::int_(fd), std::move(direction), VISIONG_HW_DMA_SYNC_START, offset, size_arg);
}

void PyHWDmaBuffer::sync_for_device(py::object direction, int offset, int size_arg) {
    hw->dma_sync(py::int_(fd), std::move(direction), VISIONG_HW_DMA_SYNC_END, offset, size_arg);
}

int PyHWDmaBuffer::fill(uint32_t value, py::object size_arg, int offset) {
    if (size_arg.is_none()) {
        size_arg = py::int_(size - offset);
    }
    return hw->dma_fill(py::int_(fd), value, std::move(size_arg), offset);
}

class PyHWIRQEvent {
public:
    PyHWIRQEvent(uint32_t sequence, uint64_t timestamp_ns, int bank, int pin)
        : sequence(sequence), timestamp_ns(timestamp_ns), bank(bank), pin(pin) {}
    uint32_t sequence = 0;
    uint64_t timestamp_ns = 0;
    int bank = 0;
    int pin = 0;
};

class PyHWIRQ {
public:
    PyHWIRQ(std::shared_ptr<PyHW> hw, uint32_t handle, int bank, int pin, std::string edge)
        : hw(std::move(hw)), handle(handle), bank(bank), pin(pin), edge(std::move(edge)) {}
    ~PyHWIRQ() { close(); }

    py::object wait(int timeout_ms) {
        if (handle == 0) {
            throw std::runtime_error("HWIRQ is closed");
        }
        visiong_hw_wait req{};
        req.size = sizeof(req);
        req.handle = handle;
        req.timeout_ms = timeout_ms;
        req.status = sequence;
        if (::ioctl(hw->require_fd(), VISIONG_HW_IRQ_WAIT, &req) < 0) {
            throw_os_error("VISIONG_HW_IRQ_WAIT failed");
        }
        const uint64_t ts = (static_cast<uint64_t>(req.timestamp_ns_hi) << 32U) | req.timestamp_ns_lo;
        if (req.status == sequence && ts == 0) {
            return py::none();
        }
        sequence = req.status;
        return py::cast(PyHWIRQEvent(sequence, ts, bank, pin));
    }

    void close() {
        if (handle == 0 || !hw || !hw->available()) {
            handle = 0;
            return;
        }
        visiong_hw_wait req{};
        req.size = sizeof(req);
        req.handle = handle;
        if (::ioctl(hw->require_fd(), VISIONG_HW_IRQ_RELEASE, &req) < 0) {
            handle = 0;
            throw_os_error("VISIONG_HW_IRQ_RELEASE failed");
        }
        handle = 0;
    }

    std::shared_ptr<PyHW> hw;
    uint32_t handle = 0;
    int bank = 0;
    int pin = 0;
    std::string edge;
    uint32_t sequence = 0;
};

std::shared_ptr<PyHWIRQ> PyHW::irq(py::object pin, py::object bank, py::object pin_index, py::object edge) {
    int bank_value = bank.is_none() ? -1 : bank.cast<int>();
    int pin_value = pin_index.is_none() ? -1 : pin_index.cast<int>();
    if (!pin.is_none()) {
        if (py::isinstance<py::tuple>(pin) || py::isinstance<py::list>(pin)) {
            py::sequence seq = pin.cast<py::sequence>();
            if (seq.size() != 2) {
                throw std::invalid_argument("irq pin tuple must be (bank, pin_index)");
            }
            bank_value = seq[0].cast<int>();
            pin_value = seq[1].cast<int>();
        } else {
            py::object parser = py::module_::import("visiong").attr("_parse_gpio_pin");
            py::tuple parsed = parser(pin).cast<py::tuple>();
            bank_value = parsed[0].cast<int>();
            pin_value = parsed[1].cast<int>();
        }
    }
    if (bank_value < 0 || pin_value < 0) {
        throw std::invalid_argument("irq() requires pin='GPIO1_C3' or bank=1, pin_index=19");
    }
    visiong_hw_irq_request req{};
    req.size = sizeof(req);
    req.bank = static_cast<uint32_t>(bank_value);
    req.pin = static_cast<uint32_t>(pin_value);
    req.edge = static_cast<uint32_t>(irq_edge(edge));
    if (::ioctl(require_fd(), VISIONG_HW_IRQ_REQUEST, &req) < 0) {
        throw_os_error("VISIONG_HW_IRQ_REQUEST failed");
    }
    return std::make_shared<PyHWIRQ>(shared_from_this(), req.handle, bank_value, pin_value, py::str(edge).cast<std::string>());
}

class PyHWSPITransfer {
public:
    PyHWSPITransfer(std::shared_ptr<PyHWSPI> spi, uint32_t handle, int size)
        : spi(std::move(spi)), handle(handle), size(size) {}

    bool wait(int timeout_ms);
    bool done() const { return done_; }

    std::shared_ptr<PyHWSPI> spi;
    uint32_t handle = 0;
    int size = 0;
    uint64_t timestamp_ns = 0;

private:
    bool done_ = false;
};

class PyHWSPI : public std::enable_shared_from_this<PyHWSPI> {
public:
    PyHWSPI(std::shared_ptr<PyHW> hw, uint32_t handle, int bus, int chip_select, int speed_hz)
        : hw(std::move(hw)), handle(handle), bus(bus), chip_select(chip_select), speed_hz(speed_hz) {}
    ~PyHWSPI() { close(); }

    std::shared_ptr<PyHWSPITransfer> submit_dma(const py::object& buffer, py::object size_arg, int offset, bool wait) {
        if (handle == 0) {
            throw std::runtime_error("HWSPI is closed");
        }
        int bytes = size_arg.is_none() ? -1 : size_arg.cast<int>();
        if (bytes < 0) {
            const int buffer_size = size_from_object(buffer);
            if (buffer_size < 0) {
                throw std::invalid_argument("submit_dma size is required when buffer is a raw fd");
            }
            bytes = buffer_size - offset;
        }
        if (bytes <= 0) {
            throw std::invalid_argument("submit_dma size must be positive");
        }
        visiong_hw_spi_display_submit req{};
        req.size = sizeof(req);
        req.handle = handle;
        req.dmabuf_fd = fd_from_object(buffer);
        req.offset = static_cast<uint32_t>(offset);
        req.bytes = static_cast<uint32_t>(bytes);
        if (::ioctl(hw->require_fd(), VISIONG_HW_SPI_DISPLAY_SUBMIT, &req) < 0) {
            throw_os_error("VISIONG_HW_SPI_DISPLAY_SUBMIT failed");
        }
        if (req.job_handle == 0) {
            throw std::runtime_error("visiong-hw SPI submit did not return a job handle");
        }
        auto transfer = std::make_shared<PyHWSPITransfer>(shared_from_this(), req.job_handle, bytes);
        if (wait) {
            transfer->wait(-1);
        }
        return transfer;
    }

    py::object write_dma(const py::object& buffer, py::object size_arg, int offset, bool wait) {
        auto transfer = submit_dma(buffer, size_arg, offset, wait);
        if (!wait) {
            return py::cast(transfer);
        }
        if (!size_arg.is_none()) {
            return py::int_(size_arg.cast<int>());
        }
        return py::int_(size_from_object(buffer) - offset);
    }

    void close() {
        if (handle == 0 || !hw || !hw->available()) {
            handle = 0;
            return;
        }
        visiong_hw_wait req{};
        req.size = sizeof(req);
        req.handle = handle;
        if (::ioctl(hw->require_fd(), VISIONG_HW_SPI_DISPLAY_CLOSE, &req) < 0) {
            handle = 0;
            throw_os_error("VISIONG_HW_SPI_DISPLAY_CLOSE failed");
        }
        handle = 0;
    }

    std::shared_ptr<PyHW> hw;
    uint32_t handle = 0;
    int bus = 0;
    int chip_select = 0;
    int speed_hz = 0;
};

bool PyHWSPITransfer::wait(int timeout_ms) {
    if (done_) {
        return true;
    }
    visiong_hw_wait req{};
    req.size = sizeof(req);
    req.handle = handle;
    req.timeout_ms = timeout_ms;
    if (::ioctl(spi->hw->require_fd(), VISIONG_HW_SPI_DISPLAY_WAIT, &req) < 0) {
        throw_os_error("VISIONG_HW_SPI_DISPLAY_WAIT failed");
    }
    timestamp_ns = (static_cast<uint64_t>(req.timestamp_ns_hi) << 32U) | req.timestamp_ns_lo;
    if (req.status == VISIONG_HW_DMA_MEMCPY_STATUS_TIMEOUT) {
        return false;
    }
    if (req.status != VISIONG_HW_DMA_MEMCPY_STATUS_DONE) {
        throw std::runtime_error("visiong-hw SPI transfer failed with status " + std::to_string(req.status));
    }
    done_ = true;
    return true;
}

std::shared_ptr<PyHWSPI> PyHW::spi_open(py::object bus,
                                        py::object chip_select,
                                        int speed_hz,
                                        int width,
                                        int height,
                                        int rotation) {
    const auto parsed = parse_spi_bus(bus, chip_select);
    visiong_hw_spi_display_open req{};
    req.size = sizeof(req);
    req.bus = static_cast<uint32_t>(parsed.first);
    req.chip_select = static_cast<uint32_t>(parsed.second);
    req.width = static_cast<uint32_t>(width);
    req.height = static_cast<uint32_t>(height);
    req.rotation = static_cast<uint32_t>(rotation);
    req.speed_hz = static_cast<uint32_t>(speed_hz);
    if (::ioctl(require_fd(), VISIONG_HW_SPI_DISPLAY_OPEN, &req) < 0) {
        throw_os_error("VISIONG_HW_SPI_DISPLAY_OPEN failed");
    }
    if (req.handle == 0) {
        throw std::runtime_error("visiong-hw SPI open did not return a handle");
    }
    return std::make_shared<PyHWSPI>(shared_from_this(), req.handle, parsed.first, parsed.second, speed_hz);
}

py::object PyHW::spi_reg_transfer(int bus,
                                  int chip_select,
                                  py::object tx_data,
                                  int rx_len,
                                  bool tx_only,
                                  int speed_hz,
                                  int source_clock_hz,
                                  int mode,
                                  int bits_per_word,
                                  int dummy) {
    std::vector<uint8_t> tx = bytes_from_object(tx_data);
    std::vector<uint8_t> rx(std::max(0, rx_len));
    visiong_hw_spi_reg_transfer req{};
    req.size = sizeof(req);
    req.bus = static_cast<uint32_t>(bus);
    req.chip_select = static_cast<uint32_t>(chip_select);
    req.speed_hz = static_cast<uint32_t>(speed_hz);
    req.source_clock_hz = static_cast<uint32_t>(source_clock_hz);
    req.mode = static_cast<uint32_t>(mode);
    req.bits_per_word = static_cast<uint32_t>(bits_per_word);
    req.flags = tx_only ? VISIONG_HW_SPI_REG_TX_ONLY : 0U;
    req.tx_ptr = tx.empty() ? 0 : reinterpret_cast<uintptr_t>(tx.data());
    req.rx_ptr = rx.empty() ? 0 : reinterpret_cast<uintptr_t>(rx.data());
    req.tx_len = static_cast<uint32_t>(tx.size());
    req.rx_len = static_cast<uint32_t>(rx.size());
    req.dummy = static_cast<uint32_t>(dummy) & 0xFFU;
    if (::ioctl(require_fd(), VISIONG_HW_SPI_REG_TRANSFER, &req) < 0) {
        throw_os_error("VISIONG_HW_SPI_REG_TRANSFER failed");
    }
    if (req.status != VISIONG_HW_DMA_MEMCPY_STATUS_DONE) {
        throw std::runtime_error("visiong-hw SPI register transfer failed with status " + std::to_string(req.status));
    }
    if (tx_only) {
        return py::int_(req.transferred);
    }
    return py::bytes(reinterpret_cast<const char*>(rx.data()), rx.size());
}

std::string gpio_event_pin(int bank, int pin) {
    const char group = static_cast<char>('A' + pin / 8);
    return "GPIO" + std::to_string(bank) + "_" + group + std::to_string(pin % 8);
}

void set_hw_constants(py::object cls) {
    cls.attr("FEATURE_REG_ACCESS") = VISIONG_HW_FEATURE_REG_ACCESS;
    cls.attr("FEATURE_PIN_SESSION") = VISIONG_HW_FEATURE_PIN_SESSION;
    cls.attr("FEATURE_GPIO_IRQ") = VISIONG_HW_FEATURE_GPIO_IRQ;
    cls.attr("FEATURE_DMA_BUFFER") = VISIONG_HW_FEATURE_DMA_BUFFER;
    cls.attr("FEATURE_DMA_MEMCPY") = VISIONG_HW_FEATURE_DMA_MEMCPY;
    cls.attr("FEATURE_SPI_DISPLAY") = VISIONG_HW_FEATURE_SPI_DISPLAY;
    cls.attr("FEATURE_DMA_FILL") = VISIONG_HW_FEATURE_DMA_FILL;
    cls.attr("FEATURE_SPI_REG") = VISIONG_HW_FEATURE_SPI_REG;
    cls.attr("REG_FLAG_HIWORD_UPDATE") = VISIONG_HW_REG_FLAG_HIWORD_UPDATE;
    cls.attr("DMA_ALLOC_WRITE_COMBINE") = VISIONG_HW_DMA_ALLOC_WRITE_COMBINE;
    cls.attr("DMA_SYNC_BIDIRECTIONAL") = VISIONG_HW_DMA_SYNC_BIDIRECTIONAL;
    cls.attr("DMA_SYNC_TO_DEVICE") = VISIONG_HW_DMA_SYNC_TO_DEVICE;
    cls.attr("DMA_SYNC_FROM_DEVICE") = VISIONG_HW_DMA_SYNC_FROM_DEVICE;
    cls.attr("DMA_SYNC_START") = VISIONG_HW_DMA_SYNC_START;
    cls.attr("DMA_SYNC_END") = VISIONG_HW_DMA_SYNC_END;
    cls.attr("DMA_MEMCPY_ASYNC") = VISIONG_HW_DMA_MEMCPY_ASYNC;
    cls.attr("DMA_MEMCPY_STATUS_DONE") = VISIONG_HW_DMA_MEMCPY_STATUS_DONE;
    cls.attr("DMA_MEMCPY_STATUS_TIMEOUT") = VISIONG_HW_DMA_MEMCPY_STATUS_TIMEOUT;
    cls.attr("DMA_MEMCPY_STATUS_ERROR") = VISIONG_HW_DMA_MEMCPY_STATUS_ERROR;
    cls.attr("DMA_MEMCPY_STATUS_PENDING") = VISIONG_HW_DMA_MEMCPY_STATUS_PENDING;
    cls.attr("SPI_REG_TX_ONLY") = VISIONG_HW_SPI_REG_TX_ONLY;
    cls.attr("IRQ_EDGE_RISING") = VISIONG_HW_IRQ_EDGE_RISING;
    cls.attr("IRQ_EDGE_FALLING") = VISIONG_HW_IRQ_EDGE_FALLING;
    cls.attr("IRQ_EDGE_BOTH") = VISIONG_HW_IRQ_EDGE_BOTH;
    cls.attr("_PATH") = kDefaultHwPath;
}

}  // namespace

void bind_hw(py::module_& m) {
    py::class_<PyHWReg, std::shared_ptr<PyHWReg>>(m, "HWReg")
        .def("read32", &PyHWReg::read32, "offset"_a)
        .def("write32", &PyHWReg::write32, "offset"_a, "value"_a, "mask"_a = 0xFFFFFFFFU, "hiword"_a = false)
        .def("update32", &PyHWReg::update32, "offset"_a, "mask"_a, "value"_a)
        .def_property_readonly("block", &PyHWReg::block)
        .def_property_readonly("base_offset", &PyHWReg::base_offset)
        .def("__repr__", [](const PyHWReg& self) {
            std::ostringstream oss;
            oss << "HWReg(block='" << self.block() << "', base_offset=0x" << std::hex << self.base_offset() << ")";
            return oss.str();
        });

    py::class_<PyHWDmaBuffer, std::shared_ptr<PyHWDmaBuffer>>(m, "HWDmaBuffer")
        .def("mmap", &PyHWDmaBuffer::mmap, "access"_a = py::none())
        .def("sync_for_cpu", &PyHWDmaBuffer::sync_for_cpu, "direction"_a = py::str("from_device"), "offset"_a = 0, "size"_a = 0)
        .def("sync_for_device", &PyHWDmaBuffer::sync_for_device, "direction"_a = py::str("to_device"), "offset"_a = 0, "size"_a = 0)
        .def("fill", &PyHWDmaBuffer::fill, "value"_a = 0, "size"_a = py::none(), "offset"_a = 0)
        .def("close", &PyHWDmaBuffer::close)
        .def_property_readonly("size", [](const PyHWDmaBuffer& self) { return self.size; })
        .def_property_readonly("fd", [](const PyHWDmaBuffer& self) { return self.fd; })
        .def("__enter__", [](std::shared_ptr<PyHWDmaBuffer> self) { return self; })
        .def("__exit__", [](PyHWDmaBuffer& self, py::object, py::object, py::object) {
            self.close();
            return false;
        })
        .def("__repr__", [](const PyHWDmaBuffer& self) {
            return "HWDmaBuffer(size=" + std::to_string(self.size) + ", fd=" + std::to_string(self.fd) + ")";
        });

    py::class_<PyHWDmaCopy, std::shared_ptr<PyHWDmaCopy>>(m, "HWDmaCopy")
        .def("wait", &PyHWDmaCopy::wait, "timeout_ms"_a = -1)
        .def("done", &PyHWDmaCopy::done)
        .def_property_readonly("handle", [](const PyHWDmaCopy& self) { return self.handle; })
        .def_property_readonly("size", [](const PyHWDmaCopy& self) { return self.size; })
        .def_property_readonly("timestamp_ns", [](const PyHWDmaCopy& self) { return self.timestamp_ns; })
        .def("__repr__", [](const PyHWDmaCopy& self) {
            return "HWDmaCopy(handle=" + std::to_string(self.handle) + ", size=" + std::to_string(self.size) +
                   ", state='" + (self.done() ? std::string("done") : std::string("pending")) + "')";
        });

    py::class_<PyHWIRQEvent>(m, "HWIRQEvent")
        .def(py::init<uint32_t, uint64_t, int, int>(), "sequence"_a, "timestamp_ns"_a, "bank"_a, "pin"_a)
        .def_readwrite("sequence", &PyHWIRQEvent::sequence)
        .def_readwrite("timestamp_ns", &PyHWIRQEvent::timestamp_ns)
        .def_readwrite("bank", &PyHWIRQEvent::bank)
        .def_readwrite("pin", &PyHWIRQEvent::pin)
        .def("__bool__", [](const PyHWIRQEvent& self) { return self.sequence > 0; })
        .def("__repr__", [](const PyHWIRQEvent& self) {
            return "HWIRQEvent(sequence=" + std::to_string(self.sequence) +
                   ", timestamp_ns=" + std::to_string(self.timestamp_ns) +
                   ", pin='" + gpio_event_pin(self.bank, self.pin) + "')";
        });

    py::class_<PyHWIRQ, std::shared_ptr<PyHWIRQ>>(m, "HWIRQ")
        .def("wait", &PyHWIRQ::wait, "timeout_ms"_a = -1)
        .def("close", &PyHWIRQ::close)
        .def_property_readonly("handle", [](const PyHWIRQ& self) { return self.handle; })
        .def_property_readonly("bank", [](const PyHWIRQ& self) { return self.bank; })
        .def_property_readonly("pin", [](const PyHWIRQ& self) { return self.pin; })
        .def_property_readonly("edge", [](const PyHWIRQ& self) { return self.edge; })
        .def_property_readonly("sequence", [](const PyHWIRQ& self) { return self.sequence; })
        .def("__repr__", [](const PyHWIRQ& self) {
            return "HWIRQ(handle=" + std::to_string(self.handle) + ", pin='" + gpio_event_pin(self.bank, self.pin) +
                   "', edge='" + self.edge + "')";
        });

    py::class_<PyHWSPITransfer, std::shared_ptr<PyHWSPITransfer>>(m, "HWSPITransfer")
        .def("wait", &PyHWSPITransfer::wait, "timeout_ms"_a = -1)
        .def("done", &PyHWSPITransfer::done)
        .def_property_readonly("handle", [](const PyHWSPITransfer& self) { return self.handle; })
        .def_property_readonly("size", [](const PyHWSPITransfer& self) { return self.size; })
        .def_property_readonly("timestamp_ns", [](const PyHWSPITransfer& self) { return self.timestamp_ns; })
        .def("__repr__", [](const PyHWSPITransfer& self) {
            return "HWSPITransfer(handle=" + std::to_string(self.handle) + ", size=" + std::to_string(self.size) +
                   ", state='" + (self.done() ? std::string("done") : std::string("pending")) + "')";
        });

    py::class_<PyHWSPI, std::shared_ptr<PyHWSPI>>(m, "HWSPI")
        .def("submit_dma", &PyHWSPI::submit_dma, "buffer"_a, "size"_a = py::none(), "offset"_a = 0, "wait"_a = false)
        .def("write_dma", &PyHWSPI::write_dma, "buffer"_a, "size"_a = py::none(), "offset"_a = 0, "wait"_a = true)
        .def("close", &PyHWSPI::close)
        .def_property_readonly("handle", [](const PyHWSPI& self) { return self.handle; })
        .def_property_readonly("bus", [](const PyHWSPI& self) { return self.bus; })
        .def_property_readonly("chip_select", [](const PyHWSPI& self) { return self.chip_select; })
        .def_property_readonly("speed_hz", [](const PyHWSPI& self) { return self.speed_hz; })
        .def("__enter__", [](std::shared_ptr<PyHWSPI> self) { return self; })
        .def("__exit__", [](PyHWSPI& self, py::object, py::object, py::object) {
            self.close();
            return false;
        })
        .def("__repr__", [](const PyHWSPI& self) {
            return "HWSPI(handle=" + std::to_string(self.handle) + ", bus='spi" + std::to_string(self.bus) + "." +
                   std::to_string(self.chip_select) + "', speed_hz=" + std::to_string(self.speed_hz) + ")";
        });

    auto hw = py::class_<PyHW, std::shared_ptr<PyHW>>(m, "HW")
                  .def(py::init([](py::object path, bool required, bool autoload) {
                           return std::make_shared<PyHW>(path.is_none() ? std::string() : py::str(path).cast<std::string>(),
                                                         required,
                                                         autoload);
                       }),
                       "path"_a = py::none(),
                       "required"_a = false,
                       "autoload"_a = true)
                  .def_static("is_available",
                              [](py::object path) {
                                  return PyHW::is_available(path.is_none() ? std::string(kDefaultHwPath)
                                                                           : py::str(path).cast<std::string>());
                              },
                              "path"_a = py::none())
                  .def_static("module_candidates", &PyHW::module_candidates, "module_path"_a = py::none())
                  .def_static("load", &PyHW::load, "module_path"_a = py::none())
                  .def_static("ensure_loaded", &PyHW::load, "module_path"_a = py::none())
                  .def_static("_block_offset", [](py::object block) {
                      const auto parsed = parse_block_offset(block);
                      return py::make_tuple(parsed.first, parsed.second);
                  })
                  .def_static("_dma_direction", &dma_direction, "direction"_a)
                  .def_static("_irq_edge", &irq_edge, "edge"_a)
                  .def("available", &PyHW::available)
                  .def("_require_fd", &PyHW::require_fd)
                  .def("caps", &PyHW::caps)
                  .def("read32", &PyHW::read32, "block"_a, "offset"_a)
                  .def("write32", &PyHW::write32, "block"_a, "offset"_a, "value"_a, "mask"_a = 0xFFFFFFFFU, "hiword"_a = false)
                  .def("update32", &PyHW::update32, "block"_a, "offset"_a, "mask"_a, "value"_a)
                  .def("reg", &PyHW::reg, "block"_a)
                  .def("dma_alloc", &PyHW::dma_alloc, "size"_a, "write_combine"_a = false)
                  .def("dma_sync_for_cpu",
                       [](PyHW& self, py::object buffer, py::object direction, int offset, int size) {
                           self.dma_sync(buffer, direction, VISIONG_HW_DMA_SYNC_START, offset, size);
                       },
                       "buffer"_a,
                       "direction"_a = py::str("from_device"),
                       "offset"_a = 0,
                       "size"_a = 0)
                  .def("dma_sync_for_device",
                       [](PyHW& self, py::object buffer, py::object direction, int offset, int size) {
                           self.dma_sync(buffer, direction, VISIONG_HW_DMA_SYNC_END, offset, size);
                       },
                       "buffer"_a,
                       "direction"_a = py::str("to_device"),
                       "offset"_a = 0,
                       "size"_a = 0)
                  .def("dma_memcpy",
                       &PyHW::dma_memcpy,
                       "dst"_a,
                       "src"_a,
                       "size"_a = py::none(),
                       "dst_offset"_a = 0,
                       "src_offset"_a = 0,
                       "wait"_a = true)
                  .def("dma_submit_memcpy",
                       [](PyHW& self, py::object dst, py::object src, py::object size, int dst_offset, int src_offset) {
                           return self.dma_memcpy(dst, src, size, dst_offset, src_offset, false);
                       },
                       "dst"_a,
                       "src"_a,
                       "size"_a = py::none(),
                       "dst_offset"_a = 0,
                       "src_offset"_a = 0)
                  .def("dma_fill", &PyHW::dma_fill, "buffer"_a, "value"_a = 0, "size"_a = py::none(), "offset"_a = 0)
                  .def("irq", &PyHW::irq, "pin"_a = py::none(), "bank"_a = py::none(), "pin_index"_a = py::none(), "edge"_a = py::str("both"))
                  .def("spi_open",
                       &PyHW::spi_open,
                       "bus"_a = py::str("spi0.0"),
                       "chip_select"_a = py::none(),
                       "speed_hz"_a = 24000000,
                       "width"_a = 0,
                       "height"_a = 0,
                       "rotation"_a = 0)
                  .def("_spi_reg_transfer",
                       &PyHW::spi_reg_transfer,
                       "bus"_a,
                       "chip_select"_a,
                       "tx_data"_a = py::bytes(""),
                       "rx_len"_a = 0,
                       "tx_only"_a = false,
                       "speed_hz"_a = 50000000,
                       "source_clock_hz"_a = 200000000,
                       "mode"_a = 0,
                       "bits_per_word"_a = 8,
                       "dummy"_a = 0xFF)
                  .def("_spi_reg_release", &PyHW::spi_reg_release, "bus"_a)
                  .def("close", &PyHW::close)
                  .def("__enter__", [](std::shared_ptr<PyHW> self) {
                      self->require_fd();
                      return self;
                  })
                  .def("__exit__", [](PyHW& self, py::object, py::object, py::object) {
                      self.close();
                      return false;
                  })
                  .def("__repr__", &PyHW::repr)
                  .def_readonly("path", &PyHW::path);
    set_hw_constants(hw);

    m.def("_spi_reg_pio_transfer_native",
          &spi_reg_pio_transfer_native,
          "bus"_a,
          "chip_select"_a,
          "tx_data"_a = py::bytes(""),
          "rx_len"_a = 0,
          "tx_only"_a = false,
          "speed_hz"_a = 50000000,
          "source_clock_hz"_a = 200000000,
          "mode"_a = 0,
          "bits_per_word"_a = 8,
          "dummy"_a = 0xff,
          "Native RV1103/RV1106 SPI register PIO transfer fallback.");
    m.def("_uart_reg_write_native",
          &uart_reg_write_native,
          "bus"_a,
          "data"_a,
          "timeout"_a = 0.0,
          "Native RV1103/RV1106 UART register TX fallback.");
    m.def("_uart_reg_read_native",
          &uart_reg_read_native,
          "bus"_a,
          "nbytes"_a = 1,
          "timeout"_a = 0.0,
          "Native RV1103/RV1106 UART register RX fallback.");
    m.def("_uart_reg_any_native",
          &uart_reg_any_native,
          "bus"_a,
          "Native RV1103/RV1106 UART RX FIFO level read.");
    m.def("_i2c_reg_writeto_native",
          &i2c_reg_writeto_native,
          "bus"_a,
          "addr"_a,
          "data"_a,
          "tuning"_a,
          "timeout"_a = 1.0,
          "Native RV1103/RV1106 I2C register write fallback.");
    m.def("_i2c_reg_readfrom_native",
          &i2c_reg_readfrom_native,
          "bus"_a,
          "addr"_a,
          "nbytes"_a,
          "memaddr"_a = py::bytes(""),
          "tuning"_a = 0,
          "timeout"_a = 1.0,
          "Native RV1103/RV1106 I2C register read fallback.");
}
