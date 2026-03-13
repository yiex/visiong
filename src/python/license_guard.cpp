// SPDX-License-Identifier: LGPL-3.0-or-later
#include "python/internal/license_guard.h"

#include <array>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

namespace visiong::python {
namespace {

constexpr const char* kUnavailableChipId = "000000000000";
constexpr const char* kOtpPath = "/sys/devices/platform/ff3d0000.otp/rockchip-otp0/nvmem";

} // namespace

void verify_license_once() {
    // No-op in the open-source build. / No-op 在 open-source build.
}

bool is_license_valid() {
    // The public open-source build does not require a product license file. / 公共 open-source 构建 does 不 require product license file.
    return true;
}

std::string license_banner() {
    return community_banner();
}

std::string get_unique_id() {
    std::ifstream file(kOtpPath, std::ios::binary);
    if (!file) {
        return kUnavailableChipId;
    }

    file.seekg(10);
    std::array<char, 6> bytes{};
    file.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (static_cast<size_t>(file.gcount()) != bytes.size()) {
        return kUnavailableChipId;
    }

    std::stringstream stream;
    stream << std::hex << std::setfill('0');
    for (char value : bytes) {
        stream << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(value));
    }
    return stream.str();
}

const char* community_banner() {
    return R"(
+=================================+
| License: GNU LGPL-3.0-or-later  |
| https://github.com/yiex/visiong |
+=================================+
)";
}

} // namespace visiong::python

