// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_CORE_NETUTILS_H
#define VISIONG_CORE_NETUTILS_H

#include <string>
#include <vector>

namespace visiong {

/// Get local IPv4 addresses of UP (non-loopback by default) interfaces. / Get local IPv4 addresses 的 UP (non-loopback 由 default) interfaces.
/// Returns numeric dotted strings like "192.168.1.10".
std::vector<std::string> get_local_ipv4_addresses(bool include_loopback = false);

} // namespace visiong

#endif  // VISIONG_CORE_NETUTILS_H

