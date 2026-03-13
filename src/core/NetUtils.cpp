// SPDX-License-Identifier: LGPL-3.0-or-later
#include "visiong/core/NetUtils.h"

#include <algorithm>

#include <cstring>
#include <ifaddrs.h>
#include <net/if.h>
#include <netdb.h>

namespace visiong {

std::vector<std::string> get_local_ipv4_addresses(bool include_loopback) {
    std::vector<std::string> addrs;
    struct ifaddrs* ifaddr = nullptr;
    if (getifaddrs(&ifaddr) != 0 || !ifaddr)
        return addrs;

    for (struct ifaddrs* ifa = ifaddr; ifa; ifa = ifa->ifa_next) {
        if (!ifa->ifa_addr)
            continue;
        if (ifa->ifa_addr->sa_family != AF_INET)
            continue;

        // Only UP interfaces / 仅 UP interfaces
        if (!(ifa->ifa_flags & IFF_UP))
            continue;

        // Skip loopback unless requested / 跳过 loopback unless requested
        if (!include_loopback && (ifa->ifa_flags & IFF_LOOPBACK))
            continue;

        char host[NI_MAXHOST] = {0};
        if (0 == getnameinfo(ifa->ifa_addr, sizeof(struct sockaddr_in), host, NI_MAXHOST, nullptr, 0,
                             NI_NUMERICHOST)) {
            if (host[0] != '\0')
                addrs.emplace_back(host);
        }
    }

    freeifaddrs(ifaddr);

    std::sort(addrs.begin(), addrs.end());
    addrs.erase(std::unique(addrs.begin(), addrs.end()), addrs.end());
    return addrs;
}

} // namespace visiong

