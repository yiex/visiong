// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_COMMON_LEGACYKEY_H
#define VISIONG_COMMON_LEGACYKEY_H

#include <string>

namespace visiong::legacy {

std::string decrypt_legacy_value(const std::string& key);

}  // namespace visiong::legacy

#endif  // VISIONG_COMMON_LEGACYKEY_H