// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_PYTHON_LICENSE_GUARD_H_
#define VISIONG_PYTHON_LICENSE_GUARD_H_

#include <string>

namespace visiong::python {

// Open-source build compatibility layer. / Open-source 构建 compatibility layer.
//
// Historically this translation unit also carried product-license verification. / Historically 该 translation unit 也 carried product-license verification.
// The public open-source build intentionally removes that logic so the Python
// extension no longer depends on embedded cryptography or board-bound key files. / extension 不 longer depends 在 embedded cryptography 或 board-bound key files.
// The compatibility symbols remain to avoid breaking downstream code.
void verify_license_once();
bool is_license_valid();
std::string license_banner();
std::string get_unique_id();
const char* community_banner();

} // namespace visiong::python

#endif // VISIONG_PYTHON_LICENSE_GUARD_H_

