// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_INTERNAL_IVE_RUNTIME_COMPAT_H
#define VISIONG_INTERNAL_IVE_RUNTIME_COMPAT_H

#include "rk_mpi_ive.h"

namespace visiong::ive_internal {

void require_gmm_separate_output_runtime();
RK_S32 gmm_separate_output(const IVE_IMAGE_S& source, const IVE_IMAGE_S* factor,
                          IVE_IMAGE_S& foreground, IVE_IMAGE_S& background,
                          IVE_IMAGE_S& matched, const IVE_MEM_INFO_S& model,
                          const IVE_MEM_INFO_S& output, int max_models, bool first_frame,
                          RK_U32 current_models);

}  // namespace visiong::ive_internal
#endif
