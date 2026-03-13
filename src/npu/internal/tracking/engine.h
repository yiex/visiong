// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef VISIONG_NPU_INTERNAL_TRACKING_ENGINE_H
#define VISIONG_NPU_INTERNAL_TRACKING_ENGINE_H

#include "datatype.h"
#include "error.h"

#include <memory>
#include <vector>

class NNEngine {
public:
    virtual ~NNEngine() = default;

    virtual nn_error_e LoadModelFile(const char* model_file) = 0;
    virtual const std::vector<tensor_attr_s>& GetInputShapes() = 0;
    virtual const std::vector<tensor_attr_s>& GetOutputShapes() = 0;
    virtual nn_error_e Run(std::vector<tensor_data_s>& inputs,
                           std::vector<tensor_data_s>& outputs,
                           bool want_float) = 0;
};

std::shared_ptr<NNEngine> CreateRKNNEngine();

#endif  // VISIONG_NPU_INTERNAL_TRACKING_ENGINE_H
