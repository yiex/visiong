<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->
# Contributing

## 构建与验证 / Build and validation

构建依赖和交叉编译步骤见 [README](README.md)。硬件接口变更应在目标板运行，并记录固件、运行库、输入尺寸和验证结果。IVE 和 YOLO26 的复现步骤分别见 [IVE](docs/IVE.md) 和 [YOLO26](docs/YOLO26.md)。

Follow the README for dependencies and cross-compilation. Validate hardware interface changes on the target board and record firmware, runtime versions, input dimensions and results. See the linked IVE and YOLO26 guides for reproducible checks.

## 代码与许可 / Code and licensing

保持现有接口和代码风格，说明必要的接口变更。保留原始版权、SPDX 标记和第三方许可；新增依赖时同步更新第三方说明与依赖策略。提交中只包含源码及必要文档，运行库与模型按各自许可单独管理。

Follow existing interfaces and style, documenting any necessary API changes. Preserve copyright, SPDX headers and third-party licenses. Update notices and dependency policy when adding dependencies. Keep source changes and documentation separate from runtime libraries and model artifacts, which retain their own licenses.
