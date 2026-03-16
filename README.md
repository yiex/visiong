<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->
<a id="zh-cn"></a>

# VisionG

[简体中文](#zh-cn) | [English](#english)

VisionG 是一个面向 Rockchip RV1103 / RV1106 设备的、以 Python 为优先接口的计算机视觉运行时。
它把摄像头采集、显示输出、图像处理、Rockchip NPU 推理以及若干嵌入式设备辅助能力，
整理成了一套更适合板端开发和部署的 Python API。

项目许可证：GNU LGPL-3.0-or-later。

## 项目起源

**visiong** 库和我另一个开源硬件项目 **VisionG_SE** 以及 VS Code 插件库中名为 **vigide** 的项目均为我毕设的组成部分：

- VisionG_SE 硬件开源链接：<https://oshwhub.com/salieri_coin/visiong_se>

由于电赛时，K210 给我留下的阴影太大，同时受到 MaixCAM 的启发，于是打算做一款便宜且易用的 AI 视觉开发板。

visiong 库并未进行功能限制、绑定和收费。任何使用了 Luckfox SDK 所生成的固件、且未大改 `/oem` 文件夹、
并携带 Python 3.11 的 RV1103 / RV1106 开发板，均可使用 visiong 库和 vigide 插件。

## VisionG 提供什么

- 摄像头采集，支持 YUV、RGB/BGR、灰度输出
- JPEG、RTSP、HTTP-FLV、HTTP-MJPEG、Framebuffer、UDP 等显示/传输路径
- 硬件加速的格式转换、缩放、裁切、letterbox、绘制等图像操作
- Rockchip NPU 的 Python 封装与后处理辅助
- 面向板端脚本、快速验证和部署的 Python 绑定

## 开发板上 Python 部署方式

如果你只需要在板端用 Python，直接把下面两个文件放到：

```text
/usr/lib/python3.11/site-packages
```

文件：

- `visiong.py`
- `_visiong.so`

如果使用了 vigide 插件，直接使用 `Update Python Library` 上传 `visiong_python.zip` 压缩包即可。

VisionG 使用到的最小 OpenCV 子集已经静态链接进 `_visiong.so`，
因此不需要额外部署 `libopencv_*.so`。

## 运行时提醒

- VisionG 运行时会动态加载一部分 Rockchip 用户态库。
- 开发板本身仍然需要提供 SDK 镜像中常见的那些运行库，通常位于 `/oem/usr/lib`。
- 如果你在模型推理时遇到算子不支持、模型运行异常、或者 RKNN 行为不稳定，优先把 `librknnmrt.so` 更新到 Rockchip 官方最新版本。

库链接：

- <https://github.com/airockchip/rknn-toolkit2/blob/master/rknpu2/runtime/Linux/librknn_api/armhf-uclibc/librknnmrt.so>

## 本地构建前提

典型主机环境：

- Ubuntu 22.04
- `cmake`
- `ninja-build`
- `zip`
- `unzip`
- `git`

安装示例：

```bash
sudo apt-get update
sudo apt-get install -y \
  cmake \
  ninja-build \
  unzip \
  zip \
  git
```

## 构建依赖目录说明

`/opt/visiong-stage` 不是仓库自带目录，也不是一个空目录就可以直接使用。

它应该是一个**预先整理好的依赖 staging 目录**，用于给 VisionG 构建过程提供所需的头文件、库文件、
CMake 配置以及其他必要的依赖内容。这个目录通常由下面这些依赖整理而来：

- Luckfox SDK
- RKNN Toolkit2
- librga

也就是说，`build.sh` 里的：

```bash
--deps-root /opt/visiong-stage
```

指向的不是任意目录，而是一个已经准备好的依赖根目录。

## 如何准备 `/opt/visiong-stage`

先拉取依赖仓库：

```bash
git clone --depth=1 --branch main \
  https://github.com/LuckfoxTECH/luckfox-pico.git _deps/luckfox-pico

git clone --depth=1 --branch master \
  https://github.com/airockchip/rknn-toolkit2.git _deps/rknn-toolkit2

git clone --depth=1 --branch main \
  https://github.com/airockchip/librga.git _deps/librga
```

然后执行 staging 脚本，把这些依赖整理到一个统一目录中：

```bash
./scripts/stage_release_deps.sh \
  --out /opt/visiong-stage \
  --luckfox-sdk _deps/luckfox-pico \
  --rknn-toolkit2 _deps/rknn-toolkit2 \
  --librga _deps/librga
```

执行完成后，`/opt/visiong-stage` 就可以作为 `build.sh` 的 `--deps-root` 来使用。

如果你不希望写入 `/opt`，也可以使用项目内路径，例如：

```bash
./scripts/stage_release_deps.sh \
  --out "$PWD/_stage" \
  --luckfox-sdk _deps/luckfox-pico \
  --rknn-toolkit2 _deps/rknn-toolkit2 \
  --librga _deps/librga
```

随后在构建时改为：

```bash
./build.sh --deps-root "$PWD/_stage"
```

## 如何选择组件开关

你可以直接修改 [release/release_components.cmake](release/release_components.cmake)，或者在 `build.sh` 里显式传参：

```bash
./build.sh \
  --deps-root /opt/visiong-stage \
  --cmake-arg -DVISIONG_ENABLE_IVE=OFF \
  --cmake-arg -DVISIONG_ENABLE_NPU=ON \
  --cmake-arg -DVISIONG_ENABLE_GUI=OFF
```

常见开关说明：

- `VISIONG_ENABLE_IVE`：控制 IVE 模块
- `VISIONG_ENABLE_NPU`：控制 RKNN / NPU 模块
- `VISIONG_ENABLE_GUI`：控制 GUI 模块

## 一个完整的本地构建示例

下面是一套从准备依赖到生成压缩包的完整示例：

```bash
sudo apt-get update
sudo apt-get install -y \
  cmake \
  ninja-build \
  unzip \
  zip \
  git

git clone --depth=1 --branch main \
  https://github.com/LuckfoxTECH/luckfox-pico.git _deps/luckfox-pico

git clone --depth=1 --branch master \
  https://github.com/airockchip/rknn-toolkit2.git _deps/rknn-toolkit2

git clone --depth=1 --branch main \
  https://github.com/airockchip/librga.git _deps/librga

./scripts/stage_release_deps.sh \
  --out "$PWD/_stage" \
  --luckfox-sdk _deps/luckfox-pico \
  --rknn-toolkit2 _deps/rknn-toolkit2 \
  --librga _deps/librga

./build.sh \
  --deps-root "$PWD/_stage" \
  --no-package \
  --cmake-arg -DVISIONG_ENABLE_IVE=ON \
  --cmake-arg -DVISIONG_ENABLE_NPU=ON \
  --cmake-arg -DVISIONG_ENABLE_GUI=ON

mkdir -p dist
./scripts/create_release_archives.sh build dist visiong_cpp.zip visiong_python.zip
```

## 编译输出

成功编译后通常会得到：

- `build/_visiong.so`
- `build/visiong.py`
- `build/libvisiong.so`
- `build/libvisiong.a`
- `visiong_cpp.zip`
- `visiong_python.zip`

其中：

- `visiong_python.zip` 最适合板端 Python 部署
- `visiong_cpp.zip` 适合 C/C++ 调用或混合部署

当前 `visiong_cpp.zip` 只保留 visiong 自己的公开头、库文件、CMake 配置、说明文件和组件清单。
如果你的 C++ 工程直接使用了 `include/visiong` 中那些暴露 Rockchip 类型的接口，
请自行从对应 SDK 提供匹配版本的外部头文件。

## 仓库结构

- `include/visiong/`：公开头文件
- `src/core/`：摄像头、图像缓冲、像素格式、运行时辅助
- `src/modules/`：显示、ISP、IVE、GUI、流媒体、VENC 相关模块
- `src/npu/`：RKNN / NPU 封装和后处理
- `src/python/`：pybind11 绑定与 Python 加载器
- `3rdparty/`：构建实际使用到的宽松许可证第三方源码
- `scripts/`：staging、编译、发布、审计脚本
- `docs/`：发布策略与合规说明

## 发布边界

- GitHub public releases publish the VisionG binary packages.
- The prepared public binary deliverable is a dynamic-linked binary package set.
- Staged vendor libraries are not published.
- Required Rockchip runtime libraries must be present on the target.
- Vendor SDK bundles are not part of the public source archive.

## 致谢

我想郑重感谢所有为 VisionG 提供过帮助的作者、维护者和项目。

- 感谢 Rockchip 提供 RV1103 / RV1106 平台、媒体栈与 NPU 生态
- 感谢 LuckfoxTECH 公开发布 Luckfox SDK、开发板支持内容和例程
- 感谢 OpenCV 及其维护者提供扎实的图像处理基础
- 感谢 `opencv-mobile` 在嵌入式 OpenCV 裁切、打包方面的工作
- 感谢 `pybind11` 提供高质量的 Python 绑定基础设施
- 感谢 `media-server` 提供 RTSP / FLV / MOV 相关基础组件
- 感谢 `quirc` 提供二维码识别能力
- 感谢 `stb` 提供轻量的图像读写工具
- 感谢 `Nuklear` 提供即时模式 GUI 基础
- 也感谢所有公开分享过 RV1103 / RV1106 经验、例程、踩坑记录和调试心得的开发者

## 合规与说明

发布或镜像产物前，建议先阅读：

- [docs/OPEN_SOURCE_COMPLIANCE.md](docs/OPEN_SOURCE_COMPLIANCE.md)
- [docs/INCLUDE_LICENSE_AUDIT.md](docs/INCLUDE_LICENSE_AUDIT.md)
- [docs/DEPENDENCY_REDISTRIBUTION_MATRIX.md](docs/DEPENDENCY_REDISTRIBUTION_MATRIX.md)
- [docs/BINARY_REDISTRIBUTION_POLICY.md](docs/BINARY_REDISTRIBUTION_POLICY.md)
- [docs/BINARY_ARTIFACT_WORKFLOW.md](docs/BINARY_ARTIFACT_WORKFLOW.md)
- [docs/GITHUB_ACTIONS_RELEASE.md](docs/GITHUB_ACTIONS_RELEASE.md)

第三方说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

---

<a id="english"></a>

# VisionG

[简体中文](#zh-cn) | [English](#english)

VisionG is a Python-first computer-vision runtime for Rockchip RV1103 / RV1106 devices.
It organizes camera capture, display output, image processing, Rockchip NPU inference,
and several embedded-device helper capabilities into a Python API that is more practical
for on-board development and deployment.

Project license: GNU LGPL-3.0-or-later.

## Origin

The `visiong` library, my other open-hardware project **VisionG_SE**, and the VS Code plugin project
named **vigide** are all parts of my graduation project.

- VisionG_SE hardware project: <https://oshwhub.com/salieri_coin/visiong_se>

During the electronics contest, the K210 left too much of a negative impression on me.
At the same time, MaixCAM inspired me, so I decided to build an AI vision board that is both cheap and easy to use.

At the same time, the `visiong` library is not feature-gated and not charged.
Any RV1103 / RV1106 development board can use VisionG and the vigide plugin as long as it:

- uses firmware generated from the Luckfox SDK,
- does not heavily modify the `/oem` directory,
- provides Python 3.11 on the target.

## What VisionG Provides

- Camera capture with YUV, RGB/BGR, and grayscale output
- Display and transport paths including JPEG, RTSP, HTTP-FLV, HTTP-MJPEG, framebuffer, and UDP
- Hardware-accelerated image operations such as format conversion, resize, crop, letterbox, and drawing
- Python bindings for the Rockchip NPU plus post-processing helpers
- Python bindings intended for on-board scripting, rapid validation, and deployment

## Python Deployment on the Board

If you only need Python on the target board, place these two files into:

```text
/usr/lib/python3.11/site-packages
```

Files:

- `visiong.py`
- `_visiong.so`

If you use the vigide plugin, you can simply upload `visiong_python.zip` through `Update Python Library`.

The minimal OpenCV subset used by VisionG is already linked statically into `_visiong.so`,
so you do not need to deploy separate `libopencv_*.so` files.

## Runtime Notes

- VisionG dynamically loads part of the Rockchip user-space runtime libraries at execution time.
- The board itself must still provide the common runtime libraries that usually come with the SDK image, typically under `/oem/usr/lib`.
- If you hit unsupported operators, abnormal model execution, or unstable RKNN behavior during inference, update `librknnmrt.so` to the latest Rockchip-provided version first.

Library link:

- <https://github.com/airockchip/rknn-toolkit2/blob/master/rknpu2/runtime/Linux/librknn_api/armhf-uclibc/librknnmrt.so>

## Local Build Prerequisites

A typical host environment is:

- Ubuntu 22.04
- `cmake`
- `ninja-build`
- `zip`
- `unzip`
- `git`

Example installation:

```bash
sudo apt-get update
sudo apt-get install -y \
  cmake \
  ninja-build \
  unzip \
  zip \
  git
```

## About the Dependency Staging Directory

`/opt/visiong-stage` is not a directory that comes with this repository,
and it is not something you can use by just creating an empty folder.

It should be a **prepared dependency staging directory** that provides the headers,
libraries, CMake files, and related materials required by the VisionG build.
This staging directory is typically assembled from:

- Luckfox SDK
- RKNN Toolkit2
- librga

So when `build.sh` uses:

```bash
--deps-root /opt/visiong-stage
```

that path is expected to point to a prepared dependency root, not an arbitrary directory.

## How to Prepare `/opt/visiong-stage`

First, fetch the dependency repositories:

```bash
git clone --depth=1 --branch main \
  https://github.com/LuckfoxTECH/luckfox-pico.git _deps/luckfox-pico

git clone --depth=1 --branch master \
  https://github.com/airockchip/rknn-toolkit2.git _deps/rknn-toolkit2

git clone --depth=1 --branch main \
  https://github.com/airockchip/librga.git _deps/librga
```

Then run the staging script to assemble them into a single dependency root:

```bash
./scripts/stage_release_deps.sh \
  --out /opt/visiong-stage \
  --luckfox-sdk _deps/luckfox-pico \
  --rknn-toolkit2 _deps/rknn-toolkit2 \
  --librga _deps/librga
```

After that, `/opt/visiong-stage` can be used as the `--deps-root` for `build.sh`.

If you do not want to write into `/opt`, you can use a project-local path instead:

```bash
./scripts/stage_release_deps.sh \
  --out "$PWD/_stage" \
  --luckfox-sdk _deps/luckfox-pico \
  --rknn-toolkit2 _deps/rknn-toolkit2 \
  --librga _deps/librga
```

Then build with:

```bash
./build.sh --deps-root "$PWD/_stage"
```

## How to Choose Component Toggles

You can directly edit [release/release_components.cmake](release/release_components.cmake),
or pass explicit arguments into `build.sh`:

```bash
./build.sh \
  --deps-root /opt/visiong-stage \
  --cmake-arg -DVISIONG_ENABLE_IVE=OFF \
  --cmake-arg -DVISIONG_ENABLE_NPU=ON \
  --cmake-arg -DVISIONG_ENABLE_GUI=OFF
```

Common switches:

- `VISIONG_ENABLE_IVE`: controls the IVE module
- `VISIONG_ENABLE_NPU`: controls the RKNN / NPU module
- `VISIONG_ENABLE_GUI`: controls the GUI module

## A Complete Local Build Example

Below is a full example from dependency preparation to package generation:

```bash
sudo apt-get update
sudo apt-get install -y \
  cmake \
  ninja-build \
  unzip \
  zip \
  git

git clone --depth=1 --branch main \
  https://github.com/LuckfoxTECH/luckfox-pico.git _deps/luckfox-pico

git clone --depth=1 --branch master \
  https://github.com/airockchip/rknn-toolkit2.git _deps/rknn-toolkit2

git clone --depth=1 --branch main \
  https://github.com/airockchip/librga.git _deps/librga

./scripts/stage_release_deps.sh \
  --out "$PWD/_stage" \
  --luckfox-sdk _deps/luckfox-pico \
  --rknn-toolkit2 _deps/rknn-toolkit2 \
  --librga _deps/librga

./build.sh \
  --deps-root "$PWD/_stage" \
  --no-package \
  --cmake-arg -DVISIONG_ENABLE_IVE=ON \
  --cmake-arg -DVISIONG_ENABLE_NPU=ON \
  --cmake-arg -DVISIONG_ENABLE_GUI=ON

mkdir -p dist
./scripts/create_release_archives.sh build dist visiong_cpp.zip visiong_python.zip
```

## Build Outputs

A successful build usually produces:

- `build/_visiong.so`
- `build/visiong.py`
- `build/libvisiong.so`
- `build/libvisiong.a`
- `visiong_cpp.zip`
- `visiong_python.zip`

Among them:

- `visiong_python.zip` is the most suitable package for Python deployment on the board
- `visiong_cpp.zip` is suitable for C/C++ consumers or mixed deployment

The current `visiong_cpp.zip` only keeps VisionG public headers, library files, CMake package files,
notices, and the component manifest.
If your C++ project directly uses interfaces in `include/visiong` that expose Rockchip SDK types,
you should provide matching external SDK headers yourself from the corresponding SDK version.

## Repository Layout

- `include/visiong/`: public headers
- `src/core/`: camera, image buffer, pixel format, and runtime helpers
- `src/modules/`: display, ISP, IVE, GUI, streaming, and VENC-related modules
- `src/npu/`: RKNN / NPU wrappers and post-processing
- `src/python/`: pybind11 bindings and Python loader
- `3rdparty/`: permissively licensed third-party source code actually used by the build
- `scripts/`: staging, build, release, and audit scripts
- `docs/`: release strategy and compliance notes

## Release Boundary

- GitHub public releases publish the VisionG binary packages.
- The prepared public binary deliverable is a dynamic-linked binary package set.
- Staged vendor libraries are not published.
- Required Rockchip runtime libraries must be present on the target.
- Vendor SDK bundles are not part of the public source archive.

## Acknowledgements

I want to sincerely thank all authors, maintainers, and projects that have helped VisionG.

- Rockchip for providing the RV1103 / RV1106 platform, media stack, and NPU ecosystem
- LuckfoxTECH for publicly releasing the Luckfox SDK, board support materials, and examples
- OpenCV and its maintainers for the solid image-processing foundation
- `opencv-mobile` for its work on trimming and packaging OpenCV for embedded environments
- `pybind11` for high-quality Python binding infrastructure
- `media-server` for RTSP / FLV / MOV related building blocks
- `quirc` for QR code recognition capability
- `stb` for lightweight image reading and writing utilities
- `Nuklear` for the immediate-mode GUI foundation
- Everyone who publicly shared RV1103 / RV1106 experience, examples, pitfalls, and debugging notes

## Compliance and Notes

Before publishing or mirroring artifacts, it is recommended to read:

- [docs/OPEN_SOURCE_COMPLIANCE.md](docs/OPEN_SOURCE_COMPLIANCE.md)
- [docs/INCLUDE_LICENSE_AUDIT.md](docs/INCLUDE_LICENSE_AUDIT.md)
- [docs/DEPENDENCY_REDISTRIBUTION_MATRIX.md](docs/DEPENDENCY_REDISTRIBUTION_MATRIX.md)
- [docs/BINARY_REDISTRIBUTION_POLICY.md](docs/BINARY_REDISTRIBUTION_POLICY.md)
- [docs/BINARY_ARTIFACT_WORKFLOW.md](docs/BINARY_ARTIFACT_WORKFLOW.md)
- [docs/GITHUB_ACTIONS_RELEASE.md](docs/GITHUB_ACTIONS_RELEASE.md)

Third-party notices are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
