#!/bin/sh
# SPDX-License-Identifier: LGPL-3.0-or-later
# Place beside visiong.py, _visiong.so and the optional RKNN runtime.
# 与 visiong.py、_visiong.so 及可选的 RKNN 运行库放在同一目录。
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$HERE:/oem/usr/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [ -f "$HERE/librknnmrt.so" ]; then
    # Select the adjacent runtime before vendor dependencies load another copy.
    # 在厂商依赖载入其他版本之前，优先选择同目录的运行库。
    export LD_PRELOAD="$HERE/librknnmrt.so${LD_PRELOAD:+:$LD_PRELOAD}"
fi
exec python3 "$@"
