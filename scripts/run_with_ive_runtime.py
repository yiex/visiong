# SPDX-License-Identifier: LGPL-3.0-or-later
"""Explicitly load a verified private IVE runtime before a Python application.
在 Python 应用启动前显式加载已校验的独立 IVE 运行库。
"""

import argparse
import ctypes
import hashlib
import os
from pathlib import Path
import runpy
import struct
import sys


CORRECTED_SHA256 = "80c5c104f5bf02b88dafd1ed0867d90f2439fded2464174080d30ff97bd03409"
ORIGINAL_SHA256 = "2446c5b1720c083b89338c33cdf3f289c8fc94b29386a0404481881a06cc3455"
_HANDLES = []


def validate_runtime(path):
    """Verify the complete measured copy and its executable instruction mapping.
    校验已实测副本的完整哈希及可执行指令映射。
    """
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError("runtime must be a regular file")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != CORRECTED_SHA256:
        raise ValueError("Runtime SHA256 does not match the measured corrected copy")
    if len(data) < 52 or data[:7] != b"\x7fELF\x01\x01\x01":
        raise ValueError("Runtime must be ELF32 little-endian")
    fields = struct.unpack_from("<HHIIIIIHHHHHH", data, 16)
    file_type, machine, version, _, phoff, _, _, ehsize, phsize, phcount, _, _, _ = (
        fields
    )
    if (
        (file_type, machine, version, ehsize) != (3, 40, 1, 52)
        or phsize != 32
        or not phcount
    ):
        raise ValueError("Runtime must be a supported ARM ELF shared library")
    if phoff + phsize * phcount > len(data):
        raise ValueError("ELF program headers exceed the file")
    mappings = []
    for index in range(phcount):
        kind, offset, address, _, file_size, memory_size, flags, _ = struct.unpack_from(
            "<IIIIIIII", data, phoff + index * phsize
        )
        if (
            kind == 1
            and flags & 1
            and address <= 0x1D650
            and 0x1D654 <= address + file_size
        ):
            if file_size > memory_size or offset + file_size > len(data):
                raise ValueError("Invalid executable ELF segment")
            mappings.append(offset + 0x1D650 - address)
    if mappings != [0x1D650] or data[0x1D650:0x1D654] != bytes.fromhex("0630a0e3"):
        raise ValueError("Corrected instruction or ELF mapping differs")
    return path


def require_unloaded():
    """Reject previously imported bindings or a runtime already mapped in this process.
    拒绝已导入绑定或本进程已经映射运行库的情况。
    """
    if "visiong" in sys.modules or "_visiong" in sys.modules:
        raise RuntimeError("Start a new process before importing VisionG")
    try:
        lines = Path("/proc/self/maps").read_text().splitlines()
    except OSError as error:
        raise RuntimeError("Cannot inspect loaded runtimes in this process") from error
    paths = set()
    for line in lines:
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        text = fields[5]
        if "librve" in Path(text).name.lower():
            raise RuntimeError("An IVE runtime is already mapped; start a new process")
        paths.add(text)
    for text in paths:
        path = Path(text)
        if path.suffix != ".so" or not path.is_file():
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() in (
            ORIGINAL_SHA256,
            CORRECTED_SHA256,
        ):
            raise RuntimeError(
                "A renamed IVE runtime is already mapped; start a new process"
            )
    no_load = getattr(os, "RTLD_NOLOAD", None)
    if no_load is None:
        raise RuntimeError("This platform cannot check the IVE loader state")
    try:
        existing = ctypes.CDLL("librve.so", mode=no_load | getattr(os, "RTLD_NOW", 2))
    except OSError:
        return
    _HANDLES.append(existing)
    raise RuntimeError("An IVE runtime is already loaded; start a new process")


def load_runtime(path):
    """Load Rockit globally before the explicitly selected corrected IVE copy.
    先全局加载 Rockit，再加载显式指定的 IVE 修正副本。
    """
    path = validate_runtime(path)
    require_unloaded()
    _HANDLES.append(ctypes.CDLL("librockit.so", mode=ctypes.RTLD_GLOBAL))
    require_unloaded()
    path = validate_runtime(path)
    _HANDLES.append(ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL))
    mapped = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/"):
            mapped.add(fields[5])
    if (
        str(path) not in mapped
        or hashlib.sha256(path.read_bytes()).hexdigest() != CORRECTED_SHA256
    ):
        raise RuntimeError("The selected IVE runtime mapping could not be verified")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the file without loading any runtime",
    )
    parser.add_argument(
        "application",
        nargs=argparse.REMAINDER,
        help="Python script and its arguments, optionally after --",
    )
    args = parser.parse_args()
    try:
        runtime = validate_runtime(args.runtime)
        if args.check:
            if args.application:
                parser.error("--check does not run an application")
            print("Verified corrected IVE runtime: " + CORRECTED_SHA256)
            return
        application = args.application
        if application and application[0] == "--":
            application = application[1:]
        if not application:
            parser.error("A Python application script is required")
        script = Path(application[0]).resolve(strict=True)
        if not script.is_file():
            parser.error("The application must be a Python script file")
        load_runtime(runtime)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, str(error) + "\n")
    if not hasattr(sys, "orig_argv"):
        # Preserve this explicit loader if the binding performs its normal restart.
        # 若绑定按原有流程重启，仍保留显式加载入口。
        sys.orig_argv = [sys.executable] + list(sys.argv)
    sys.argv = [str(script)] + application[1:]
    sys.path.insert(0, str(script.parent))
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
