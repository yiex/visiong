# SPDX-License-Identifier: LGPL-3.0-or-later
"""Create a separate LK stride correction for one verified IVE runtime build.
为一个已验证的 IVE 运行库版本生成独立的 LK 步长修正副本。

The supplied vendor runtime retains its original copyright and license.
输入的厂商运行库保留原版权及许可。
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct


SOURCE_SHA256 = "2446c5b1720c083b89338c33cdf3f289c8fc94b29386a0404481881a06cc3455"
CORRECTED_SHA256 = "80c5c104f5bf02b88dafd1ed0867d90f2439fded2464174080d30ff97bd03409"
PATCH_ADDRESS = 0x1D650
PATCH_FILE_OFFSET = 120400
ORIGINAL_INSTRUCTION = bytes.fromhex("033083e2")
CORRECTED_INSTRUCTION = bytes.fromhex("0630a0e3")
READ_CHUNK_BYTES = 1024 * 1024


def elf_patch_offset(data):
    """Locate the complete ARM instruction in one executable file-backed LOAD.
    在唯一的可执行且有文件内容的 LOAD 段中定位完整 ARM 指令。
    """
    if len(data) < 52 or data[:7] != b"\x7fELF\x01\x01\x01":
        raise ValueError("Expected an ELF32 little-endian version-1 image")
    header = struct.unpack_from("<16sHHIIIIIHHHHHH", data)
    if header[1:4] != (3, 40, 1):
        raise ValueError("Expected an ARM version-1 shared object")
    phoff, ehsize, phentsize, phnum = header[5], header[8], header[9], header[10]
    if ehsize != 52 or phentsize != 32 or not phnum or phnum == 0xFFFF:
        raise ValueError("Unsupported ELF header or program-header layout")
    if phoff < ehsize or phoff + phentsize * phnum > len(data):
        raise ValueError("Program-header table is outside the input file")
    mappings = []
    for index in range(phnum):
        kind, offset, address, _, filesz, memsz, flags, alignment = struct.unpack_from(
            "<8I", data, phoff + index * phentsize
        )
        if kind != 1:
            continue
        if filesz > memsz or offset + filesz > len(data) or address + memsz > 1 << 32:
            raise ValueError("Invalid LOAD segment bounds")
        if alignment not in (0, 1) and (
            alignment & (alignment - 1) or (address - offset) % alignment
        ):
            raise ValueError("Invalid LOAD segment alignment")
        if address <= PATCH_ADDRESS and PATCH_ADDRESS + 4 <= address + filesz:
            if not flags & 1:
                raise ValueError("Patch address is not in an executable LOAD segment")
            mappings.append(offset + PATCH_ADDRESS - address)
    if len(mappings) != 1:
        raise ValueError(
            "Patch address must have exactly one complete file-backed LOAD mapping"
        )
    if mappings[0] != PATCH_FILE_OFFSET:
        raise ValueError("Unexpected file offset for the patch address")
    return mappings[0]


def prepare_copy(data):
    """Require the measured source and corrected hashes around a four-byte edit.
    四字节修改前后均要求与实测版本的哈希完全一致。
    """
    source_hash = hashlib.sha256(data).hexdigest()
    if source_hash == CORRECTED_SHA256:
        raise ValueError("Input is already the verified corrected runtime")
    offset = elf_patch_offset(data)
    if source_hash != SOURCE_SHA256:
        raise ValueError("Unsupported input SHA256: " + source_hash)
    if data[offset : offset + 4] != ORIGINAL_INSTRUCTION:
        raise ValueError("Original ARM instruction does not match")
    corrected = data[:offset] + CORRECTED_INSTRUCTION + data[offset + 4 :]
    corrected_hash = hashlib.sha256(corrected).hexdigest()
    if corrected_hash != CORRECTED_SHA256:
        raise ValueError("Corrected SHA256 does not match the measured result")
    return corrected, dict(
        input_sha256=source_hash,
        output_sha256=corrected_hash,
        bytes=len(data),
        virtual_address=hex(PATCH_ADDRESS),
        file_offset=offset,
        original_bytes=ORIGINAL_INSTRUCTION.hex(),
        corrected_bytes=CORRECTED_INSTRUCTION.hex(),
    )


def write_new_copy(source, output, corrected):
    """Write a new file exclusively and remove it if synchronization or readback fails.
    以独占方式写入新文件；同步或读回失败时删除该输出。
    """
    if source.resolve() == output.resolve():
        raise ValueError("Output must differ from the source file")
    if output.exists() or output.is_symlink():
        raise ValueError("Output already exists; choose a new filename")
    if not output.parent.is_dir():
        raise ValueError("Output directory does not exist")
    created = False
    try:
        with output.open("xb") as stream:
            created = True
            if stream.write(corrected) != len(corrected):
                raise OSError("Incomplete corrected-runtime write")
            stream.flush()
            os.fsync(stream.fileno())
        digest = hashlib.sha256()
        with output.open("rb") as stream:
            while True:
                chunk = stream.read(READ_CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
        if digest.hexdigest() != CORRECTED_SHA256:
            raise ValueError("Corrected-runtime readback SHA256 mismatch")
    except BaseException:
        if created:
            output.unlink()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Vendor librve.so / 厂商 librve.so")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--check", action="store_true", help="Validate without writing / 只检查，不写入"
    )
    action.add_argument("--output", type=Path, help="New corrected copy / 新的修正副本")
    args = parser.parse_args()
    try:
        if not args.input.is_file():
            raise ValueError("Input must be an existing regular file")
        corrected, report = prepare_copy(args.input.read_bytes())
        if args.check:
            report["status"] = "patchable"
        else:
            write_new_copy(args.input, args.output, corrected)
            report["status"] = "written"
        print(json.dumps(report), flush=True)
    except (OSError, ValueError, struct.error) as error:
        parser.exit(2, "error: %s\n" % error)


if __name__ == "__main__":
    main()
