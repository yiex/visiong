#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWLEVEL_CPP = REPO_ROOT / "src" / "npu" / "LowLevelNPU.cpp"


def extract_block(text: str, signature: str, next_signature: str) -> str:
    start = text.find(signature)
    if start < 0:
        raise ValueError(f"missing signature: {signature}")
    end = text.find(next_signature, start + len(signature))
    if end < 0:
        raise ValueError(f"missing next signature after: {signature}")
    return text[start:end]


def require_snippets(block: str, function_name: str, snippets) -> None:
    for snippet in snippets:
        if snippet not in block:
            raise ValueError(f"missing '{snippet}' in {function_name}")


def main() -> int:
    text = LOWLEVEL_CPP.read_text(encoding="utf-8")

    if "bool build_strided_input_copy_plan(" not in text:
        print(f"missing build_strided_input_copy_plan helper in {LOWLEVEL_CPP}", file=sys.stderr)
        return 1

    try:
        set_input_buffer_block = extract_block(
            text,
            "void LowLevelNPU::set_input_buffer(",
            "void LowLevelNPU::set_input_from_float(",
        )
        set_input_from_float_block = extract_block(
            text,
            "void LowLevelNPU::set_input_from_float(",
            "SourceDmaContext prepare_source_dma(",
        )
    except ValueError as exc:
        print(f"{LOWLEVEL_CPP}: {exc}", file=sys.stderr)
        return 1

    try:
        require_snippets(
            set_input_buffer_block,
            "set_input_buffer",
            [
                "build_strided_input_copy_plan(",
                "std::vector<uint8_t> packed(",
                "copy_data_with_stride(mem->virt_addr,",
            ],
        )
        require_snippets(
            set_input_from_float_block,
            "set_input_from_float",
            [
                "build_strided_input_copy_plan(",
                "packed.assign(packed_bytes, 0);",
                "copy_data_with_stride(mem->virt_addr,",
            ],
        )
    except ValueError as exc:
        print(f"{LOWLEVEL_CPP}: {exc}", file=sys.stderr)
        return 1

    print("LowLevelNPU stride guard looks correct.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
