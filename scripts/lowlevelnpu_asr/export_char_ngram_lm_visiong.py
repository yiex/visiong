#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import annotations

import argparse
import gzip
import pickle
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a pickle.gz CharNgramLm payload into the plain-text format used by visiong.ASR."
    )
    parser.add_argument("--input-pkl-gz", type=Path, required=True)
    parser.add_argument("--output-txt", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with gzip.open(args.input_pkl_gz.resolve(), "rb") as handle:
        payload = pickle.load(handle)

    if payload.get("lm_type") != "char_ngram_stupid_backoff":
        raise ValueError(f"unsupported lm_type: {payload.get('lm_type')!r}")

    order = int(payload["order"])
    backoff_alpha = float(payload["backoff_alpha"])
    unknown_log_prob = float(payload["unknown_log_prob"])
    unigram_log_probs = dict(payload["unigram_log_probs"])
    higher_order_log_probs = dict(payload["higher_order_log_probs"])

    output_path = args.output_txt.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("VISIONG_CHAR_NGRAM_LM_V1\n")
        handle.write(f"O\t{order}\n")
        handle.write(f"A\t{backoff_alpha:.9g}\n")
        handle.write(f"K\t{unknown_log_prob:.9g}\n")

        for token, score in sorted(unigram_log_probs.items(), key=lambda item: item[0]):
            handle.write(f"U\t{token}\t{float(score):.9g}\n")

        for order_key, values in sorted(higher_order_log_probs.items(), key=lambda item: int(item[0])):
            ngram_order = int(order_key)
            for joined_key, score in sorted(values.items(), key=lambda item: item[0]):
                fields = joined_key.split("\t")
                if len(fields) != ngram_order:
                    raise ValueError(
                        f"ngram key {joined_key!r} has {len(fields)} tokens, expected {ngram_order}"
                    )
                handle.write("N\t")
                handle.write(str(ngram_order))
                for token in fields:
                    handle.write("\t")
                    handle.write(token)
                handle.write(f"\t{float(score):.9g}\n")

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
