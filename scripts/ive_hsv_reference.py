# SPDX-License-Identifier: LGPL-3.0-or-later
"""Integer RGB-to-HSV reference for the RV1106 IVE runtime.
RV1106 IVE 的 RGB 转 HSV 整数参考实现。

The reciprocal precision was inferred from controlled hardware output. It is
an observable compatibility model, not a description of the internal circuit.
倒数精度来自受控硬件输出拟合，此兼容模型不代表内部电路实现。
"""

import numpy as np


def rgb_to_ive_hsv(rgb):
    """Convert uint8 RGB pixels to the IVE's full-range HSV888 representation.

    The last dimension must contain R, G, B. H wraps modulo 256; S and V are
    unsigned bytes. This function runs in software and does not call IVE.
    最后一维为 R/G/B；H 按 256 回绕，S/V 为无符号字节。本函数在 CPU 上运行。
    """
    pixels = np.asarray(rgb)
    if pixels.dtype != np.uint8 or pixels.ndim < 1 or pixels.shape[-1] != 3:
        raise ValueError("Expected an RGB uint8 array with a final dimension of 3.")
    red = pixels[..., 0].astype(np.int32)
    green = pixels[..., 1].astype(np.int32)
    blue = pixels[..., 2].astype(np.int32)
    value = np.maximum(np.maximum(red, green), blue)
    minimum = np.minimum(np.minimum(red, green), blue)
    delta = value - minimum
    value_divisor = np.maximum(value, 1)
    delta_divisor = np.maximum(6 * delta, 1)
    saturation_reciprocal = ((255 << 8) + value_divisor // 2) // value_divisor
    hue_reciprocal = ((256 << 8) + delta_divisor // 2) // delta_divisor
    numerator = np.where(
        value == red,
        green - blue,
        np.where(value == green, blue - red + 2 * delta, red - green + 4 * delta),
    )
    saturation = (delta * saturation_reciprocal + 128) >> 8
    hue = ((numerator * hue_reciprocal + 128) >> 8) & 255
    return np.stack((hue, saturation, value), axis=-1).astype(np.uint8)


def rgb_pixel_to_ive_hsv(red, green, blue):
    """Scalar equivalent with signed hue arithmetic and wrap.
    单像素整数版本，包含带符号色相运算与回绕。
    """
    if any(
        type(channel) is not int or not 0 <= channel <= 255
        for channel in (red, green, blue)
    ):
        raise ValueError("RGB channels must be integers in [0,255].")
    value = max(red, green, blue)
    delta = value - min(red, green, blue)
    if delta == 0:
        return 0, 0, value
    numerator = (
        green - blue
        if value == red
        else blue - red + 2 * delta if value == green else red - green + 4 * delta
    )
    saturation_reciprocal = ((255 << 8) + value // 2) // value
    hue_reciprocal = ((256 << 8) + 3 * delta) // (6 * delta)
    return (
        ((numerator * hue_reciprocal + 128) >> 8) & 255,
        (delta * saturation_reciprocal + 128) >> 8,
        value,
    )
