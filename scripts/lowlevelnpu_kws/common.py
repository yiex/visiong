#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import math
import random
import shutil
import wave
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None


KEYWORDS = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
]
UNKNOWN_LABEL = "unknown"
SILENCE_LABEL = "silence"
ALL_LABELS = KEYWORDS + [UNKNOWN_LABEL, SILENCE_LABEL]

SAMPLE_RATE = 16000
CLIP_SAMPLES = 16000
WINDOW_SIZE_MS = 30
WINDOW_STRIDE_MS = 20
FRAME_LENGTH = SAMPLE_RATE * WINDOW_SIZE_MS // 1000
FRAME_STEP = SAMPLE_RATE * WINDOW_STRIDE_MS // 1000
FFT_SIZE = 512
NUM_MEL_BINS = 40
LOWER_EDGE_HERTZ = 20.0
UPPER_EDGE_HERTZ = 4000.0
NUM_FRAMES = 1 + max(0, (CLIP_SAMPLES - FRAME_LENGTH) // FRAME_STEP)
EPSILON = 1e-6
TIME_SHIFT_MS = 100
TIME_SHIFT_SAMPLES = SAMPLE_RATE * TIME_SHIFT_MS // 1000


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def reset_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_lines(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def softmax(logits: np.ndarray) -> np.ndarray:
    values = logits.astype(np.float64)
    values = values - np.max(values, axis=-1, keepdims=True)
    exp_values = np.exp(values)
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)


def label_to_index(label: str) -> int:
    return ALL_LABELS.index(label)


def index_to_label(index: int) -> str:
    return ALL_LABELS[index]


def read_wav_pcm16(path: Path, expected_sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    with wave.open(str(path), "rb") as wav_file:
        num_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()
        pcm = wav_file.readframes(num_frames)

    if sample_width != 2:
        raise ValueError("Expected 16-bit PCM WAV, got sample_width={} for {}".format(sample_width, path))
    if sample_rate != expected_sample_rate:
        raise ValueError("Expected sample_rate={}, got {} for {}".format(expected_sample_rate, sample_rate, path))

    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    if num_channels > 1:
        audio = audio.reshape(-1, num_channels).mean(axis=1)
    return audio / 32768.0


def write_wav_pcm16(path: Path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def pad_or_trim(audio: np.ndarray, length: int = CLIP_SAMPLES) -> np.ndarray:
    audio = audio.astype(np.float32).reshape(-1)
    if audio.shape[0] == length:
        return audio
    if audio.shape[0] > length:
        return audio[:length].astype(np.float32, copy=False)
    out = np.zeros(length, dtype=np.float32)
    out[: audio.shape[0]] = audio
    return out


def apply_time_shift(audio: np.ndarray, shift_samples: int) -> np.ndarray:
    if shift_samples == 0:
        return audio.astype(np.float32, copy=True)
    shifted = np.zeros_like(audio, dtype=np.float32)
    if shift_samples > 0:
        shifted[shift_samples:] = audio[:-shift_samples]
    else:
        shifted[:shift_samples] = audio[-shift_samples:]
    return shifted


def slice_noise(noise_audio: np.ndarray, start: int, length: int = CLIP_SAMPLES) -> np.ndarray:
    if noise_audio.shape[0] < length:
        repeats = int(math.ceil(float(length) / float(max(noise_audio.shape[0], 1))))
        tiled = np.tile(noise_audio, repeats)
        return tiled[:length].astype(np.float32)
    max_start = max(0, noise_audio.shape[0] - length)
    clipped_start = max(0, min(start, max_start))
    return noise_audio[clipped_start : clipped_start + length].astype(np.float32, copy=False)


def mix_background(audio: np.ndarray, background: np.ndarray, volume: float) -> np.ndarray:
    mixed = audio.astype(np.float32) + background.astype(np.float32) * float(volume)
    return np.clip(mixed, -1.0, 1.0)


def hz_to_mel(freq_hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + freq_hz / 700.0)


def mel_to_hz(mels: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)


def build_mel_filterbank(
    sample_rate: int = SAMPLE_RATE,
    fft_size: int = FFT_SIZE,
    num_mel_bins: int = NUM_MEL_BINS,
    lower_edge_hertz: float = LOWER_EDGE_HERTZ,
    upper_edge_hertz: float = UPPER_EDGE_HERTZ,
) -> np.ndarray:
    num_spectrogram_bins = fft_size // 2 + 1
    mel_edges = np.linspace(
        hz_to_mel(np.array([lower_edge_hertz], dtype=np.float64))[0],
        hz_to_mel(np.array([upper_edge_hertz], dtype=np.float64))[0],
        num_mel_bins + 2,
        dtype=np.float64,
    )
    hz_edges = mel_to_hz(mel_edges)
    fft_bins = np.floor((fft_size + 1) * hz_edges / sample_rate).astype(np.int64)

    filters = np.zeros((num_mel_bins, num_spectrogram_bins), dtype=np.float32)
    for mel_index in range(1, num_mel_bins + 1):
        left = int(fft_bins[mel_index - 1])
        center = int(fft_bins[mel_index])
        right = int(fft_bins[mel_index + 1])
        center = max(center, left + 1)
        right = max(right, center + 1)

        for bin_index in range(left, min(center, num_spectrogram_bins)):
            filters[mel_index - 1, bin_index] = float(bin_index - left) / float(center - left)
        for bin_index in range(center, min(right, num_spectrogram_bins)):
            filters[mel_index - 1, bin_index] = float(right - bin_index) / float(right - center)
    return filters


MEL_FILTERBANK = build_mel_filterbank()
WINDOW_FN = np.hanning(FRAME_LENGTH).astype(np.float32)


def compute_log_mel_features(audio: np.ndarray) -> np.ndarray:
    signal = pad_or_trim(audio, CLIP_SAMPLES)
    frame_starts = np.arange(NUM_FRAMES, dtype=np.int64)[:, None] * FRAME_STEP
    frame_offsets = np.arange(FRAME_LENGTH, dtype=np.int64)[None, :]
    frames = signal[frame_starts + frame_offsets]
    frames = frames * WINDOW_FN[None, :]
    spectrum = np.fft.rfft(frames, n=FFT_SIZE, axis=1)
    power = (np.abs(spectrum) ** 2).astype(np.float32) / float(FFT_SIZE)
    mel = np.matmul(power, MEL_FILTERBANK.T)
    log_mel = np.log(mel + EPSILON).astype(np.float32)
    mean = float(np.mean(log_mel))
    std = float(np.std(log_mel))
    if std < 1e-5:
        std = 1.0
    normalized = (log_mel - mean) / std
    return normalized.astype(np.float32)


def feature_tensor_from_audio(audio: np.ndarray) -> np.ndarray:
    features = compute_log_mel_features(audio)
    return features[np.newaxis, np.newaxis, :, :].astype(np.float32)


def load_feature_tensor_from_wav(path: Path) -> np.ndarray:
    return feature_tensor_from_audio(read_wav_pcm16(path))


def create_augmented_audio(
    audio: np.ndarray,
    rng: np.random.Generator,
    background_noises: Sequence[np.ndarray],
    allow_noise: bool,
    allow_shift: bool,
) -> np.ndarray:
    out = pad_or_trim(audio, CLIP_SAMPLES)
    if allow_shift:
        shift = int(rng.integers(-TIME_SHIFT_SAMPLES, TIME_SHIFT_SAMPLES + 1))
        out = apply_time_shift(out, shift)
    if allow_noise and background_noises and float(rng.random()) < 0.8:
        noise = background_noises[int(rng.integers(0, len(background_noises)))]
        max_start = max(0, noise.shape[0] - CLIP_SAMPLES)
        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        noise_clip = slice_noise(noise, start)
        volume = float(rng.uniform(0.0, 0.12))
        out = mix_background(out, noise_clip, volume)
    return out.astype(np.float32)


def create_silence_audio(
    rng: np.random.Generator,
    background_noises: Sequence[np.ndarray],
    deterministic_index: int = 0,
) -> np.ndarray:
    if background_noises:
        noise = background_noises[deterministic_index % len(background_noises)]
        max_start = max(0, noise.shape[0] - CLIP_SAMPLES)
        start = 0
        if max_start > 0:
            start = int(rng.integers(0, max_start + 1))
        clip = slice_noise(noise, start)
        volume = float(rng.uniform(0.05, 0.7))
        return np.clip(clip * volume, -1.0, 1.0).astype(np.float32)
    return np.zeros(CLIP_SAMPLES, dtype=np.float32)


if nn is not None:
    class DepthwiseSeparableBlock(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.depthwise = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
            self.pointwise = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            x = self.depthwise(x)
            return self.pointwise(x)


    class KeywordSpottingDSCNN(nn.Module):
        def __init__(self, num_classes: int = len(ALL_LABELS), channels: int = 48, dropout: float = 0.2) -> None:
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(1, channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
            self.blocks = nn.Sequential(
                DepthwiseSeparableBlock(channels),
                DepthwiseSeparableBlock(channels),
                DepthwiseSeparableBlock(channels),
                DepthwiseSeparableBlock(channels),
            )
            self.head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Linear(channels, num_classes)

        def forward(self, x):
            x = self.stem(x)
            x = self.blocks(x)
            x = self.head(x)
            x = torch.flatten(x, 1)
            return self.classifier(x)
else:
    class KeywordSpottingDSCNN(object):
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("KeywordSpottingDSCNN requires PyTorch on the training host.")
