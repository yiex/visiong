<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->
# API Naming Policy

VisionG now treats the Python binding names as the canonical public API.

## Rule

- Python `snake_case` names are the source of truth for public API naming.
- The C++ layer keeps existing legacy names for backward compatibility.
- Matching C++ `snake_case` wrappers are added whenever the public Python name
  differs from the historical implementation name.

## Why this policy exists

The Python API is the most visible surface of the project. Earlier revisions had
several cases where the Python name and the underlying C++ symbol diverged, which
made the codebase harder to navigate and harder to document.

Using one naming convention across the public surface improves:

- discoverability,
- documentation quality,
- contributor onboarding,
- and long-term maintenance.

## Examples covered by this candidate

- `Camera.skip()` wraps legacy `skip_frames()`.
- `NPU.infer()` wraps legacy `inference()`.
- `Touch.read()` wraps legacy `get_touch_points()`.
- `NpuClock.set_rate_hz()` wraps legacy `set_rate()`.
- `GUI` methods now have matching C++ snake_case wrappers for the Python names.

## Compatibility stance

Legacy C++ method names are still present in this candidate to avoid breaking
existing code. New documentation should prefer the snake_case forms.

