"""Decode discrete LocateAnything reference and coordinate tokens."""

import re


REFERENCE_PATTERN = re.compile(r"<ref>(.*?)</ref>((?:\s*<box>(?:<\d+>)+</box>)+)", re.DOTALL)
BOX_PATTERN = re.compile(r"<box>((?:<\d+>)+)</box>")
COORDINATE_PATTERN = re.compile(r"<(\d+)>")


def absolute_box(values, width, height, coordinate_max):
    """Scale, sort, and clamp one coordinate-token box."""
    x1, y1, x2, y2 = values
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    scaled = [
        x1 / coordinate_max * width,
        y1 / coordinate_max * height,
        x2 / coordinate_max * width,
        y2 / coordinate_max * height,
    ]
    return [
        min(width, max(0.0, scaled[0])),
        min(height, max(0.0, scaled[1])),
        min(width, max(0.0, scaled[2])),
        min(height, max(0.0, scaled[3])),
    ]


def absolute_point(values, width, height, coordinate_max):
    """Scale and clamp one coordinate-token point."""
    x, y = values
    return [
        min(width, max(0.0, x / coordinate_max * width)),
        min(height, max(0.0, y / coordinate_max * height)),
    ]


def decode(raw_response, width, height, coordinate_max=1000, target_size=4):
    """Return complete reference groups as absolute spatial targets."""
    if target_size not in (2, 4):
        raise ValueError("target_size must be 2 or 4")
    predictions = {}
    for reference_match in REFERENCE_PATTERN.finditer(raw_response):
        reference = reference_match.group(1).strip()
        if not reference:
            continue
        for payload in BOX_PATTERN.findall(reference_match.group(2)):
            values = [int(value) for value in COORDINATE_PATTERN.findall(payload)]
            if len(values) != target_size or not all(
                0 <= value <= coordinate_max for value in values
            ):
                continue
            if target_size == 2:
                target = absolute_point(values, width, height, coordinate_max)
            else:
                target = absolute_box(values, width, height, coordinate_max)
            predictions.setdefault(reference, []).append(target)
    return predictions
