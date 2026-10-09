"""Decode the trained grouped-JSON special-coordinate response format."""

import json
import re

from decoders.discrete_ntp import absolute_box, absolute_point


FORMAT = "mixed_special_xyxy_grouped"


def _parse_grouped_lines(text):
    """Parse the trained plain variant: ``label:[[x1,y1,x2,y2], ...]``."""
    grouped = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        separator = line.rfind(":[")
        if separator < 0:
            return None
        label, payload = line[:separator], line[separator + 1 :]
        if not label.strip():
            return None
        try:
            targets = json.loads(payload)
        except (ValueError, TypeError):
            return None
        if not isinstance(targets, list):
            return None
        grouped[label.strip()] = targets
    return grouped or None


def decode(raw, width, height, coordinate_max=1000, target_size=4):
    text = raw.strip()
    while text.startswith("<think>"):
        if "</think>" not in text:
            return {}
        text = text.split("</think>", 1)[1].lstrip()
    for ending in ("<|im_end|>", "<|endoftext|>"):
        text = text.removesuffix(ending).rstrip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    # Convert spatial tokens only outside JSON string labels.
    text = re.sub(
        r'"(?:\\.|[^"\\])*"|<(\d+)>',
        lambda match: match.group(1) if match.group(1) is not None else match.group(0),
        text,
    )
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        # A malformed JSON-looking response must not be reinterpreted as the
        # legacy ``label:[[...]]`` text variant.
        if text.startswith(("{", "[")):
            return {}
        data = _parse_grouped_lines(text)
    if not isinstance(data, dict):
        return {}
    predictions = {}
    for label, targets in data.items():
        if not isinstance(label, str) or not label.strip() or not isinstance(targets, list):
            continue
        for values in targets:
            if not isinstance(values, list) or len(values) != target_size:
                continue
            if not all(
                type(value) in (int, float) and 0 <= value <= coordinate_max
                for value in values
            ):
                continue
            converter = absolute_point if target_size == 2 else absolute_box
            predictions.setdefault(label.strip(), []).append(
                converter(values, width, height, coordinate_max)
            )
    return predictions
