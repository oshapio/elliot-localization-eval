"""Shared localization prompts and parsers; no model loading or runtime patches."""

import re
import tasks as T


def _task_for(entry, family):
    """Which task this sample is, for THIS model. Only rex was trained to point at GUI
    elements; asking qwen/elliot for a point gets worse results than asking for a box, and
    the sspro scorer accepts either (it tests whether a predicted box's CENTER lands in
    the GT box), so they answer the box form of the same question."""
    protocol = entry.get("benchmark_protocol")
    task = (T.TaskType.POINTING if protocol == "point" else
            T.TaskType.GUI_POINTING if protocol == "gui" else T.task_for_entry(entry))
    if task is T.TaskType.GUI_POINTING and family != "rex-omni":
        task = T.TaskType.GUI_DETECTION
    return task


def _question(family):
    """Expose the central task registry through the legacy adapter interface."""
    def q(entry):
        return T.build_prompt_for_entry(entry, family)
    return q


def extract_numbers(text):
    # validated parser: first 4 numbers, preferring bracketed groups
    cleaned = text.replace("−", "-").replace("，", ",")
    regions = re.findall(r"(?:<\|box_start\|>|<box>|bbox\s*[:=]?\s*)?\s*[\[\(]\s*([^\]\)]+)[\]\)]", cleaned, re.I)
    for candidate in regions + [cleaned]:
        vals = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", candidate)
        if len(vals) >= 4:
            return [float(x) for x in vals[:4]]
    return None


def normalize_box(vals, scale, w, h):
    """Model coordinates -> absolute pixels on the ORIGINAL image, which is what the
    scorers expect. `scale` is the model's coordinate range from tasks.COORD (1000 for
    qwen's ints, 1.0 for elliot's decimals). Corners are sorted because models sometimes
    emit x2 < x1, and clamped because they sometimes run off the edge -- an unsorted or
    negative box silently scores IoU 0 instead of matching."""
    sx, sy = scale if isinstance(scale, (tuple, list)) else (scale, scale)
    x1, y1, x2, y2 = vals
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return [min(w, max(0.0, x1 / sx * w)), min(h, max(0.0, y1 / sy * h)),
            min(w, max(0.0, x2 / sx * w)), min(h, max(0.0, y2 / sy * h))]


_NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
_BOX = re.compile(r"[\[\(]\s*(" + _NUM + r")\s*,\s*(" + _NUM + r")\s*,\s*(" + _NUM + r")\s*,\s*("
                  + _NUM + r")\s*[\]\)]")
_JSON_LABEL = re.compile(r'"(?:label|text_content)"\s*:\s*"([^"]*)"')
_LINE_LABEL = re.compile(r'([^\n\[\]{}:,]{1,60}?)\s*:\s*$')


def extract_labeled_boxes(text):
    """(label|None, [x1,y1,x2,y2]) per 4-number group, in order. Label comes from the
    "label": "x" that follows inside the same JSON object (qwen's trained format), else
    from a `name:` prefix immediately before the bracket."""
    cleaned = text.replace("−", "-").replace("，", ",")
    out = []
    for m in _BOX.finditer(cleaned):
        vals = [float(g) for g in m.groups()]
        label, tail = None, cleaned[m.end():m.end() + 80]
        jm = _JSON_LABEL.search(tail)
        if jm and "{" not in tail[:jm.start()]:            # same object, not the next one
            label = jm.group(1)
        else:
            lm = _LINE_LABEL.search(cleaned[max(0, m.start() - 70):m.start()])
            if lm:
                label = lm.group(1).strip().strip('"\'-*• ')
        out.append((label, vals))
    return out


def extract_spatial_boxes(text):
    """Extract complete reference groups from spatial-token output."""
    groups = []
    pattern = re.compile(r"<ref>(.*?)</ref>((?:<box>(?:<\d+>)+</box>)*)", re.DOTALL)
    for match in pattern.finditer(text):
        boxes = []
        for payload in re.findall(r"<box>((?:<\d+>)+)</box>", match.group(2)):
            values = [float(value) for value in re.findall(r"<(\d+)>", payload)]
            if len(values) == 4 and all(0 <= value <= 1000 for value in values):
                boxes.append(values)
        if boxes:
            groups.append((match.group(1), boxes))
    return groups


def extract_points(text, scale, w, h):
    sx, sy = scale if isinstance(scale, (tuple, list)) else (scale, scale)
    # every bracketed 2-number group -> one [x, y] point, scaled to absolute pixels
    cleaned = text.replace("−", "-").replace("，", ",")
    pts = []
    for grp in re.findall(r"[\[\(]([^\[\]\(\)]+)[\]\)]", cleaned):
        nums = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", grp)
        if len(nums) == 2:
            x, y = float(nums[0]), float(nums[1])
            pts.append([min(w, max(0.0, x / sx * w)), min(h, max(0.0, y / sy * h))])
    return pts


# tasks whose answer is ONE box: keep the validated single-box recipe (extra boxes would
# only cost precision -- referring/GUI ask for a single region)
_SINGLE_BOX = (T.TaskType.REFERRING, T.TaskType.GUI_DETECTION)


def _parse(raw, entry, family, w, h, coordinate_size=None):
    """Raw model text -> {label: [box|point, ...]} in absolute pixels, the schema the
    scorers read. Boxes the model left unlabeled fall back to the sample's first category
    (elliot never labels; qwen labels via its JSON format)."""
    task = _task_for(entry, family)
    label = entry["categories"][0] if isinstance(entry["categories"], list) else str(entry["categories"])
    scale = coordinate_size if coordinate_size is not None else T.COORD[family]["scale"]
    if family == "elliot-loc":
        preds = {}
        for reference, boxes in extract_spatial_boxes(raw):
            preds.setdefault(reference, []).extend(
                normalize_box(box, scale, w, h) for box in boxes)
        return preds
    if T.TASKS[task]["output"] == "points":
        pts = extract_points(raw, scale, w, h)
        return {label: pts} if pts else {}
    if task in _SINGLE_BOX and not (entry["dataset_name"] == "HumanRef"
                                        and entry.get("benchmark_protocol") == "box"):
        vals = extract_numbers(raw)
        return {label: [normalize_box(vals, scale, w, h)]} if vals else {}
    preds = {}                                   # detection/OCR: every instance the model listed
    for lab, vals in extract_labeled_boxes(raw):
        preds.setdefault(lab or label, []).append(normalize_box(vals, scale, w, h))
    return preds

