# Evaluator prompt cardinality proof

**PASS.** The actual benchmark-850 prompt entry point rendered identical text after its hidden answer was replaced by 0, 1, and 20 targets.

- Manifest rows: 850
- Counterfactual renders: 2550
- Prompt differences: **0**
- Rows missing `ALL matching instances`: **0**
- Forbidden legacy singleton wording: **0**
- Manifest SHA256: `76d50a246b1e463af3ed258aa723d2ed5a249bafb10b3d402bf52b33df519bc1`
- Prompt source SHA256: `ff22c747479ed47e16626d829147d749b0e5d1dc778f80227f2fa1e63725d2b2`

The evaluator receives only task metadata when constructing prompts; answer annotations are not in its allowed prompt-input field set.

## Coverage

| Family | Protocol | Rows |
|---|---:|---:|
| common_detection | box | 100 |
| dense_detection | box | 100 |
| document_layout | box | 100 |
| gui_grounding | gui | 100 |
| ocr_localization | box | 200 |
| pointing | point | 100 |
| referring | box | 150 |

## Exact prompts by family/protocol

### common_detection · box

```text
Locate every person, backpack, kite in the image. Output a separate tight box for each matching instance; do not merge instances.
Return a separate tight box for ALL matching instances of every requested label. Group ALL boxes for the same label in its coordinate list; do not merge instances or limit the answer to one box per label.
Return each box as [x_min,y_min,x_max,y_max].
Respond only with the requested JSON-like structure grouped by label.
```

### dense_detection · box

```text
Locate every steel pipe in the image. Output a separate tight box for each matching instance; do not merge instances.
Return a separate tight box for ALL matching instances of every requested label. Group ALL boxes for the same label in its coordinate list; do not merge instances or limit the answer to one box per label.
Return each box as [x_min,y_min,x_max,y_max].
Respond only with the requested JSON-like structure grouped by label.
```

### document_layout · box

```text
Locate every Section-header, Text, Caption, Page-footer, Picture in the image. Output a separate tight box for each matching instance; do not merge instances.
Return a separate tight box for ALL matching instances of every requested label. Group ALL boxes for the same label in its coordinate list; do not merge instances or limit the answer to one box per label.
Return each box as [x_min,y_min,x_max,y_max].
Respond only with the requested JSON-like structure grouped by label.
```

### gui_grounding · gui

```text
Locate every region matching the description: access the personal account interface.
Return a separate tight box for ALL matching instances of every requested label. Group ALL boxes for the same label in its coordinate list; do not merge instances or limit the answer to one box per label.
Return each box as [x_min,y_min,x_max,y_max].
Respond only with the requested JSON-like structure grouped by label.
```

### ocr_localization · box

```text
Locate each text item below:
1. "CiTYLOFT"
2. "Coconut shrimp."
3. "Howl of the Moon D"
4. "PARKING"
5. "b"
6. "here's"
7. "is the fruit of the"
8. "shrimp salad,"
9. "shrimp sandwich"
10. "stir fry shrimp,"
Return a separate tight box for ALL matching instances of every requested label. Group ALL boxes for the same label in its coordinate list; do not merge instances or limit the answer to one box per label.
Return each box as [x_min,y_min,x_max,y_max].
Respond only with the requested JSON-like structure grouped by label.
```

### pointing · point

```text
Point to all targets matching the description: donut.
Return a separate point for ALL matching instances. Group all points under their matching label; do not merge instances.
Return each point as [x,y].
Respond only with the requested JSON-like structure grouped by label.
```

### referring · box

```text
Locate every region matching the description: Masayoshi Son.
Return a separate tight box for ALL matching instances of every requested label. Group ALL boxes for the same label in its coordinate list; do not merge instances or limit the answer to one box per label.
Return each box as [x_min,y_min,x_max,y_max].
Respond only with the requested JSON-like structure grouped by label.
```
