#!/usr/bin/env python3
"""Regression tests for discrete spatial-token decoding."""

import sys
import unittest
from pathlib import Path


EVALUATOR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVALUATOR))

from decoders.discrete_ntp import decode


class DiscreteDecoderTest(unittest.TestCase):
    """Check complete groups, malformed output, and coordinate scaling."""

    def test_multiple_references_and_boxes(self):
        raw = (
            "<ref>car</ref><box><100><200><300><400></box>"
            "<box><500><600><700><800></box>"
            "<ref>person</ref><box><0><0><1000><1000></box>"
        )
        result = decode(raw, 2000, 1000)
        self.assertEqual(result["car"], [[200, 200, 600, 400], [1000, 600, 1400, 800]])
        self.assertEqual(result["person"], [[0, 0, 2000, 1000]])

    def test_incomplete_groups_are_rejected(self):
        raw = (
            "<ref>open ref<box><1><2><3><4></box>"
            "<ref>short</ref><box><1><2><3></box>"
            "<ref>too many</ref><box><1><2><3><4><5></box>"
            "<ref>valid</ref><box><1><2><3><4></box>"
        )
        self.assertEqual(list(decode(raw, 1000, 1000)), ["valid"])

    def test_out_of_range_box_is_rejected(self):
        raw = "<ref>bad</ref><box><0><0><1001><1000></box>"
        self.assertEqual(decode(raw, 640, 480), {})

    def test_reversed_corners_are_sorted(self):
        raw = "<ref>thing</ref><box><900><800><100><200></box>"
        self.assertEqual(decode(raw, 1000, 500), {"thing": [[100, 100, 900, 400]]})

    def test_surrounding_text_does_not_change_complete_groups(self):
        raw = "noise <ref>cat</ref> <box><10><20><30><40></box> trailing"
        self.assertEqual(decode(raw, 1000, 1000), {"cat": [[10, 20, 30, 40]]})

    def test_point_tokens_decode_to_absolute_coordinates(self):
        raw = "<ref>donut</ref><box><250><750></box><box><1000><0></box>"
        self.assertEqual(
            decode(raw, 800, 400, target_size=2),
            {"donut": [[200, 300], [800, 0]]},
        )

    def test_point_tokens_are_not_accepted_as_boxes(self):
        raw = "<ref>donut</ref><box><250><750></box>"
        self.assertEqual(decode(raw, 800, 400), {})


if __name__ == "__main__":
    unittest.main()
