import unittest

import decoders.mixed_special as mixed_special
from decoders.mixed_special import decode


class MixedSpecialTests(unittest.TestCase):
    def test_boxes_and_coordinate_like_label_text(self):
        self.assertEqual(
            decode('{"car <123>":[[<100>,<200>,<300>,<400>]]}', 200, 100),
            {"car <123>": [[20, 20, 60, 40]]},
        )

    def test_points(self):
        self.assertEqual(
            decode('{"car":[[<500>,<250>]]}', 200, 100, target_size=2),
            {"car": [[100, 25]]},
        )

    def test_malformed_and_out_of_range(self):
        for raw in [
            '{"car":[[<1>,<2>,<3>,<4>]]',
            '{"car":[[<1001>,0,10,20]]}',
            '{"car":[[true,0,10,20]]}',
            "<think>unfinished",
        ]:
            self.assertEqual(decode(raw, 200, 100), {})

    def test_completed_thinking(self):
        self.assertEqual(
            decode('<think>\n\n</think>\n{"car":[[100,200,300,400]]}<|im_end|>', 200, 100),
            {"car": [[20, 20, 60, 40]]},
        )

    def test_decoder_has_no_prompt_mutation_hook(self):
        self.assertFalse(hasattr(mixed_special, "prepare_prompt"))


if __name__ == "__main__":
    unittest.main()
