"""Match HF's output-only repetition penalty for inputs_embeds generation."""

import torch
from transformers import RepetitionPenaltyLogitsProcessor
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor


class ElliotRepetitionPenalty(AdapterLogitsProcessor):
    def is_argmax_invariant(self):
        return False

    def new_req_logits_processor(self, params):
        penalty = (params.extra_args or {}).get("elliot_repetition_penalty", 1.0)
        if penalty == 1.0:
            return None
        processor = RepetitionPenaltyLogitsProcessor(float(penalty))

        def apply(output_ids, logits):
            if not output_ids:
                return logits
            ids = torch.tensor([output_ids], device=logits.device, dtype=torch.long)
            return processor(ids, logits.unsqueeze(0)).squeeze(0)

        return apply
