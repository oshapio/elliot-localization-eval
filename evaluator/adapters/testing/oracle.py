"""Ground-truth adapter used only to validate evaluator plumbing."""


def create_adapter(config):
    """Create a deterministic perfect-prediction adapter."""
    def predict_batch(records, prompts, max_new_tokens):
        return [{"predictions": record["gt"], "raw_response": "oracle"} for record in records]

    return {"name": config.get("name", "oracle"), "predict_batch": predict_batch}
