"""Empty-prediction adapter used to validate failure baselines."""


def create_adapter(config):
    """Create a deterministic empty-prediction adapter."""
    def predict_batch(records, prompts, max_new_tokens):
        return [{"predictions": {}, "raw_response": ""} for record in records]

    return {"name": config.get("name", "empty"), "predict_batch": predict_batch}
