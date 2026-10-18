"""Endpoint and model checks (context feasibility needs an inference benchmark)."""
from .ollama_client import Ollama


def doctor(config, log):
    client = Ollama(config.ollama_url, timeout=15)
    log.kv("Ollama", client.version())
    models = {m["name"] for m in client.list_models()}
    missing = []
    for model in dict.fromkeys((config.planner_model, config.worker_model)):
        found = model in models or model + ":latest" in models
        log.kv(model, "available" if found else "MISSING")
        if not found:
            missing.append(model)
    log.kv("Context / output", f"{config.num_ctx} / {config.output_reserve}")
    log.info("VRAM fit is not inferred from model metadata; verify with a real run.")
    return not missing
