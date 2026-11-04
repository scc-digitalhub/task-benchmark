from __future__ import annotations

import json
import os

from task_benchmark import BenchmarkManager
from task_benchmark.implementations.data_loader.text_transcription import (
    HuggingFaceTextTranscriptionDataLoader,
)
from task_benchmark.implementations.evaluator.text_transcription import WordErrorRateEvaluator
from task_benchmark.implementations.invoker.text_transcription import (
    OpenInferenceTextTranscriptionInvoker,
)
from task_benchmark.implementations.metrics_manager.base_metrics_manager import JSONValue


def main() -> JSONValue:
    base_url = os.getenv(
        "OPEN_INFERENCE_BASE_URL", "http://localhost:9082/v2/models/whisper-tiny.en"
    ).rstrip("/").removesuffix("/infer")
    dataset_id = os.getenv("HF_DATASET_ID", "hf-internal-testing/librispeech_asr_dummy")
    max_samples = int(os.getenv("HF_MAX_SAMPLES", "50"))
    if max_samples < 1:
        raise ValueError("HF_MAX_SAMPLES must be greater than zero")
    params = {
        "split": os.getenv("HF_DATASET_SPLIT", "validation"),
        "audio_column": os.getenv("HF_AUDIO_COLUMN", "audio"),
        "text_column": os.getenv("HF_TEXT_COLUMN", "text"),
        "id_column": os.getenv("HF_ID_COLUMN", "id"),
        "streaming": True,
    }
    config_name = os.getenv("HF_DATASET_CONFIG", "clean")
    if config_name:
        params["name"] = config_name
    loader = HuggingFaceTextTranscriptionDataLoader(
        dataset_id, os.getenv("HF_CACHE_DIR", ".cache/huggingface"), params,
        max_samples=max_samples,
    )
    invoker = OpenInferenceTextTranscriptionInvoker(f"{base_url}/infer")
    evaluator = WordErrorRateEvaluator(loader)
    metrics = BenchmarkManager(
        data_loader=loader,
        invoker=invoker,
        evaluator=evaluator,
        service_id=base_url,
    ).run_benchmark()
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    main()