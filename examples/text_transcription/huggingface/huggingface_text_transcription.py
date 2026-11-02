from __future__ import annotations

import json
import os
from itertools import islice
from time import perf_counter
from typing import Any

from task_benchmark.implementations.data_loader.text_transcription import (
    HuggingFaceTextTranscriptionDataLoader,
)
from task_benchmark.implementations.invoker.text_transcription import (
    HuggingFaceTextTranscriptionInvoker,
)
from task_benchmark.implementations.evaluator.text_transcription import WordErrorRateEvaluator


def main() -> dict[str, Any]:
    max_samples = int(os.getenv("HF_MAX_SAMPLES", "50"))
    if max_samples < 1:
        raise ValueError("HF_MAX_SAMPLES must be greater than zero")
    dataset_id = os.getenv("HF_DATASET_ID", "hf-internal-testing/librispeech_asr_dummy")
    model_name = os.getenv("HF_MODEL_ID", "openai/whisper-tiny.en")
    cache_dir = os.getenv("HF_CACHE_DIR", ".cache/huggingface")
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
    loader = HuggingFaceTextTranscriptionDataLoader(dataset_id, cache_dir, params)
    invoker = HuggingFaceTextTranscriptionInvoker(
        model_name=model_name,
        device=os.getenv("DEVICE", "cpu"),
        model_kwargs={"cache_dir": cache_dir},
        processor_kwargs={"cache_dir": cache_dir},
    )
    evaluator = WordErrorRateEvaluator(loader)
    results = []
    started = perf_counter()
    for sample_id, audio in islice(loader, max_samples):
        inference_started = perf_counter()
        transcription = invoker.invoke(audio)
        inference_seconds = perf_counter() - inference_started
        score = evaluator.evaluate(sample_id, transcription)
        wer = evaluator.word_error_rate(sample_id, transcription)
        reference = loader.transcript_for(sample_id)
        results.append({
            "sample_id": sample_id,
            "reference": reference,
            "transcription": transcription,
            "wer": wer,
            "quality_score": score,
            "audio_seconds": audio.waveform.numel() / audio.sampling_rate,
            "inference_seconds": inference_seconds,
        })
        print(f"{sample_id}: WER={wer:.3f}, quality_score={score:.3f}")
        print(f"  reference: {reference}")
        print(f"  predicted: {transcription}")
    if not results:
        raise ValueError("The dataset split contains no samples")
    report = {
        "dataset_id": dataset_id,
        "model_name": model_name,
        "samples": len(results),
        "mean_sample_wer": sum(item["wer"] for item in results) / len(results),
        "mean_quality_score": sum(item["quality_score"] for item in results) / len(results),
        "mean_inference_seconds": sum(item["inference_seconds"] for item in results) / len(results),
        "elapsed_seconds": perf_counter() - started,
        "results": results,
    }
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()