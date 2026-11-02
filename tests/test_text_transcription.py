from __future__ import annotations

import unittest
import importlib.util
import os
import sys
from io import BytesIO
from types import ModuleType, SimpleNamespace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import soundfile
import torch
from datasets import Dataset

from task_benchmark.abstract.audio_sample import AudioSample
from task_benchmark.implementations.data_loader.text_transcription import (
    TextTranscriptionDataLoader, HuggingFaceTextTranscriptionDataLoader,
)
from task_benchmark.implementations.invoker.text_transcription import (
    TextTranscriptionInvoker, HuggingFaceTextTranscriptionInvoker,
)
from task_benchmark.implementations.evaluator.text_transcription import (
    TextTranscriptionEvaluator, WordErrorRateEvaluator,
)


class TextTranscriptionContractTests(unittest.TestCase):
    def test_bases_are_abstract(self) -> None:
        for base, args in [
            (TextTranscriptionDataLoader, ("dataset", "./data")),
            (TextTranscriptionInvoker, ()),
            (TextTranscriptionEvaluator, ()),
        ]:
            with self.subTest(base=base):
                with self.assertRaises(TypeError):
                    base(*args)


class TextTranscriptionDataLoaderTests(unittest.TestCase):
    def dataset(self, sample_id="sample-1", transcript="Hello world"):
        buffer = BytesIO()
        stereo = np.column_stack([np.ones(80) * 0.25, np.ones(80) * 0.75])
        soundfile.write(buffer, stereo, 8000, format="FLAC")
        return Dataset.from_dict({
            "audio": [{"bytes": buffer.getvalue(), "path": None}],
            "text": [transcript], "id": [sample_id],
        })

    def test_decodes_audio_and_retains_reference(self) -> None:
        dataset = self.dataset().to_iterable_dataset()
        with patch("datasets.load_dataset", return_value=dataset) as load_dataset:
            loader = HuggingFaceTextTranscriptionDataLoader(
                "sample/dataset", "./cache", {"name": "clean", "streaming": True}
            )
            sample_id, sample = next(loader)
        load_dataset.assert_called_once_with(
            "sample/dataset", split="validation", cache_dir="./cache",
            name="clean", streaming=True,
        )
        self.assertIsInstance(loader, TextTranscriptionDataLoader)
        self.assertEqual(sample_id, "sample-1")
        self.assertEqual(loader.transcript_for(sample_id), "Hello world")
        self.assertEqual(sample.sampling_rate, 8000)
        torch.testing.assert_close(sample.waveform, torch.full((80,), 0.5))
        self.assertEqual(loader.current_index, 1)
        with self.assertRaises(StopIteration):
            next(loader)
        with self.assertRaises(KeyError):
            loader.transcript_for("unknown")

    def test_generates_identifier_when_missing(self) -> None:
        with patch("datasets.load_dataset", return_value=self.dataset(sample_id=None)):
            loader = HuggingFaceTextTranscriptionDataLoader("sample/dataset", "")
            sample_id, _ = loader.next()
        self.assertEqual(sample_id, "sample/dataset:validation:0")


class TextTranscriptionInvokerTests(unittest.TestCase):
    def test_real_whisper_feature_extractor_accepts_waveform(self) -> None:
        from transformers import WhisperFeatureExtractor

        features = WhisperFeatureExtractor()(
            np.zeros(16000, dtype=np.float32), sampling_rate=16000,
            return_tensors="pt", return_attention_mask=True,
        )
        self.assertEqual(tuple(features["input_features"].shape), (1, 80, 3000))
        self.assertEqual(tuple(features["attention_mask"].shape), (1, 3000))

    def setUp(self) -> None:
        self.processor = Mock(return_value={
            "input_features": torch.zeros(1, 80, 3000),
            "attention_mask": torch.ones(1, 3000),
        })
        self.processor.feature_extractor = SimpleNamespace(sampling_rate=16000, chunk_length=30)
        self.processor.batch_decode.return_value = [" Hello world. "]
        self.model = Mock()
        self.inference_modes = []

        def generate(**kwargs):
            self.inference_modes.append(torch.is_inference_mode_enabled())
            return torch.tensor([[1, 2]])

        self.model.generate.side_effect = generate
        self.processor_factory = Mock(return_value=self.processor)
        self.model_factory = Mock(return_value=self.model)
        transformers = ModuleType("transformers")
        transformers.AutoProcessor = SimpleNamespace(from_pretrained=self.processor_factory)
        transformers.AutoModelForSpeechSeq2Seq = SimpleNamespace(from_pretrained=self.model_factory)
        with patch.dict(sys.modules, {"transformers": transformers}):
            self.invoker = HuggingFaceTextTranscriptionInvoker(
                model_kwargs={"revision": "main"}, generate_kwargs={"max_new_tokens": 30}
            )

    def test_resamples_and_generates_without_gradients(self) -> None:
        sample = AudioSample(torch.zeros(4000), 8000)
        text = self.invoker.invoke(sample)
        self.assertIsInstance(self.invoker, TextTranscriptionInvoker)
        self.assertEqual(text, "Hello world.")
        self.assertEqual(len(self.processor.call_args.args[0]), 8000)
        self.assertEqual(self.processor.call_args.kwargs["sampling_rate"], 16000)
        self.assertTrue(self.processor.call_args.kwargs["return_attention_mask"])
        self.assertEqual(self.model.generate.call_args.kwargs["max_new_tokens"], 30)
        self.assertEqual(self.inference_modes, [True])
        self.model.eval.assert_called_once_with()
        self.model.to.assert_called_once_with(torch.device("cpu"))
        self.model_factory.assert_called_once_with("openai/whisper-tiny.en", revision="main")

    def test_rejects_invalid_or_long_audio(self) -> None:
        for sample in (
            AudioSample(torch.zeros(1), 0), AudioSample(torch.zeros(1), True),
            AudioSample(torch.zeros(2, 2), 16000), AudioSample(torch.zeros(0), 16000),
            AudioSample(torch.tensor([float("nan")]), 16000),
            AudioSample(torch.zeros(480001), 16000),
        ):
            with self.subTest(sample=sample):
                with self.assertRaises(ValueError):
                    self.invoker.invoke(sample)
        with self.assertRaises(TypeError):
            self.invoker.invoke(torch.zeros(100))
        self.processor.assert_not_called()


class ReferenceLoader(TextTranscriptionDataLoader):
    def next(self) -> tuple[str, AudioSample]:
        raise StopIteration

    def transcript_for(self, sample_id: str) -> str:
        return {"sample-1": "Hello world", "silence": ""}[sample_id]


class WordErrorRateEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.evaluator = WordErrorRateEvaluator(ReferenceLoader("test", ""))

    def test_normalizes_case_and_punctuation(self) -> None:
        self.assertEqual(self.evaluator.evaluate("sample-1", " HELLO,   world! "), 1.0)

    def test_partial_and_empty_predictions(self) -> None:
        self.assertEqual(self.evaluator.word_error_rate("sample-1", "hello"), 0.5)
        self.assertEqual(self.evaluator.evaluate("sample-1", "hello"), 0.5)
        self.assertEqual(self.evaluator.evaluate("sample-1", ""), 0.0)

    def test_insertion_errors_are_clamped_only_in_quality_score(self) -> None:
        prediction = "one two three four five six"
        self.assertGreater(self.evaluator.word_error_rate("sample-1", prediction), 1.0)
        self.assertEqual(self.evaluator.evaluate("sample-1", prediction), 0.0)
        self.assertEqual(self.evaluator.evaluate("silence", ""), 1.0)
        self.assertEqual(self.evaluator.evaluate("silence", "hello"), 0.0)

    def test_missing_reference_and_invalid_response(self) -> None:
        with self.assertRaises(KeyError):
            self.evaluator.evaluate("unknown", "hello")
        with self.assertRaises(TypeError):
            self.evaluator.evaluate("sample-1", torch.zeros(1))


class TextTranscriptionExampleTests(unittest.TestCase):
    def test_main_connects_loader_invoker_and_evaluator(self) -> None:
        path = (Path(__file__).resolve().parents[1] / "examples" / "text_transcription"
                / "huggingface" / "huggingface_text_transcription.py")
        spec = importlib.util.spec_from_file_location("text_transcription_example", path)
        example = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(example)

        class ExampleLoader(ReferenceLoader):
            def __iter__(self):
                return iter([("sample-1", AudioSample(torch.zeros(16000), 16000))])

        loader = ExampleLoader("sample", "")
        invoker = Mock(spec=TextTranscriptionInvoker)
        invoker.invoke.return_value = "Hello world!"
        with (
            patch.dict(os.environ, {"HF_MAX_SAMPLES": "1", "DEVICE": "cpu"}, clear=True),
            patch.object(example, "HuggingFaceTextTranscriptionDataLoader", return_value=loader) as loader_factory,
            patch.object(example, "HuggingFaceTextTranscriptionInvoker", return_value=invoker) as invoker_factory,
            patch("builtins.print"),
        ):
            report = example.main()
        self.assertEqual(report["samples"], 1)
        self.assertEqual(report["mean_sample_wer"], 0.0)
        self.assertEqual(report["mean_quality_score"], 1.0)
        self.assertEqual(report["results"][0]["audio_seconds"], 1.0)
        self.assertEqual(loader_factory.call_args.args[2]["name"], "clean")
        self.assertEqual(invoker_factory.call_args.kwargs["model_name"], "openai/whisper-tiny.en")


if __name__ == "__main__":
    unittest.main()