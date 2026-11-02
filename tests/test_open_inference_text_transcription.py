from __future__ import annotations

import json
import importlib.util
import os
import threading
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib import error, request

import torch

from task_benchmark.abstract.audio_sample import AudioSample
from task_benchmark.implementations.invoker.text_transcription import (
    OpenInferenceTextTranscriptionInvoker,
)


class OpenInferenceTextTranscriptionInvokerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.invoker = OpenInferenceTextTranscriptionInvoker(
            "http://localhost:8081/v2/models/whisper-tiny.en/infer"
        )

    def test_posts_waveform_and_decodes_text(self) -> None:
        result = {"outputs": [{"name": "text", "datatype": "BYTES",
                               "shape": [1], "data": ["Hello world."]}]}
        with patch(
            "task_benchmark.implementations.invoker.text_transcription.open_inference.request.urlopen",
            return_value=BytesIO(json.dumps(result).encode()),
        ) as urlopen:
            text = self.invoker.invoke(AudioSample(torch.tensor([0.0, 0.5]), 16000))
        self.assertEqual(text, "Hello world.")
        payload = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(payload["inputs"][0]["data"], [0.0, 0.5])
        self.assertEqual(payload["inputs"][1]["data"], [16000])
        self.assertEqual(payload["outputs"], [{"name": "text"}])

    def test_rejects_malformed_text_response(self) -> None:
        for result in ({}, {"outputs": []}, {"outputs": [{
            "name": "text", "datatype": "FP32", "shape": [1], "data": [0.5],
        }]}):
            with self.subTest(result=result):
                with patch(
                    "task_benchmark.implementations.invoker.text_transcription.open_inference.request.urlopen",
                    return_value=BytesIO(json.dumps(result).encode()),
                ):
                    with self.assertRaises(RuntimeError):
                        self.invoker.invoke(AudioSample(torch.zeros(100), 16000))


class WhisperV2ServerTests(unittest.TestCase):
    def setUp(self) -> None:
        path = (Path(__file__).resolve().parents[1] / "examples" / "text_transcription"
                / "open_inference_simple" / "server.py")
        spec = importlib.util.spec_from_file_location("whisper_v2_server", path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.calls = []

        def transcribe(audio, sampling_rate):
            self.calls.append((audio, sampling_rate))
            return "Hello world."

        backend = SimpleNamespace(max_duration=30.0, transcribe=transcribe)
        self.server = self.module.create_server(backend, "127.0.0.1", 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_metadata_and_transcription_over_http(self) -> None:
        for path in ("/v2/health/live", "/v2/health/ready", "/v2/models/whisper-tiny.en/ready"):
            with request.urlopen(self.base_url + path) as response:
                self.assertEqual(response.status, 200)
        with request.urlopen(self.base_url + "/v2/models/whisper-tiny.en") as response:
            metadata = json.load(response)
        self.assertEqual(metadata["outputs"][0]["datatype"], "BYTES")
        invoker = OpenInferenceTextTranscriptionInvoker(self.base_url + "/v2/models/whisper-tiny.en/infer")
        self.assertEqual(invoker.invoke(AudioSample(torch.zeros(8000), 8000)), "Hello world.")
        self.assertEqual(self.calls[0][1], 8000)

    def test_invalid_audio_returns_v2_error(self) -> None:
        payload = {"inputs": [
            {"name": "audio", "datatype": "FP32", "shape": [1], "data": [0.0]},
            {"name": "sampling_rate", "datatype": "INT64", "shape": [1], "data": [0]},
        ]}
        http_request = request.Request(
            self.base_url + "/v2/models/whisper-tiny.en/infer",
            data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"},
        )
        with self.assertRaises(error.HTTPError) as caught:
            request.urlopen(http_request)
        self.assertEqual(caught.exception.code, 400)
        self.assertIn("error", json.loads(caught.exception.read()))
        self.assertEqual(self.calls, [])

    def test_dataset_client_main_over_http(self) -> None:
        from task_benchmark.implementations.data_loader.text_transcription import TextTranscriptionDataLoader

        class DatasetLoader(TextTranscriptionDataLoader):
            def __iter__(self):
                return iter([("sample-1", AudioSample(torch.zeros(16000), 16000))])

            def next(self):
                raise StopIteration

            def transcript_for(self, sample_id):
                return "HELLO WORLD"

        path = (Path(__file__).resolve().parents[1] / "examples" / "text_transcription"
                / "open_inference_simple" / "open_inference_hf.py")
        spec = importlib.util.spec_from_file_location("whisper_remote_example", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with (
            patch.dict(os.environ, {
                "OPEN_INFERENCE_BASE_URL": self.base_url + "/v2/models/whisper-tiny.en",
                "HF_MAX_SAMPLES": "1",
            }, clear=True),
            patch.object(module, "HuggingFaceTextTranscriptionDataLoader", return_value=DatasetLoader("test", "")),
            patch("builtins.print"),
        ):
            report = module.main()
        self.assertEqual(report["samples"], 1)
        self.assertEqual(report["mean_sample_wer"], 0.0)
        self.assertEqual(report["mean_quality_score"], 1.0)


if __name__ == "__main__":
    unittest.main()