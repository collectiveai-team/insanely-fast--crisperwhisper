import os
import time
import subprocess
from typing import Any

import numpy as np
import torch
import requests
from cog import BasePredictor, Input, Path
from torchaudio import functional as F
from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperTokenizerFast,
    pipeline,
)
from transformers.models.whisper.tokenization_whisper import LANGUAGES
from transformers.pipelines.audio_utils import ffmpeg_read

MODEL = "nyrahealth/CrisperWhisper"


def prepare_weights():
    """Shows how to get the weights from HuggingFace Hub and then upload to Replicate google cloud bucket for faster boot time."""
    model_id = MODEL
    torch_dtype = torch.float16
    model_cache = "model_cache"
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        cache_dir=model_cache,
    )

    tokenizer = WhisperTokenizerFast.from_pretrained(model_id, cache_dir=model_cache)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        model_id, cache_dir=model_cache
    )

    pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        model_kwargs={"use_flash_attention_2": True},
        torch_dtype=torch_dtype,
    )


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


def adjust_pauses_for_hf_pipeline_output(pipeline_output, split_threshold=0.12):
    """
    Adjust pause timings by distributing pauses up to the threshold evenly between adjacent words.
    """

    adjusted_chunks = pipeline_output["chunks"].copy()

    for i in range(len(adjusted_chunks) - 1):
        current_chunk = adjusted_chunks[i]
        next_chunk = adjusted_chunks[i + 1]

        current_start, current_end = current_chunk["timestamp"]
        next_start, next_end = next_chunk["timestamp"]
        pause_duration = next_start - current_end

        if pause_duration > 0:
            if pause_duration > split_threshold:
                distribute = split_threshold / 2
            else:
                distribute = pause_duration / 2

            # Adjust current chunk end time
            adjusted_chunks[i]["timestamp"] = (current_start, current_end + distribute)

            # Adjust next chunk start time
            adjusted_chunks[i + 1]["timestamp"] = (next_start - distribute, next_end)
    pipeline_output["chunks"] = adjusted_chunks

    return pipeline_output


class Predictor(BasePredictor):
    def setup(self):
        """Loads whisper models into memory to make running multiple predictions efficient"""
        # model_id = "openai/whisper-large-v3"
        # torch_dtype = torch.float16
        self.device = "cuda:0"

        self.pipe = pipeline(
            task="automatic-speech-recognition",
            model=MODEL,
            device=self.device,
        )

    def predict(
        self,
        audio: Path = Input(description="Audio file"),
        task: str = Input(
            choices=["transcribe", "translate"],
            default="transcribe",
            description="Task to perform: transcribe or translate to another language.",
        ),
        language: str = Input(
            default="None",
            choices=["None"] + sorted(list(LANGUAGES.values())),
            description="Language spoken in the audio, specify 'None' to perform language detection.",
        ),
        batch_size: int = Input(
            default=24,
            description="Number of parallel batches you want to compute. Reduce if you face OOMs.",
        ),
        timestamp: str = Input(
            default="chunk",
            choices=["chunk", "word"],
            description="Whisper supports both chunked as well as word level timestamps.",
        ),
    ) -> Any:
        """Transcribes and optionally translates a single audio file"""

        outputs = self.pipe(
            str(audio),
            chunk_length_s=30,
            batch_size=batch_size,
            generate_kwargs={
                "task": task,
                "language": None if language == "None" else language,
            },
            return_timestamps="word" if timestamp == "word" else True,
        )

        outputs = adjust_pauses_for_hf_pipeline_output(outputs)

        print("Voila!✨ Your file has been transcribed!")
        return outputs


def preprocess_inputs(inputs):
    if isinstance(inputs, str):
        if inputs.startswith("http://") or inputs.startswith("https://"):
            # We need to actually check for a real protocol, otherwise it's impossible to use a local file
            # like http_huggingface_co.png
            inputs = requests.get(inputs).content
        else:
            with open(inputs, "rb") as f:
                inputs = f.read()

    if isinstance(inputs, bytes):
        inputs = ffmpeg_read(inputs, 16000)

    if isinstance(inputs, dict):
        # Accepting `"array"` which is the key defined in `datasets` for better integration
        if not ("sampling_rate" in inputs and ("raw" in inputs or "array" in inputs)):
            raise ValueError(
                "When passing a dictionary to ASRDiarizePipeline, the dict needs to contain a "
                '"raw" key containing the numpy array representing the audio and a "sampling_rate" key, '
                "containing the sampling_rate associated with that array"
            )

        _inputs = inputs.pop("raw", None)
        if _inputs is None:
            # Remove path which will not be used from `datasets`.
            inputs.pop("path", None)
            _inputs = inputs.pop("array", None)
        in_sampling_rate = inputs.pop("sampling_rate")
        inputs = _inputs
        if in_sampling_rate != 16000:
            inputs = F.resample(
                torch.from_numpy(inputs), in_sampling_rate, 16000
            ).numpy()

    if not isinstance(inputs, np.ndarray):
        raise ValueError(f"We expect a numpy ndarray as input, got `{type(inputs)}`")
    if len(inputs.shape) != 1:
        raise ValueError(
            "We expect a single channel audio input for ASRDiarizePipeline"
        )

    # diarization model expects float32 torch tensor of shape `(channels, seq_len)`
    diarizer_inputs = torch.from_numpy(inputs).float()
    diarizer_inputs = diarizer_inputs.unsqueeze(0)

    return inputs, diarizer_inputs


def post_process_segments_and_transcripts(new_segments, transcript, group_by_speaker):
    # get the end timestamps for each chunk from the ASR output
    end_timestamps = np.array([chunk["timestamp"][-1] for chunk in transcript])
    segmented_preds = []

    # align the diarizer timestamps and the ASR timestamps
    for segment in new_segments:
        # get the diarizer end timestamp
        end_time = segment["segment"]["end"]
        # find the ASR end timestamp that is closest to the diarizer's end timestamp and cut the transcript to here
        upto_idx = np.argmin(np.abs(end_timestamps - end_time))

        if group_by_speaker:
            segmented_preds.append(
                {
                    "speaker": segment["speaker"],
                    "text": "".join(
                        [chunk["text"] for chunk in transcript[: upto_idx + 1]]
                    ),
                    "timestamp": (
                        transcript[0]["timestamp"][0],
                        transcript[upto_idx]["timestamp"][1],
                    ),
                }
            )
        else:
            for i in range(upto_idx + 1):
                segmented_preds.append({"speaker": segment["speaker"], **transcript[i]})

        # crop the transcripts and timestamp lists according to the latest timestamp (for faster argmin)
        transcript = transcript[upto_idx + 1 :]
        end_timestamps = end_timestamps[upto_idx + 1 :]

    return segmented_preds


if __name__ == "__main__":
    prepare_weights()
