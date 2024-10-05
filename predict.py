import time
import subprocess
from functools import lru_cache
from typing import Any

import torch
from cog import BasePredictor, Input, Path
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.models.whisper.tokenization_whisper import LANGUAGES

MODEL = "nyrahealth/CrisperWhisper"


@lru_cache(maxsize=None)
def get_pipeline(token: str):
    """Loads whisper models into memory to make running multiple predictions efficient"""
    device = "cuda:0"
    torch_dtype = torch.float16

    print("Loading model...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL,
        token=token,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(
        MODEL,
        token=token,
    )

    print("Loading pipeline...")
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        model_kwargs={"use_flash_attention_2": True},
        torch_dtype=torch_dtype,
        return_timestamps="word",
        return_language=True,
        device=device,
    )
    print("Done loading.")

    return pipe


def prepare_weights():
    """Shows how to get the weights from HuggingFace Hub and then upload to Replicate google cloud bucket for faster boot time."""
    model_id = MODEL
    torch_dtype = torch.float16

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )

    processor = AutoProcessor.from_pretrained(model_id)

    pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
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
    def predict(
        self,
        hf_token: str = Input(
            description="HuggingFace token to access CrispWhisper models.",
        ),
        audio: Path = Input(
            "https://replicate.delivery/pbxt/IZjTvet2ZGiyiYaMEEPrzn0xY1UDNsh0NfcO9qeTlpwCo7ig/lex-levin-4min.mp3",
            description="Audio file",
        ),
        language: str = Input(
            default="None",
            choices=["None"] + sorted(list(LANGUAGES.values())),
            description="Language spoken in the audio, specify 'None' to perform language detection.",
        ),
        batch_size: int = Input(
            default=4,
            description="Number of parallel batches you want to compute. Reduce if you face OOMs.",
        ),
    ) -> Any:
        """Transcribes and optionally translates a single audio file"""

        pipe = get_pipeline(hf_token)

        print("Starting transcription...")
        outputs = pipe(
            str(audio),
            chunk_length_s=30,
            batch_size=batch_size,
            generate_kwargs={
                "language": None if language == "None" else language,
            },
            return_timestamps="word",
        )

        print("Adjusting pauses...")
        outputs = adjust_pauses_for_hf_pipeline_output(outputs)

        print("Voila!✨ Your file has been transcribed!")
        return outputs


if __name__ == "__main__":
    prepare_weights()
