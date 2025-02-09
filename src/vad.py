import argparse
from enum import IntEnum

import numpy as np
import soundfile
from funasr import AutoModel

from .utils import disable_tqdm

disable_tqdm()


class VADSignal(IntEnum):
    start = 1
    end = 2


class VoiceActivityDetector:
    def __init__(self, chunk_size: int = 60):
        self.chunk_size = chunk_size
        self.model = AutoModel(
            model="fsmn-vad", model_revision="v2.0.4", disable_update=True
        )

    def detect(self, speech_chunk: np.ndarray, cache: dict) -> VADSignal | None:
        res = self.model.generate(
            input=speech_chunk,
            cache=cache,
            is_final=False,  # we never know it stops for now.
            chunk_size=self.chunk_size,
            speech_noise_thres=0.6,
        )
        if len(res[0]["value"]):
            if res[0]["value"][0][0] == -1:
                return VADSignal.end  # [-1, 2250], means ending at 2250
            else:
                return VADSignal.start  # [90, -1], means starting at 90

    def test_file(self, wav_file: str) -> None:
        speech, sample_rate = soundfile.read(wav_file)
        chunk_stride = int(self.chunk_size * sample_rate / 1000)
        total_chunk_num = int(len(speech - 1) / chunk_stride + 1)

        cache = {}

        for i in range(total_chunk_num):
            speech_chunk = speech[i * chunk_stride : (i + 1) * chunk_stride]
            res = self.detect(speech_chunk, cache)
            if res is not None:
                print(res)


def main():
    parser = argparse.ArgumentParser(description="Voice Activity Detection")
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Path to input WAV file"
    )
    parser.add_argument(
        "-c",
        "--chunk-size",
        type=int,
        default=60,
        help="Chunk size in milliseconds (default: 60)",
    )

    args = parser.parse_args()

    detector = VoiceActivityDetector(chunk_size=args.chunk_size)
    detector.test_file(args.input)


if __name__ == "__main__":
    main()
