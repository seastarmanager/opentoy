from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess


class SenseVoiceASR:
    def __init__(
        self,
        model_dir="iic/SenseVoiceSmall",
        device="cuda:0",
    ):
        self.model = AutoModel(
            model=model_dir,
            device=device,
            disable_update=True,
        )

    def to_text(
        self,
        audio_data: bytes,
        language="zh",
        use_itn=True,
        batch_size_s=60,
        merge_vad=True,
        merge_length_s=15,
    ) -> str:
        res = self.model.generate(
            input=audio_data,
            language=language,
            use_itn=use_itn,
            batch_size_s=batch_size_s,
            merge_vad=merge_vad,
            merge_length_s=merge_length_s,
        )
        return rich_transcription_postprocess(res[0]["text"])


def main():
    asr = SenseVoiceASR()
    with open("sample.wav", "rb") as f:
        wav = f.read()
    text = asr.to_text(wav)
    print(text)


if __name__ == "__main__":
    main()
