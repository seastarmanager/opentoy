# -*- coding: utf-8 -*-
import numpy as np
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

class ASRModel:
    def __init__(self, asr_config):
        self.model = AutoModel(
            model="iic/SenseVoiceSmall",
            device="cuda",
            disable_update=True,
            disable_pbar=True,
        )

    def recognize(self, audio_buffer, is_np = False):
        frame_fp32 = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768 if not is_np else audio_buffer
        result = self.model.generate(input=frame_fp32, cache={}, language='zh', use_itn=True)
        return rich_transcription_postprocess(result[0]['text'])


# if __name__ == '__main__':
#     import config
#     import soundfile
#     import os
#     cfg = config.Config()
#     model = ASRModel(cfg.asr)
#     chunk_size = [0, 10, 5]
#
#     wav_file = os.path.join('../.temp_data', "recording_5464766800_20250209214052.wav")
#     speech, sample_rate = soundfile.read(wav_file)
#     #chunk_stride = chunk_size[1] * 320  # 600ms
#     res = model.recognize(speech, is_np=True)
#     print(res)