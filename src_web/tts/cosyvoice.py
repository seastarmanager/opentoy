# coding=utf-8

import dashscope
from dashscope.audio.tts_v2 import *

class TTSModel:
    def __init__(self, tts_config):
        self.tts_config = tts_config
        dashscope.api_key = tts_config.api_key
        self.output_format = 'mp3'
        self.synthesizer=None

    def generate_audio(self, text, file_path=None):
        # file_path为mp3文件保存路径
        tts_config = self.tts_config
        self.synthesizer = SpeechSynthesizer(model=tts_config.model_name, voice=tts_config.voice_person)
        audio = self.synthesizer.call(text)
        if audio is None:
            # print('text: ', text)
            audio = self.synthesizer.call(text)
        print('requestId: ', self.synthesizer.get_last_request_id())
        if file_path:
            with open(file_path, 'wb') as f:
                f.write(audio)
        return audio


# if __name__ == '__main__':
#     import config
#     cfg = config.Config()
#     model = TTSModel(cfg.tts)
#     audio = model.generate_audio("今天天气怎么样？", '../.temp_data/output.mp3')
