# coding=utf-8

class TTSConfig:
    def __init__(self):
        self.model_name = "cosyvoice-v1"
        self.voice_person = "longxiaochun"
        self.api_key = "sk-********"


class LLMConfig:
    def __init__(self):
        #self.model_name = "deepseek-r1"
        self.model_name = "qwen-plus-2025-01-12"
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.api_key = "sk-********"

class ASRConfig:
    def __init__(self):
        #self.model_name = "sensevoice-v1"
        self.model_name = "paraformer-realtime-v2"
        self.api_key = "sk-********"

class VADConfig:
    def __init__(self):
        self.model_name = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"


class Config:
    def __init__(self):
        self.tts = TTSConfig()
        self.llm = LLMConfig()
        self.asr = ASRConfig()
        self.vad = VADConfig()

