from fish_audio_sdk import ReferenceAudio, TTSRequest, WebSocketSession
import pyaudio
from loguru import logger

from .llm import LLM

sync_websocket = WebSocketSession("3d52ab6a36a84508887c5764221c8ad8")


class TTS:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=48000,
            output=True,
            frames_per_buffer=960,
        )

    def synth(self, text_stream):
        tts_request = TTSRequest(
            text="",
            reference_id="54a5170264694bfc8e9ad98df7bd89c3",
            format="wav",
        )

        try:
            # Accumulate MP3 data
            for chunk in sync_websocket.tts(tts_request, text_stream):
                self.stream.write(chunk)
                yield chunk

        except Exception as e:
            logger.error("TTS synthesis error", error=str(e))

    def cleanup(self):
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'audio'):
            self.audio.terminate()


if __name__ == "__main__":
    tts = TTS()
    llm = LLM()
    try:
        for _ in tts.synth(llm.generate("模仿丁真谈谈天气")):
            pass
    finally:
        tts.cleanup()
