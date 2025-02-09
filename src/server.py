import asyncio
from collections import deque

import numpy as np
import opuslib
import websockets
from loguru import logger

from .asr import SenseVoiceASR
from .llm import LLM
from .tts import TTS
from .vad import VADSignal, VoiceActivityDetector


class AudioProcessor:
    def __init__(self, sample_rate=48000, channels=1, frame_size=28800):
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_size = frame_size

        # Initialize components
        self.decoder = opuslib.Decoder(fs=sample_rate, channels=channels)
        self.vad = VoiceActivityDetector()
        self.asr = SenseVoiceASR()
        self.llm = LLM()
        self.tts = TTS()

        # Audio buffer
        self.buffer = deque()
        self.recording = False
        self.cache = {}

    async def _process(self, audio_data: bytes):
        # Decode opus data
        decoded_data = await asyncio.get_event_loop().run_in_executor(
            None, self.decoder.decode, audio_data, self.frame_size
        )

        # Add to buffer
        self.buffer.append(decoded_data)

        # Check VAD
        signal = self.vad.detect(
            np.frombuffer(decoded_data, dtype=np.int16), self.cache
        )

        if signal == VADSignal.start and not self.recording:
            self.recording = True
            logger.info("Speech started")

        elif signal == VADSignal.end and self.recording:
            self.recording = False
            logger.info("Speech ended")

            # Combine all buffered audio
            full_audio = b"".join(self.buffer)
            
            # Play the recorded audio first
            self.tts.stream.write(full_audio)
            
            self.buffer.clear()

            # Process through ASR
            text = self.asr.to_text(full_audio)
            text = text.strip()
            if len(text) <= 2:
                return
            logger.info("ASR result '{}'", text)

            # Process through LLM and TTS
            llm_response = self.llm.generate(text)
            for chunk in self.tts.synth(llm_response):
                yield chunk

    async def handle_connection(self, websocket):
        try:
            async for message in websocket:
                async for audio_response in self._process(message):
                    await websocket.send(audio_response)
        except Exception as e:
            logger.error("Connection error", error=str(e))


async def main():
    processor = AudioProcessor()
    logger.info("Starting WebSocket server")
    try:
        async with websockets.serve(
            processor.handle_connection, "localhost", 8765
        ) as server:
            logger.success("WebSocket server started", url="ws://localhost:8765")
            await asyncio.Future()
    except Exception as e:
        logger.error("Server error", error=str(e))


if __name__ == "__main__":
    asyncio.run(main())
