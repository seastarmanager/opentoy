import asyncio
from asyncio import Queue

import opuslib
import pyaudio
import websockets
from loguru import logger


class AudioStreamer:
    def __init__(self, sample_rate=48000, channels=1, frame_size=2880):  # 60ms
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_size = frame_size

        logger.info(
            f"Initializing AudioStreamer with rate={sample_rate}, channels={channels}"
        )
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.frame_size,
        )

        # Initialize Opus encoder
        self.encoder = opuslib.Encoder(
            fs=self.sample_rate,
            channels=self.channels,
            application=opuslib.APPLICATION_AUDIO,
        )

        self.audio_queue = Queue()
        self.is_running = False

    async def start_recording(self):
        self.is_running = True
        logger.info("Starting audio recording")
        while self.is_running:
            audio_data = await asyncio.get_event_loop().run_in_executor(
                None, self.stream.read, self.frame_size
            )
            encoded_data = await asyncio.get_event_loop().run_in_executor(
                None, self.encoder.encode, audio_data, self.frame_size
            )
            await self.audio_queue.put(encoded_data)
            # logger.debug(f"Encoded audio chunk of size {len(encoded_data)}")

    def stop_recording(self):
        logger.info("Stopping audio recording")
        self.is_running = False
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

    async def stream_audio(self, websocket_url):
        logger.info(f"Starting audio streaming to {websocket_url}")
        async with websockets.connect(websocket_url) as websocket:
            while self.is_running:
                encoded_data = await self.audio_queue.get()
                await websocket.send(encoded_data)
                # logger.debug(f"Sent audio chunk of size {len(encoded_data)}")


async def main():
    streamer = AudioStreamer()
    try:
        logger.info("Starting audio streaming service")
        await asyncio.gather(
            streamer.start_recording(), streamer.stream_audio("ws://localhost:8765")
        )
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down")
        streamer.stop_recording()
    except Exception as e:
        logger.error(f"Error in audio streaming: {str(e)}")
        streamer.stop_recording()
    finally:
        streamer.stop_recording()


if __name__ == "__main__":
    asyncio.run(main())
