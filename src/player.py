import asyncio
import websockets
import pyaudio
import opuslib
from loguru import logger


class AudioPlayer:
    def __init__(self, sample_rate=48000, channels=1, frame_size=960):
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_size = frame_size

        logger.info("Initializing AudioPlayer", sample_rate=sample_rate, channels=channels)
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=self.frame_size,
        )

        # Initialize Opus decoder
        self.decoder = opuslib.Decoder(fs=self.sample_rate, channels=self.channels)

    def cleanup(self):
        logger.info("Cleaning up audio resources")
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

    async def play_audio(self, audio_data):
        logger.debug("Processing audio chunk", chunk_size=len(audio_data))
        decoded_data = await asyncio.get_event_loop().run_in_executor(
            None, self.decoder.decode, audio_data, self.frame_size
        )
        await asyncio.get_event_loop().run_in_executor(
            None, self.stream.write, decoded_data
        )


async def handle_connection(websocket):
    player = AudioPlayer()
    client = websocket.remote_address
    logger.info("New client connected", client=client)
    try:
        async for message in websocket:
            await player.play_audio(message)
    except websockets.exceptions.ConnectionClosed:
        logger.info("Client disconnected", client=client)
    except Exception as e:
        logger.error("Error handling client", client=client, error=str(e))
    finally:
        player.cleanup()


async def main():
    logger.info("Starting WebSocket server")
    try:
        async with websockets.serve(handle_connection, "localhost", 8765) as server:
            logger.success("WebSocket server started", url="ws://localhost:8765")
            await asyncio.Future()
    except Exception as e:
        logger.error("Server error", error=str(e))


if __name__ == "__main__":
    asyncio.run(main())
