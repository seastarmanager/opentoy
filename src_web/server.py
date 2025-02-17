import asyncio
import websockets
import wave
import os
from datetime import datetime
from vad.fsmn_vad import VADModel
from asr.sensevoice import ASRModel
from llm.qwen_plus import LLMModel
from tts.cosyvoice import TTSModel
import config
import time
import json
import base64


cfg = config.Config()
vad_model = VADModel(cfg.vad)
print("VAD model loaded")
asr_model = ASRModel(cfg.asr)
print("ASR model loaded")
# llm_model = LLMModel(cfg.llm)
# print("LLM model loaded")
tts_model = TTSModel(cfg.tts)
print("TTS model loaded")


class AudioBuffer:
    def __init__(self, client_id, llm, websocket):
        self.buffer = bytearray()
        self.vad = vad_model
        self.asr = asr_model
        self.llm = llm
        self.tts = tts_model
        self.vad_res = []
        self.speech_frames = 0
        self.vad_cache = {}
        self.silence_thr = 0
        self.trunc_begin = 0
        self.trunc_end = 0
        self.status = 'idle' # idle, listening, replying
        self.frame_time = 200
        self.sample_rate = 16000
        self.need_reply = False
        self.cache_clear_interval = 10000
        self.websocket = websocket

    def add_frame(self, frame):
        self.buffer.extend(frame)
        t1 = time.time()
        new_res = self.vad.detect(frame, cache=self.vad_cache, is_final=0, chunk_size=self.frame_time)
        use_time = time.time() - t1
        if len(new_res) > 0:
            print(self.speech_frames, new_res, len(self.buffer), self.status, use_time)
        self.speech_frames += 1
        self.vad_res.extend(new_res)

    def check_status(self):
        if len(self.vad_res) == 0: # 无有效音频
            if self.speech_frames * self.frame_time > self.cache_clear_interval:
                self.reset()
            return
        elif self.vad_res[-1][1] == -1: # 音频尚未结束
            self.status = 'listening'
            return
        elif self.speech_frames * self.frame_time - self.vad_res[-1][1] < self.silence_thr : # 音频间断 未到等待时间
            return
        else: # 音频结束
            self.status = 'replying'
            self.trunc_begin = int(self.vad_res[0][0] / self.frame_time) * self.frame_time
            self.trunc_end = int(self.vad_res[-1][1] / self.frame_time) * self.frame_time + self.frame_time - 1
            self.need_reply = True

    def get_useful_audio(self):
        return self.buffer[int(self.trunc_begin * self.sample_rate / 1000) * 2 :
                           int(self.trunc_end * self.sample_rate / 1000) * 2]

    async def save_recording(self, useful_audio, client_id):
        filename = f"user_{client_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
        with wave.open(os.path.join('.temp_data',filename), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(useful_audio)
        print(f"Saved {filename}")

    def reset(self):
        self.buffer = bytearray()
        self.vad_res = []
        self.speech_frames = 0
        self.vad_cache = {}
        self.status = 'idle'
        self.trunc_begin = 0
        self.trunc_end = 0
        self.need_reply = False # 重置后不需要回复

    async def proc(self, frame, client_id, **kwargs):
        self.add_frame(frame)
        self.check_status()
        await self.websocket.send(json.dumps({
            "type": "status",
            "status": self.status
        }))
        if self.status == 'replying':
            print(self.status)
        await asyncio.sleep(0)
        if self.need_reply:
            useful_audio = self.get_useful_audio()
            text = self.asr.recognize(useful_audio)
            if text in ('嗯。','。') or len(text)<=2:
                self.reset()
                return
            print(text)
            asyncio.create_task(
                self.websocket.send(json.dumps({"type": "text_user", "text": text})))
            asyncio.create_task(
                self.save_recording(useful_audio, client_id))
            await asyncio.sleep(0)
            self.reset()
            reply_text = self.llm.generate_response(text)
            print(reply_text)
            asyncio.create_task(
                self.websocket.send(json.dumps({ "type": "text_robot", "text": reply_text})))
            await asyncio.sleep(0)
            reply_audio = self.tts.generate_audio(reply_text,
                file_path=f"./.temp_data/robot_{client_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.{self.tts.output_format}")
            asyncio.create_task(
                self.websocket.send(json.dumps({
                    "type": "audio",
                    "audio": base64.b64encode(reply_audio).decode('utf-8'),
                    "format": "mp3",  # 或实际使用的音频格式
                    "sampleRate": self.sample_rate
                })))


async def server(websocket):
    client_id = str(id(websocket))
    print(f"Client {client_id} connected")
    llm_model = LLMModel(cfg.llm)
    print("LLM model loaded")
    buffer = AudioBuffer(client_id,llm=llm_model,websocket=websocket)
    async for message in websocket:
        if isinstance(message, bytes):
            await buffer.proc(message, client_id)
        elif isinstance(message, str):
            # 处理文本消息（如果有）
            data = json.loads(message)
            if data.get("type") == "user_info":
                buffer.llm.set_user_name(data.get("userName"))
                print(f"User info received: {data}")
            pass


async def main():
    async with websockets.serve(server, "0.0.0.0", 8765):
        await asyncio.Future()  # 永久运行


if __name__ == "__main__":
    asyncio.run(main())

