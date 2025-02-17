from funasr import AutoModel

class VADModel:
    def __init__(self, vad_config):
        if vad_config.model_name == "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch":
            self.model = AutoModel(
                model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                model_revision="v2.0.4",
                max_end_silence_time=440,
                speech_noise_thres=0.8,
                disable_update=True,
                disable_pbar=True,
                #device="cuda",
            )

    def detect(self, speech_chunk, cache={}, is_final=0, chunk_size=200):
        res = self.model.generate(input=speech_chunk, cache=cache, is_final=is_final, chunk_size=chunk_size)
        if len(res[0]["value"]):
            #print(res)
            return res[0]["value"]
        elif res[0]["value"] is not None:
            return res[0]["value"]


# if __name__ == '__main__':
#     import config
#     cfg = config.Config()
#     model = VADModel(cfg.vad)
#     import soundfile
#
#     wav_file = f"{model.model.model_path}/example/vad_example.wav"
#     speech, sample_rate = soundfile.read(wav_file)
#     chunk_size=200
#     chunk_stride = int(chunk_size * sample_rate / 1000)
#
#     cache = {}
#     total_chunk_num = int(len((speech) - 1) / chunk_stride + 1)
#     for i in range(total_chunk_num):
#         speech_chunk = speech[i * chunk_stride:(i + 1) * chunk_stride]
#         is_final = i == total_chunk_num - 1
#         #print(cache)
#         model.detect(speech_chunk=speech_chunk, cache=cache, is_final=is_final, chunk_size=chunk_size)

