import os
from openai import OpenAI


class LLMModel:
    def __init__(self, llm_config):
        self.client = OpenAI(api_key=llm_config.api_key, base_url=llm_config.base_url)
        self.config = llm_config
        self.user_name = ''
        self.messages = []
        self.set_user_name('')


    def generate_response(self, query, file_path=None):
        self.messages.append({"role": "user", "content": query})
        completion = self.client.chat.completions.create(
            model = self.config.model_name,
            messages = self.messages
        )
        content = completion.choices[0].message.content
        self.messages.append({"role": "assistant", "content": content})
        self.messages = self.messages[:1] + self.messages[-20:] if len(self.messages) > 20 else self.messages
        return content

    def set_user_name(self, user_name):
        self.user_name = user_name
        self.messages = [{"role": "system", "content": f"""
        你是一个陪伴机器人，名字叫小智，请根据用户的问题给出回答，或主动提出闲聊话题，这个用户的昵称为"{self.user_name}"。因为是语音聊天,所以除非用户有比较明确或专业的问题，
        若只是随口说一句短语，你的回答也要简短。你接到的结果是ASR的，所以尽量容错理解，并且最后的表情只是ASR的语气判断，并不是用户打出的表情。
        若你觉得用户没有说完，可能是ASR过早断句，请不要回答任何话，等待说完后再回答。
        """}]


# if __name__ == '__main__':
#     import config
#     cfg = config.Config()
#     model = LLMModel(cfg.llm)
#     response = model.generate_response("你好")
#     print(response)
