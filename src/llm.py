import json

import requests


class LLM:
    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/chat"

    def generate(self, prompt: str):
        try:
            headers = {"Content-Type": "application/json"}
            data = {
                "model": "qwen2.5:7b",
                "messages": [{"role": "user", "content": prompt + ", 一句话回答"}],
                "stream": True,
            }

            response = requests.post(
                self.ollama_url, headers=headers, json=data, stream=True
            )

            if response.status_code == 200:
                # Process the streamed response from Ollama
                for chunk in response.iter_lines(decode_unicode=True):
                    if chunk:
                        try:
                            # Parse the chunk as JSON
                            chunk_data = json.loads(chunk)
                            sentence = chunk_data.get("message", {}).get("content")
                            if sentence:
                                yield sentence
                        except Exception as e:
                            print(f"Error parsing chunk: {e}")
            else:
                return f"Error: {response.status_code}, {response.text}"

        except Exception as e:
            return f"Error contacting Ollama server: {e}"


if __name__ == "__main__":
    llm = LLM()
    for text in llm.generate("模仿丁真谈谈天气"):
        print(text, end=" ")
