from openai import OpenAI
from typing import List, Dict

class GPTAPI:
    def __init__(self):
        self.client = client = OpenAI(
            api_key='sk-KBZNJaExkRwjzmxrKQZramsiZVHauPCubjk0ZGX26wsUfgyA',
            base_url="https://api.chatanywhere.tech/v1"
        )

    def call_gpt(
        self,
        messages: List[Dict[str, str]],
        model_name: str = 'gpt-3.5-turbo'
    ):
        response = self.client.chat.completions.create(
            model=model_name,
            messages=messages
        )

        text = response.choices[0].message.content
        return text