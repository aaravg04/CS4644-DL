from ollama import chat
from ollama import ChatResponse


def get_response(prompt: str):
    response: ChatResponse = chat(model='llama3.2', messages=[
    {
        'role': 'user',
        'content': prompt,
    },
    ])

    return response.message.content
