import os

from dotenv import load_dotenv
from ollama import Client

load_dotenv()

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "gemma4:12b"
)

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://localhost:11434"
)

client = Client(
    host=OLLAMA_HOST
)


def query_model(prompt: str) -> str:
    """Send a prompt to the configured Ollama model."""

    response = client.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response["message"]["content"]


def main():

    response = query_model(
        "Why is the sky blue?"
    )

    print(response)


if __name__ == "__main__":
    main()