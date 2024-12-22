import requests

class OllamaClient:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def chat(self, model: str, messages: list[dict], temperature: float = 0.7, max_tokens: int = 1024):
        """
        Sends a chat request to the local Ollama server.
        """
        payload = {
            "model": model,
            "messages": messages,
            # "temperature": temperature,
            # "max_tokens": max_tokens,
            "stream": False,
        }
        print(payload)
        response = requests.post(f"{self.base_url}/api/chat", json=payload)

        # Log raw response content
        print("Raw response content:", response.content.decode("utf-8"))

        response.raise_for_status()

        # Attempt to parse the JSON response
        try:
            return response.json()
        except ValueError as e:
            print(f"Error parsing JSON: {e}")
            raise
