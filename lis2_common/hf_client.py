import requests
import os
from typing import Optional

class HFClient:
    """
    A client for interacting with the LIS2Server API.
    
    Usage:
        client = HFClient("google/gemma-3-27b-it")
        response = client.generate("Hello, how are you?")
    """
    def __init__(self, model_name: str, api_url: str = "http://localhost:8080/api/v1", api_token: Optional[str] = None):
        self.model_name = model_name
        self.api_url = api_url.rstrip("/")
        self.api_token = api_token or os.getenv("LIS2_API_TOKEN")
        
        # Trigger model loading upon initialization
        self._load_model()

    def _get_headers(self):
        headers = {}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        return headers

    def _load_model(self):
        """Triggers the server to load the model."""
        url = f"{self.api_url}/load"
        payload = {"model": self.model_name}
        try:
            response = requests.post(url, json=payload, headers=self._get_headers())
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            msg = f"Warning: Failed to pre-load model {self.model_name}: {e}"
            if e.response is not None:
                msg += f"\nServer details: {e.response.text}"
            print(msg)

    def generate(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.7, top_p: float = 1.0) -> str:
        """Generates text using the model."""
        url = f"{self.api_url}/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        
        response = requests.post(url, json=payload, headers=self._get_headers())
        response.raise_for_status()
        
        data = response.json()
        return data.get("response", "")
