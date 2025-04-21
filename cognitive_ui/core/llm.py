"""
Large Language Model (LLM) integration for the Cognitive Digital Twin Framework
"""

from typing import Any, cast
import os
import json
import requests
from dotenv import load_dotenv

from cognitive_ui.config import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GEMINI_MAX_TOKENS,
    MAX_TOKENS,
)

# Load environment variables from .env file
load_dotenv()

# Retrieve an API key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
# Select gemini specific limit if set, otherwise use the global limit
GEMINI_MAX_TOKENS = DEFAULT_GEMINI_MAX_TOKENS or MAX_TOKENS


def query_gemini(prompt: str, max_tokens: int = GEMINI_MAX_TOKENS) -> str | None:
    """Query the Google Gemini API with a prompt"""
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY environment variable not set.")
        return None

    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}
    request_data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.7},
    }

    try:
        response = requests.post(api_url, headers=headers, json=request_data, params=params)
        response.raise_for_status()
        result = response.json()
        return cast(str, result["candidates"][0]["content"]["parts"][0]["text"])

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        if response is not None:
            print(f"Response status: {response.status_code}")
            print(f"Response content: {response.text}")
        return None

    except Exception as e:
        print("Error calling Gemini API:", e)
        return None


def construct_prompt(template: str, context_data: Any) -> str:
    """
    Construct a prompt for the LLM with context data

    Args:
        template: The prompt template string
        context_data: Data to include in the prompt context

    Returns:
        Formatted prompt string
    """
    context_text = json.dumps(context_data, indent=2) if context_data else ""
    return template.format(context=context_text)
