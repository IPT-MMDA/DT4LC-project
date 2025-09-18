import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


def test() -> None:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    model = "gemini-2.5-flash"

    # Build a simple chat history
    history = [
        types.Content(role="user", parts=[types.Part(text="Hello! Act as a context understanding agent.")]),
        types.Content(role="model", parts=[types.Part(text="Sure—what data do you have?")]),
        types.Content(role="user", parts=[types.Part(text="I have a Sentinel-2 raster; propose tags + a pipeline.")]),
    ]

    # Stream the reply
    for chunk in client.models.generate_content_stream(model=model, contents=history):
        if txt := chunk.text:
            print(txt, end="", flush=True)


if __name__ == "__main__":
    test()
