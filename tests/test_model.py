import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env
load_dotenv()

# Read OPENAI_API_KEY from environment
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError(
        "Missing OPENAI_API_KEY. Please configure it in your .env file."
    )

# Create an OpenAI Chat model
model = ChatOpenAI(
    model="gpt-4.1-mini",  # Can be switched to gpt-4.1 / gpt-4o-mini, etc.
    api_key=api_key,
)

# Run a simple smoke test
response = model.invoke("Hello, who are you?")
print(response)
