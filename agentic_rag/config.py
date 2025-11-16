import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_CHAT_MODEL = os.getenv("DEFAULT_CHAT_MODEL", "gpt-4.1")
DEFAULT_TEMPERATURE = int(os.getenv("DEFAULT_TEMPERATURE", 0))
