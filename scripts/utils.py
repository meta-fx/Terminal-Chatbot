# scripts/utils.py

import os
import sys
from together import Together

VALID_MODELS = {
    "openai": [
        "gpt-4o-mini",
        "gpt-4o-2024-08-06",
    ]
}


def load_system_prompt(file_path="system_prompt.txt"):
    _, file_extension = os.path.splitext(file_path)

    try:
        with open(file_path, 'r') as file:
            if file_extension == '.txt':
                return file.read().strip()
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
    except FileNotFoundError:
        print(f"Error: System prompt file not found: {file_path}")
        sys.exit(1)
    except IOError:
        print(f"Error: Unable to read system prompt file: {file_path}")
        sys.exit(1)


def get_api_key(provider):
    if provider == "together":
        api_key = os.environ.get("TOGETHER_API_KEY")
        env_var = "TOGETHER_API_KEY"
    elif provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        env_var = "OPENAI_API_KEY"
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    if not api_key:
        print(f"Error: {env_var} environment variable is not set.")
        print(f"Please set the {env_var} environment variable and try again.")
        sys.exit(1)

    return api_key


def get_together_chat_models():
    api_key = get_api_key("together")
    client = Together(api_key=api_key)
    models = client.models.list()
    return [model.id for model in models if model.type == "chat"]


def validate_model(provider, model):
    if provider == "together":
        valid_models = get_together_chat_models()
    else:
        valid_models = VALID_MODELS[provider]

    if model not in valid_models:
        print(f"Error: Invalid model '{model}' for provider '{provider}'.")
        print(f"Valid models are: {', '.join(valid_models)}")
        sys.exit(1)
