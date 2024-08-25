# scripts/chat.py

import os
import sys
from scripts.tts import TTSFactory
from scripts.utils import load_system_prompt

BLACK = "\033[30m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"
RESET = "\033[0m"

BG_BLACK = "\033[40m"
BG_RED = "\033[41m"
BG_GREEN = "\033[42m"
BG_YELLOW = "\033[43m"
BG_BLUE = "\033[44m"
BG_MAGENTA = "\033[45m"
BG_CYAN = "\033[46m"
BG_WHITE = "\033[47m"

BOLD = "\033[1m"
DIM = "\033[2m"
ITALIC = "\033[3m"
UNDERLINE = "\033[4m"
INVERSE = "\033[7m"
HIDDEN = "\033[8m"
STRIKETHROUGH = "\033[9m"


def print_start_message(llm, args):
    print("")
    print(f"{GREEN}Chat Started")
    print(f"Provider: {args.provider}")
    print(f"Model: {llm.model}")
    print(f"Temperature: {llm.temperature}")
    print(f"Max Tokens: {llm.max_tokens}{RESET}")
    print("")


def print_end_message(token_usage, cost):
    print("")
    print(f"{RED}Chat Ended")
    print(f"Cost: ${cost:.6f}")
    print(f"Token Usage")
    print(f"Prompt: {token_usage['prompt_tokens']}")
    print(f"Completion: {token_usage['completion_tokens']}")
    print(f"Total: {token_usage['total_tokens']}{RESET}")


def chat(args, llm):
    system_prompt = load_system_prompt()

    if not system_prompt or system_prompt == "":
        system_prompt = None

    tts_provider = None
    if args.tts:
        tts_config = {
            "api_key": os.environ.get("ELEVEN_API_KEY"),
            "region_name": "us-west-2",
            "engine": "neural"
        }
        tts_provider = TTSFactory.create_provider(args.tts, tts_config)

    if not tts_provider and args.tts:
        print(f"Failed to initialize TTS provider: {args.tts}")
        sys.exit(1)

    conversation_history = []
    if system_prompt:
        conversation_history.append(
            {"role": "system", "content": system_prompt})

    print_start_message(llm, args)

    while True:
        user_input = input(f"")

        if user_input.lower() == 'bye':
            break

        conversation_history.append({"role": "user", "content": user_input})

        print(f"\n{WHITE}", end="", flush=True)
        if args.stream:
            full_response = ""
            for content, usage in llm.chat_completion_stream(conversation_history):
                if content:
                    print(content, end="", flush=True)
                    full_response += content
                if usage:
                    print(f"{RESET}")  # New line after streaming is complete
        else:
            full_response = llm.chat_completion(conversation_history)
            print(full_response)

        conversation_history.append(
            {"role": "assistant", "content": full_response})
        print(f"{RESET}")  # Extra newline for readability

        if tts_provider:
            tts_provider.synthesize(full_response, args.voice)

    print_end_message(llm.get_token_usage(), llm.get_cost())
