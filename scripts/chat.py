# scripts/chat.py

import os
import sys
from scripts.tts import TTSFactory
from scripts.utils import load_system_prompt

# ANSI codes
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


def chat(args, llm):
    system_prompt = None
    if args.system_prompt_file:
        system_prompt = load_system_prompt(args.system_prompt_file)

    tts_provider = None
    if args.tts:
        tts_config = {
            "api_key": os.environ.get("ELEVEN_API_KEY_TEST"),
            "region_name": "us-west-2",  # Default region for Amazon Polly
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

    print(f"\nChat started with {llm.model}\n")

    while True:
        user_input = input("")

        if user_input.lower() == 'bye':
            break

        conversation_history.append({"role": "user", "content": user_input})

        print(f"\n{GREEN}", end="", flush=True)
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

        # TTS synthesis
        if tts_provider:
            tts_provider.synthesize(full_response, args.tts_voice)

    print("Chat ended. Final statistics:")
    print(f"Total Token Usage: {llm.get_token_usage()}")
    print(f"Total Cost: ${llm.get_cost():.6f}")
