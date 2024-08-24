# main.py

import sys
import argparse
from scripts.chat import chat
from scripts.llm import TogetherAILLM, OpenAILLM
from scripts.utils import get_api_key, validate_model

DEFAULT_MODELS = {
    "together": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "openai": "gpt-4o-mini"
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Configure and run an LLM (Language Model) with specified provider and parameters.",
        epilog="Example usage: python script_name.py --provider openai --model gpt-4o-mini --temperature 0.8 --max-tokens 150 --system-prompt-file path/to/prompt.json --stream --tts elevenlabs"
    )

    parser.add_argument("--provider", type=str, default="together", choices=["together", "openai"],
                        help="LLM provider (default: %(default)s)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (default: %(default)s, 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo' for Together, 'gpt-4o-mini' for OpenAI)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for text generation (default: %(default)s)")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Maximum number of tokens to generate (default: %(default)s)")
    parser.add_argument("--system-prompt-file", type=str, default=None,
                        help="Path to the file containing the system prompt (optional, supports .txt, .json, .yml, .yaml)")
    parser.add_argument("--stream", action="store_true",
                        help="Enable response streaming")
    parser.add_argument("--tts", type=str, choices=["elevenlabs", "streamelements", "polly"], default=None,
                        help="Enable Text-to-Speech with the specified provider")
    parser.add_argument("--tts-voice", type=str, default="en-US-Wavenet-D",
                        help="Voice to use for Text-to-Speech (default: %(default)s)")

    args = parser.parse_args()

    if args.model is None:
        args.model = DEFAULT_MODELS[args.provider]

    # Validate temperature
    if args.temperature < 0 or args.temperature > 2:
        parser.error("Temperature must be between 0 and 2.")

    return args


def main():
    args = parse_arguments()

    try:
        validate_model(args.provider, args.model)
        api_key = get_api_key(args.provider)

        if args.provider == "together":
            llm = TogetherAILLM(
                api_key=api_key,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature
            )
        elif args.provider == "openai":
            llm = OpenAILLM(
                api_key=api_key,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature
            )
        else:
            raise ValueError(f"Unsupported provider: {args.provider}")

        chat(args, llm)

    except ValueError as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
