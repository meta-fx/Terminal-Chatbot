# Terminal Chatbot

This project is an simple AI chatbot that can communicate using various Language Models (LLMs) and optionally convert responses to speech.

## Features

- Chat with different AI models from providers like OpenAI and TogetherAI
- Optional Text-to-Speech (TTS) for AI responses
- Customizable conversation settings (e.g., temperature, max tokens)
- Support for streaming responses
- Easy-to-use command-line interface

## Prerequisites

- Python 3.7 or higher
- API keys for the services you want to use (OpenAI, TogetherAI, ElevenLabs, etc.)

## Installation

1. Clone this repository:

   ```
   git clone https://github.com/yourusername/ai-chatbot-tts.git
   cd ai-chatbot-tts
   ```

2. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

3. Set up your API keys as environment variables:
   - For OpenAI: `export OPENAI_API_KEY="your-api-key-here"`
   - For TogetherAI: `export TOGETHER_API_KEY="your-api-key-here"`
   - For ElevenLabs TTS: `export ELEVEN_API_KEY="your-api-key-here"`

## Usage

Run the chatbot using the following command:

```
python main.py [options]
```

Options:

- `--provider`: Choose the AI provider (default: together)
- `--model`: Specify the AI model to use
- `--temperature`: Set the response randomness (0.0 to 2.0, default: 0.7)
- `--max-tokens`: Set the maximum response length
- `--stream`: Enable response streaming
- `--tts`: Enable Text-to-Speech (choices: elevenlabs, streamelements, polly)
- `--voice`: Specify the TTS voice to use

## Configuration

You can customize the chatbot's initial behavior by editing the `system_prompt.txt` file. This sets the context for the AI at the start of each conversation.

## Examples

1. Basic chat with TogetherAI:

   ```
   python main.py
   ```

2. Chat with OpenAI's GPT-4 and Text-to-Speech:

   ```
   python main.py --provider openai --model gpt-4o-mini --tts elevenlabs --voice Sarah
   ```

3. Streaming chat with custom settings:
   ```
   python main.py --provider together --model meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo --temperature 0.9 --max-tokens 150 --stream
   ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For updates and more information, follow us on Twitter: [@pogtx\_](https://x.com/pogtx_)

Visit our project repository: [Terminal Chatbot on GitHub](https://github.com/meta-fx/Terminal-Chatbot)
