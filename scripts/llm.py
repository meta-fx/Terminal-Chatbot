# scripts/llm.py

import abc
from openai import OpenAI
from together import Together
from typing import Dict, List, Generator, Tuple


class BaseLLM(abc.ABC):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.token_usage = {"prompt_tokens": 0,
                            "completion_tokens": 0, "total_tokens": 0}
        self.cost = 0.0

    @abc.abstractmethod
    def chat_completion(self, messages: List[Dict[str, str]]) -> str:
        pass

    @abc.abstractmethod
    def chat_completion_stream(self, messages: List[Dict[str, str]]) -> Generator[Tuple[str, Dict[str, int]], None, None]:
        pass

    @abc.abstractmethod
    def get_token_usage(self) -> Dict[str, int]:
        pass

    @abc.abstractmethod
    def get_cost(self) -> float:
        pass

    def update_token_usage(self, new_usage: Dict[str, int]):
        for key in self.token_usage:
            self.token_usage[key] += new_usage.get(key, 0)


class OpenAILLM(BaseLLM):
    def __init__(self, api_key: str, model: str, max_tokens: int, temperature: float):
        super().__init__(api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=self.api_key)
        self.cost_per_1M_tokens = {
            "gpt-4o-2024-08-06": (2.5, 10),
            "gpt-4o-mini": (0.15, 0.6),
        }

    def chat_completion(self, messages: List[Dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        if response.usage:
            self.update_token_usage(response.usage.model_dump())
            self.calculate_cost()
        return response.choices[0].message.content

    def chat_completion_stream(self, messages: List[Dict[str, str]]) -> Generator[Tuple[str, Dict[str, int]], None, None]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=True,
            stream_options={"include_usage": True}
        )
        full_content = ""
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_content += content
                yield content, {}

            if chunk.usage:
                usage = chunk.usage.model_dump()
                self.update_token_usage(usage)
                self.calculate_cost()
                yield "", usage

        # If we didn't get usage information in the stream, estimate it
        if not self.token_usage["completion_tokens"]:
            estimated_tokens = len(full_content.split())
            self.update_token_usage({"completion_tokens": estimated_tokens})
            self.calculate_cost()
            yield "", {"estimated_completion_tokens": estimated_tokens}

    def get_token_usage(self) -> Dict[str, int]:
        return self.token_usage

    def get_cost(self) -> float:
        return self.cost

    def calculate_cost(self):
        input_cost, output_cost = self.cost_per_1M_tokens.get(
            self.model, (0.0, 0.0))
        self.cost = (self.token_usage["prompt_tokens"] * input_cost / 1_000_000 +
                     self.token_usage["completion_tokens"] * output_cost / 1_000_000)


class TogetherAILLM(BaseLLM):
    def __init__(self, api_key: str, model: str,  max_tokens: int, temperature: float):
        super().__init__(api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = Together(api_key=self.api_key)
        self.pricing = self.get_model_price()

    def get_model_price(self):
        models = self.client.models.list()
        for model in models:
            if model.id == self.model:
                return model.pricing
        raise ValueError(
            f"Model {self.model} not found in Together AI's model list")

    def chat_completion(self, messages: List[Dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        self.update_token_usage(response.usage.model_dump())
        self.calculate_cost()
        return response.choices[0].message.content

    def chat_completion_stream(self, messages: List[Dict[str, str]]) -> Generator[Tuple[str, Dict[str, int]], None, None]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=True
        )
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content, {}

        # The last chunk contains the usage information
        if hasattr(chunk, 'usage'):
            usage = chunk.usage.model_dump()
            self.update_token_usage(usage)
            self.calculate_cost()
            yield "", usage

    def get_token_usage(self) -> Dict[str, int]:
        return self.token_usage

    def get_cost(self) -> float:
        return self.cost

    def calculate_cost(self):
        input_cost = self.pricing.input * \
            self.token_usage["prompt_tokens"] / 1_000_000
        output_cost = self.pricing.output * \
            self.token_usage["completion_tokens"] / 1_000_000
        self.cost = input_cost + output_cost + self.pricing.base
