# scripts/tts.py

from abc import ABC, abstractmethod
from typing import Optional
import requests
from elevenlabs import stream
from elevenlabs.client import ElevenLabs
import boto3
import os
import sys
import io
from botocore.exceptions import BotoCoreError, ClientError

original_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')

import pygame

sys.stdout.close()
sys.stdout = original_stdout


pygame.mixer.init()

class TTSProvider(ABC):
    @abstractmethod
    def synthesize(self, text: str, voice: str) -> bool:
        pass

    @staticmethod
    def _play_audio(audio_stream: io.BytesIO) -> bool:
        try:
            pygame.mixer.music.load(audio_stream)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            return True
        except pygame.error as e:
            print(f"Failed to play audio: {e}")
            return False

class ElevenLabsProvider(TTSProvider):
    def __init__(self, api_key: str):
        self.client = ElevenLabs(api_key=api_key)

    def synthesize(self, text: str, voice: str) -> bool:
        try:
            audio_stream = self.client.generate(
                text=text,
                voice=voice,
                model="eleven_multilingual_v2",
                stream=True
            )
            stream(audio_stream)
            return True
        except Exception as e:
            print(f"ElevenLabs synthesis failed: {e}")
            return False

class StreamElementsProvider(TTSProvider):
    def synthesize(self, text: str, voice: str) -> bool:
        tts_url = f"https://api.streamelements.com/kappa/v2/speech?voice={voice}&text={text}"
        try:
            response = requests.get(tts_url)
            response.raise_for_status()
            audio_stream = io.BytesIO(response.content)
            return self._play_audio(audio_stream)
        except requests.RequestException as e:
            print(f"StreamElements request failed: {e}")
            return False

class AmazonPollyProvider(TTSProvider):
    def __init__(self, region_name: str, engine: str = 'neural'):
        self.polly_client = boto3.client('polly', region_name=region_name)
        self.engine = engine

    def synthesize(self, text: str, voice: str) -> bool:
        try:
            response = self.polly_client.synthesize_speech(
                Text=text,
                OutputFormat='mp3',  # Changed from 'ogg_vorbis' to 'mp3'
                VoiceId=voice,
                Engine=self.engine
            )
            
            if "AudioStream" in response:
                audio_stream = response["AudioStream"]
                audio_data = audio_stream.read()
                audio_io = io.BytesIO(audio_data)
                return self._play_audio(audio_io)
            else:
                print("Could not stream audio")
                return False
            
        except (BotoCoreError, ClientError) as error:
            if isinstance(error, ClientError) and error.response['Error']['Code'] == 'ValidationException' and "engine is not supported in this region" in str(error):
                print(f"Neural engine not supported. Falling back to standard engine.")
                self.engine = 'standard'
                return self.synthesize(text, voice)
            else:
                print(f"Polly synthesis failed: {error}")
                return False

    @staticmethod
    def _play_audio(audio_stream: io.BytesIO) -> bool:
        try:
            pygame.mixer.music.load(audio_stream)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            return True
        except pygame.error as e:
            print(f"Failed to play audio: {e}")
            return False

class TTSFactory:
    _instances = {}

    @classmethod
    def create_provider(cls, provider_type: str, config: dict) -> Optional[TTSProvider]:
        if provider_type not in cls._instances:
            try:
                if provider_type == 'elevenlabs':
                    cls._instances[provider_type] = ElevenLabsProvider(config['api_key'])
                elif provider_type == 'streamelements':
                    cls._instances[provider_type] = StreamElementsProvider()
                elif provider_type == 'polly':
                    cls._instances[provider_type] = AmazonPollyProvider(config['region_name'], config.get('engine', 'neural'))
                else:
                    print(f"Unsupported TTS provider: {provider_type}")
                    return None
            except Exception as e:
                print(f"Failed to create TTS provider: {e}")
                return None
        return cls._instances[provider_type]

def tts(text: str, voice: str, provider_type: str, config: dict) -> bool:
    provider = TTSFactory.create_provider(provider_type, config)
    if provider:
        return provider.synthesize(text, voice)
    else:
        return False