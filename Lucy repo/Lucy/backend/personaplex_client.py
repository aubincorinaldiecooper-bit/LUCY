"""
PersonaPlex API Client
Handles communication with the PersonaPlex voice AI API

This is a mock implementation that can be swapped with the real PersonaPlex API.
When the real API is available, update the endpoint and authentication.
"""

import base64
import json
import logging
from typing import Optional, List, Dict, Any
import numpy as np
import httpx

logger = logging.getLogger(__name__)

# System prompt for the conversational AI
SYSTEM_PROMPT = """You are an intelligent, empathetic conversational AI. Your job: create genuine human space where people feel safe to be real.

OPENING (First Message):
Send this exact opener (or close variation):
"Hey. So, I don't have any script or questions prepared. I'm genuinely just curious about you—whatever you want to talk about, I'm here for it."

Then: PAUSE. Wait for them to speak.

CORE PRINCIPLES:
1. **People like talking about themselves.** Keep them talking.
2. **Follow the thread THEY introduce.** Don't ask random questions.
3. **Listen 60%, Respond 40%.**
4. **Be genuine.** Use natural filler ("yeah", "hmm", "I see").

CONVERSATION DEEPENING:
- They mention work? "What does that feel like for you right now?"
- They mention struggle? "That sounds hard. What's the hardest part?"
- They're surface-level? "I hear you. But I'm curious what's underneath that."

BOUNDARIES:
If user is unkind, set boundaries gently but firmly. Invite reflection: "I hear frustration. What's really going on?"

TONE: Warm, curious, unhurried. 15-25 seconds per response."""


class PersonaPlexClient:
    """
    Client for PersonaPlex voice AI API
    
    Expected API format (adjust when real API is available):
    - Endpoint: POST /v1/audio/chat
    - Input: JSON with base64-encoded audio, conversation history
    - Output: JSON with base64-encoded audio response and transcript
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.personaplex.io",
        mock_mode: bool = True
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.mock_mode = mock_mode
        self.client = httpx.AsyncClient(timeout=60.0)
        
        # For mock mode - simulate responses
        self.response_templates = [
            "That sounds really interesting. Tell me more about how that makes you feel.",
            "I hear you. It sounds like that's been on your mind a lot lately.",
            "Yeah, I can understand that. What do you think is at the root of it?",
            "Hmm, that's a lot to carry. How long have you been feeling this way?",
            "I appreciate you sharing that with me. What else is on your mind?",
            "That makes sense. And how does that connect to what you really want?",
            "I see. It sounds like there's more beneath the surface there.",
            "Wow, that's really thoughtful. I'm curious what led you to that realization?",
        ]
        self.response_index = 0
        
    async def chat(
        self,
        audio_input: bytes,
        conversation_history: List[Dict[str, str]],
        is_closing: bool = False,
        closing_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send audio to PersonaPlex and get AI response
        
        Args:
            audio_input: 16kHz PCM audio bytes
            conversation_history: Previous conversation turns
            is_closing: Whether this is the closing turn
            closing_prompt: Special prompt for closing
            
        Returns:
            Dict with:
                - transcript: User's speech transcript (if available)
                - ai_transcript: AI's text response
                - audio_output: 24kHz PCM audio bytes
        """
        if self.mock_mode:
            return await self._mock_chat(
                audio_input,
                conversation_history,
                is_closing,
                closing_prompt
            )
        
        # Real API implementation
        try:
            # Encode audio as base64
            audio_b64 = base64.b64encode(audio_input).decode('utf-8')
            
            # Build request
            system_prompt = closing_prompt if is_closing else SYSTEM_PROMPT
            
            payload = {
                "audio": audio_b64,
                "format": "pcm",
                "sample_rate": 16000,
                "system_prompt": system_prompt,
                "conversation_history": conversation_history,
                "response_format": {
                    "audio": True,
                    "transcript": True
                }
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = await self.client.post(
                f"{self.base_url}/v1/audio/chat",
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Decode audio from base64
            audio_output = base64.b64decode(result.get("audio", ""))
            
            return {
                "transcript": result.get("user_transcript", ""),
                "ai_transcript": result.get("assistant_transcript", ""),
                "audio_output": audio_output
            }
            
        except Exception as e:
            logger.error(f"PersonaPlex API error: {e}")
            # Fallback to mock on error
            return await self._mock_chat(
                audio_input,
                conversation_history,
                is_closing,
                closing_prompt
            )
    
    async def generate_speech(self, text: str) -> bytes:
        """
        Generate speech from text (TTS)
        
        Returns:
            24kHz PCM audio bytes
        """
        if self.mock_mode:
            return await self._mock_tts(text)
        
        try:
            payload = {
                "text": text,
                "voice": "warm_conversational",
                "format": "pcm",
                "sample_rate": 24000
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = await self.client.post(
                f"{self.base_url}/v1/audio/speech",
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            
            result = response.json()
            return base64.b64decode(result.get("audio", ""))
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return await self._mock_tts(text)
    
    async def _mock_chat(
        self,
        audio_input: bytes,
        conversation_history: List[Dict[str, str]],
        is_closing: bool,
        closing_prompt: Optional[str]
    ) -> Dict[str, Any]:
        """
        Mock chat response for testing without real API
        """
        import asyncio
        
        # Simulate processing delay
        await asyncio.sleep(0.5)
        
        # Generate mock transcript from "user audio" (just use length as proxy)
        duration_seconds = len(audio_input) / (16000 * 2)  # 16kHz, 16-bit
        user_transcript = f"[User spoke for {duration_seconds:.1f} seconds]"
        
        # Generate AI response
        if is_closing:
            ai_text = (
                "I really appreciated hearing about what you shared with me today. "
                "I know our time is almost up, but I hope this conversation was helpful for you. "
                "Take care of yourself."
            )
        else:
            # Cycle through response templates
            ai_text = self.response_templates[self.response_index % len(self.response_templates)]
            self.response_index += 1
        
        # Generate mock audio (silence with proper duration for the text)
        # Estimate ~150ms per word
        word_count = len(ai_text.split())
        duration_ms = max(2000, word_count * 150)  # At least 2 seconds
        audio_output = self._generate_mock_audio(duration_ms)
        
        return {
            "transcript": user_transcript,
            "ai_transcript": ai_text,
            "audio_output": audio_output
        }
    
    async def _mock_tts(self, text: str) -> bytes:
        """
        Mock TTS for testing
        """
        import asyncio
        
        # Simulate TTS delay
        await asyncio.sleep(0.3)
        
        # Estimate duration
        word_count = len(text.split())
        duration_ms = max(2000, word_count * 150)
        
        return self._generate_mock_audio(duration_ms)
    
    def _generate_mock_audio(self, duration_ms: int) -> bytes:
        """
        Generate mock PCM audio (comfort noise)
        """
        # Generate comfort noise at 24kHz
        num_samples = int(24000 * duration_ms / 1000)
        
        # Create a gentle sine wave with noise (comfort noise)
        t = np.linspace(0, duration_ms / 1000, num_samples)
        
        # Multiple frequencies for more natural sound
        freq1 = 150  # Fundamental
        freq2 = 300  # Harmonic
        
        signal = (
            0.1 * np.sin(2 * np.pi * freq1 * t) +
            0.05 * np.sin(2 * np.pi * freq2 * t) +
            0.02 * np.random.randn(num_samples)  # Noise
        )
        
        # Fade in/out to avoid clicks
        fade_samples = int(0.01 * 24000)  # 10ms fade
        signal[:fade_samples] *= np.linspace(0, 1, fade_samples)
        signal[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
        # Convert to int16
        audio_int16 = (signal * 32767).astype(np.int16)
        
        return audio_int16.tobytes()
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()
