"""
PersonaPlex Voice AI Backend
Real-time conversational AI with WebSocket audio streaming
"""

import asyncio
import base64
import json
import logging
import time
import io
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from enum import Enum

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from audio_processor import AudioProcessor, VADState
from personaplex_client import PersonaPlexClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
SESSION_DURATION = 15 * 60  # 15 minutes in seconds
CLOSING_THRESHOLD = 14 * 60  # Start closing at 14 minutes
SILENCE_THRESHOLD_MS = 400  # 400ms silence for VAD
SAMPLE_RATE_INPUT = 16000   # 16kHz for input
SAMPLE_RATE_OUTPUT = 24000  # 24kHz for output

app = FastAPI(
    title="PersonaPlex Voice AI",
    description="Real-time conversational AI with genuine human connection",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConversationState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    CLOSING = "closing"
    ENDED = "ended"


class ConversationSession:
    """Manages a single conversation session with timer and state"""
    
    def __init__(self, websocket: WebSocket, api_key: Optional[str] = None):
        self.websocket = websocket
        self.start_time = datetime.now()
        self.state = ConversationState.IDLE
        self.audio_processor = AudioProcessor(
            silence_threshold_ms=SILENCE_THRESHOLD_MS,
            sample_rate=SAMPLE_RATE_INPUT
        )
        self.personaplex = PersonaPlexClient(api_key=api_key)
        self.is_speaking = False
        self.current_ai_audio_task: Optional[asyncio.Task] = None
        self.conversation_history: list = []
        self.closing_initiated = False
        
    @property
    def elapsed_seconds(self) -> float:
        return (datetime.now() - self.start_time).total_seconds()
    
    @property
    def remaining_seconds(self) -> float:
        return max(0, SESSION_DURATION - self.elapsed_seconds)
    
    @property
    def should_close(self) -> bool:
        return self.elapsed_seconds >= CLOSING_THRESHOLD and not self.closing_initiated
    
    def get_closing_prompt(self) -> str:
        """Get the closing system prompt injection"""
        return """
Current Status: The conversation is ending (time limit reached).
Goal: Wrap up gracefully and leave a positive impact.

Your Final Response:
1. Briefly acknowledge what they just said (validate it one last time).
2. Transition to closing: "I know our time is almost up..."
3. Offer a final thoughtful sentiment: "I really appreciated hearing about this. I hope talking through this was helpful for you today."
4. Say goodbye warmly: "Take care of yourself."
"""


# Store active sessions
active_sessions: Dict[str, ConversationSession] = {}


@app.get("/")
async def root():
    return {
        "service": "PersonaPlex Voice AI",
        "status": "running",
        "websocket_endpoint": "/ws/chat",
        "session_duration": f"{SESSION_DURATION // 60} minutes"
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "active_sessions": len(active_sessions),
        "timestamp": datetime.now().isoformat()
    }


@app.websocket("/ws/chat")
async def websocket_chat(
    websocket: WebSocket,
    api_key: Optional[str] = Query(None, description="Optional PersonaPlex API key")
):
    """
    WebSocket endpoint for real-time voice conversation
    
    Message Types FROM Client:
    - Binary: PCM audio chunks (16kHz, 16-bit, mono)
    - JSON: {"type": "interrupt"} - Stop AI speaking
    - JSON: {"type": "start"} - Begin conversation (triggers AI opener)
    
    Message Types TO Client:
    - Binary: PCM audio chunks (24kHz, 16-bit, mono)
    - JSON: {"type": "state", "state": "listening|thinking|speaking"}
    - JSON: {"type": "transcript", "role": "user|assistant", "text": "..."}
    - JSON: {"type": "time_update", "remaining": seconds}
    - JSON: {"type": "ended", "message": "Conversation complete"}
    """
    await websocket.accept()
    session_id = f"session_{id(websocket)}"
    
    try:
        session = ConversationSession(websocket, api_key)
        active_sessions[session_id] = session
        
        logger.info(f"New session started: {session_id}")
        
        # Send initial ready state
        await send_state(websocket, "idle")
        
        # Handle messages
        while session.remaining_seconds > 0:
            try:
                # Check if we should initiate closing
                if session.should_close:
                    session.closing_initiated = True
                    session.state = ConversationState.CLOSING
                
                # Receive message with timeout to allow timer checks
                message = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=1.0
                )
                
                if "bytes" in message:
                    # Audio data received
                    await handle_audio_chunk(session, message["bytes"])
                    
                elif "text" in message:
                    # Control message received
                    data = json.loads(message["text"])
                    await handle_control_message(session, data)
                    
            except asyncio.TimeoutError:
                # Send time update periodically
                await send_time_update(session)
                continue
                
    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {session_id}")
    except Exception as e:
        logger.error(f"Error in session {session_id}: {e}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except:
            pass
    finally:
        # Cleanup
        if session_id in active_sessions:
            session = active_sessions[session_id]
            if session.current_ai_audio_task:
                session.current_ai_audio_task.cancel()
            del active_sessions[session_id]
        logger.info(f"Session ended: {session_id}")


async def handle_audio_chunk(session: ConversationSession, audio_bytes: bytes):
    """Process incoming audio chunk from client"""
    
    # Convert bytes to numpy array (16-bit PCM)
    audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
    
    # If we're in closing or ended state, ignore audio
    if session.state in [ConversationState.CLOSING, ConversationState.ENDED]:
        return
    
    # Check for interruption if AI is speaking
    if session.state == ConversationState.SPEAKING:
        # User is speaking over AI - interrupt
        if session.current_ai_audio_task:
            session.current_ai_audio_task.cancel()
            session.current_ai_audio_task = None
        session.is_speaking = False
        session.state = ConversationState.LISTENING
        await send_state(session.websocket, "listening")
    
    # Process audio through VAD
    vad_result = session.audio_processor.process_chunk(audio_data)
    
    if vad_result.state == VADState.SPEECH:
        session.is_speaking = True
        if session.state != ConversationState.LISTENING:
            session.state = ConversationState.LISTENING
            await send_state(session.websocket, "listening")
            
    elif vad_result.state == VADState.SILENCE and vad_result.is_final:
        # User stopped speaking - process the utterance
        await handle_user_turn_complete(session)


async def handle_user_turn_complete(session: ConversationSession):
    """Handle when user finishes speaking"""
    
    # Get accumulated audio
    user_audio = session.audio_processor.get_accumulated_audio()
    if len(user_audio) < SAMPLE_RATE_INPUT * 0.5:  # Less than 500ms
        return  # Too short, probably noise
    
    # Clear accumulator for next turn
    session.audio_processor.clear_accumulated()
    
    # Convert to format for API
    audio_bytes = user_audio.tobytes()
    
    # Update state to thinking
    session.state = ConversationState.THINKING
    await send_state(session.websocket, "thinking")
    
    try:
        # Determine if this is the closing turn
        is_closing = session.state == ConversationState.CLOSING
        
        # Call PersonaPlex API
        response = await session.personaplex.chat(
            audio_input=audio_bytes,
            conversation_history=session.conversation_history,
            is_closing=is_closing,
            closing_prompt=session.get_closing_prompt() if is_closing else None
        )
        
        # Update conversation history
        if response.get("transcript"):
            session.conversation_history.append({
                "role": "user",
                "content": response["transcript"]
            })
            # Send user transcript
            await send_transcript(session.websocket, "user", response["transcript"])
        
        if response.get("ai_transcript"):
            session.conversation_history.append({
                "role": "assistant", 
                "content": response["ai_transcript"]
            })
            # Send AI transcript
            await send_transcript(session.websocket, "assistant", response["ai_transcript"])
        
        # Play AI audio response
        if response.get("audio_output"):
            await play_ai_response(session, response["audio_output"])
        
        # If closing, end the conversation
        if is_closing:
            session.state = ConversationState.ENDED
            await session.websocket.send_json({
                "type": "ended",
                "message": "Conversation Complete. Thanks for chatting."
            })
            
    except Exception as e:
        logger.error(f"Error processing turn: {e}")
        session.state = ConversationState.LISTENING
        await send_state(session.websocket, "listening")


async def play_ai_response(session: ConversationSession, audio_bytes: bytes):
    """Stream AI audio response to client"""
    
    session.state = ConversationState.SPEAKING
    await send_state(session.websocket, "speaking")
    
    # Create task for streaming audio
    async def stream_audio():
        try:
            # Convert 24kHz PCM to chunks and send
            chunk_size = SAMPLE_RATE_OUTPUT * 2 * 20  # 20ms chunks
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i:i + chunk_size]
                await session.websocket.send_bytes(chunk)
                await asyncio.sleep(0.02)  # 20ms pacing
        except asyncio.CancelledError:
            # Interrupted by user
            pass
        except Exception as e:
            logger.error(f"Error streaming audio: {e}")
    
    session.current_ai_audio_task = asyncio.create_task(stream_audio())
    try:
        await session.current_ai_audio_task
    except asyncio.CancelledError:
        pass
    finally:
        session.current_ai_audio_task = None
        if session.state != ConversationState.ENDED:
            session.state = ConversationState.LISTENING
            await send_state(session.websocket, "listening")


async def handle_control_message(session: ConversationSession, data: dict):
    """Handle control messages from client"""
    msg_type = data.get("type")
    
    if msg_type == "start":
        # Send AI opener
        await send_ai_opener(session)
    elif msg_type == "interrupt":
        # Stop AI speaking
        if session.current_ai_audio_task:
            session.current_ai_audio_task.cancel()
            session.current_ai_audio_task = None
        session.state = ConversationState.LISTENING
        await send_state(session.websocket, "listening")


async def send_ai_opener(session: ConversationSession):
    """Send the AI's opening message"""
    opener_text = "Hey. So, I don't have any script or questions prepared. I'm genuinely just curious about you—whatever you want to talk about, I'm here for it."
    
    # Add to history
    session.conversation_history.append({
        "role": "assistant",
        "content": opener_text
    })
    
    # Send transcript
    await send_transcript(session.websocket, "assistant", opener_text)
    
    try:
        # Generate audio for opener
        audio = await session.personaplex.generate_speech(opener_text)
        if audio:
            await play_ai_response(session, audio)
    except Exception as e:
        logger.error(f"Error generating opener audio: {e}")
        session.state = ConversationState.LISTENING
        await send_state(session.websocket, "listening")


async def send_state(websocket: WebSocket, state: str):
    """Send state update to client"""
    try:
        await websocket.send_json({
            "type": "state",
            "state": state
        })
    except:
        pass


async def send_transcript(websocket: WebSocket, role: str, text: str):
    """Send transcript to client"""
    try:
        await websocket.send_json({
            "type": "transcript",
            "role": role,
            "text": text
        })
    except:
        pass


async def send_time_update(session: ConversationSession):
    """Send time remaining update"""
    try:
        await session.websocket.send_json({
            "type": "time_update",
            "remaining": int(session.remaining_seconds),
            "elapsed": int(session.elapsed_seconds)
        })
    except:
        pass


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
