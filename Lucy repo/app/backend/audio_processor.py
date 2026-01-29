"""
Audio Processing and Voice Activity Detection (VAD)
Handles audio conversion, silence detection, and buffering
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional
from collections import deque


class VADState(Enum):
    SILENCE = "silence"
    SPEECH = "speech"


@dataclass
class VADResult:
    state: VADState
    is_final: bool = False  # True when speech segment is complete


class AudioProcessor:
    """
    Processes audio chunks with Voice Activity Detection
    
    Features:
    - Energy-based VAD (lightweight, no ML model needed)
    - Configurable silence threshold
    - Accumulates speech segments
    - Converts between sample rates and formats
    """
    
    def __init__(
        self,
        silence_threshold_ms: float = 400,
        sample_rate: int = 16000,
        energy_threshold: float = 0.02,
        buffer_duration_ms: float = 500
    ):
        self.sample_rate = sample_rate
        self.silence_threshold_ms = silence_threshold_ms
        self.energy_threshold = energy_threshold
        
        # Calculate frame sizes
        self.frame_size = int(sample_rate * 0.02)  # 20ms frames
        self.silence_frames_needed = int(silence_threshold_ms / 20)
        
        # State
        self.is_speaking = False
        self.silence_frame_count = 0
        self.speech_buffer: List[np.ndarray] = []
        self.accumulated_audio: List[np.ndarray] = []
        
        # Ring buffer for pre-speech audio (catch the start of utterances)
        self.pre_buffer = deque(maxlen=int(buffer_duration_ms / 20))
        
    def process_chunk(self, audio_data: np.ndarray) -> VADResult:
        """
        Process an audio chunk and return VAD state
        
        Args:
            audio_data: 16-bit PCM audio samples
            
        Returns:
            VADResult with current state and whether speech segment is complete
        """
        # Split into 20ms frames
        frames = self._split_into_frames(audio_data)
        
        for frame in frames:
            result = self._process_frame(frame)
            
        return result
    
    def _split_into_frames(self, audio_data: np.ndarray) -> List[np.ndarray]:
        """Split audio into 20ms frames"""
        frames = []
        for i in range(0, len(audio_data), self.frame_size):
            frame = audio_data[i:i + self.frame_size]
            if len(frame) == self.frame_size:  # Only complete frames
                frames.append(frame)
        return frames
    
    def _process_frame(self, frame: np.ndarray) -> VADResult:
        """Process a single frame"""
        # Normalize to float [-1, 1]
        normalized = frame.astype(np.float32) / 32768.0
        
        # Calculate energy (RMS)
        energy = np.sqrt(np.mean(normalized ** 2))
        
        is_speech = energy > self.energy_threshold
        
        if is_speech:
            if not self.is_speaking:
                # Speech started - include pre-buffer
                self.is_speaking = True
                self.silence_frame_count = 0
                self.speech_buffer = list(self.pre_buffer)
            
            self.speech_buffer.append(frame)
            self.pre_buffer.append(frame)
            return VADResult(state=VADState.SPEECH, is_final=False)
            
        else:
            # Silence detected
            self.pre_buffer.append(frame)
            
            if self.is_speaking:
                self.silence_frame_count += 1
                self.speech_buffer.append(frame)
                
                if self.silence_frame_count >= self.silence_frames_needed:
                    # Silence threshold reached - speech segment complete
                    self.is_speaking = False
                    self.accumulated_audio.extend(self.speech_buffer)
                    self.speech_buffer = []
                    return VADResult(state=VADState.SILENCE, is_final=True)
                else:
                    return VADResult(state=VADState.SPEECH, is_final=False)
            else:
                return VADResult(state=VADState.SILENCE, is_final=False)
    
    def get_accumulated_audio(self) -> np.ndarray:
        """Get all accumulated speech audio"""
        if not self.accumulated_audio:
            return np.array([], dtype=np.int16)
        return np.concatenate(self.accumulated_audio)
    
    def clear_accumulated(self):
        """Clear accumulated audio buffer"""
        self.accumulated_audio = []
        self.speech_buffer = []
        self.pre_buffer.clear()
        self.is_speaking = False
        self.silence_frame_count = 0
    
    @staticmethod
    def resample(audio_data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate using linear interpolation"""
        if orig_sr == target_sr:
            return audio_data
            
        # Calculate new length
        duration = len(audio_data) / orig_sr
        new_length = int(duration * target_sr)
        
        # Linear interpolation
        indices = np.linspace(0, len(audio_data) - 1, new_length)
        indices_floor = np.floor(indices).astype(np.int32)
        indices_ceil = np.minimum(indices_floor + 1, len(audio_data) - 1)
        fractions = indices - indices_floor
        
        resampled = audio_data[indices_floor] * (1 - fractions) + audio_data[indices_ceil] * fractions
        return resampled.astype(np.int16)
    
    @staticmethod
    def convert_float_to_int16(float_audio: np.ndarray) -> np.ndarray:
        """Convert float audio [-1, 1] to int16"""
        return (float_audio * 32767).astype(np.int16)
    
    @staticmethod
    def convert_int16_to_float(int_audio: np.ndarray) -> np.ndarray:
        """Convert int16 audio to float [-1, 1]"""
        return int_audio.astype(np.float32) / 32768.0
    
    @staticmethod
    def webm_to_pcm(webm_bytes: bytes, target_sample_rate: int = 16000) -> np.ndarray:
        """
        Convert WebM audio to PCM
        Requires pydub - falls back to returning empty if not available
        """
        try:
            from pydub import AudioSegment
            import io
            
            # Load WebM
            audio = AudioSegment.from_file(io.BytesIO(webm_bytes), format="webm")
            
            # Convert to mono if stereo
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            # Resample if needed
            if audio.frame_rate != target_sample_rate:
                audio = audio.set_frame_rate(target_sample_rate)
            
            # Export as raw PCM
            pcm_data = np.frombuffer(audio.raw_data, dtype=np.int16)
            return pcm_data
            
        except ImportError:
            raise RuntimeError("pydub required for WebM conversion")
        except Exception as e:
            raise RuntimeError(f"WebM conversion failed: {e}")


class SimpleVAD:
    """
    Simple energy-based Voice Activity Detector
    Lightweight alternative to ML-based VAD like Silero
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        threshold: float = 0.02,
        min_speech_duration_ms: float = 250,
        min_silence_duration_ms: float = 400
    ):
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.threshold = threshold
        self.min_speech_frames = int(min_speech_duration_ms / frame_duration_ms)
        self.min_silence_frames = int(min_silence_duration_ms / frame_duration_ms)
        
        self.speech_frames = 0
        self.silence_frames = 0
        self.is_speaking = False
        self.buffer: List[np.ndarray] = []
        
    def process(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """
        Process audio chunk and return speech segment when complete
        
        Returns:
            Speech audio array when segment is complete, None otherwise
        """
        # Split into frames
        for i in range(0, len(audio_chunk), self.frame_size):
            frame = audio_chunk[i:i + self.frame_size]
            if len(frame) < self.frame_size:
                continue
                
            # Normalize and calculate energy
            normalized = frame.astype(np.float32) / 32768.0
            energy = np.sqrt(np.mean(normalized ** 2))
            
            if energy > self.threshold:
                self.speech_frames += 1
                self.silence_frames = 0
                
                if not self.is_speaking and self.speech_frames >= 2:
                    self.is_speaking = True
                    
                if self.is_speaking:
                    self.buffer.append(frame)
                    
            else:
                self.silence_frames += 1
                
                if self.is_speaking:
                    self.buffer.append(frame)
                    
                    if self.silence_frames >= self.min_silence_frames:
                        # Speech segment complete
                        if self.speech_frames >= self.min_speech_frames:
                            result = np.concatenate(self.buffer)
                        else:
                            result = None  # Too short
                            
                        self.reset()
                        return result
                        
                else:
                    self.speech_frames = max(0, self.speech_frames - 1)
                    
        return None
    
    def reset(self):
        """Reset VAD state"""
        self.speech_frames = 0
        self.silence_frames = 0
        self.is_speaking = False
        self.buffer = []
    
    def force_complete(self) -> Optional[np.ndarray]:
        """Force completion of current speech segment"""
        if self.is_speaking and self.speech_frames >= self.min_speech_frames:
            result = np.concatenate(self.buffer)
            self.reset()
            return result
        return None
