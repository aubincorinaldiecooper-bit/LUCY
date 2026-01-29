import { useState, useRef, useCallback, useEffect } from 'react';

export type AgentState = 'idle' | 'connecting' | 'listening' | 'thinking' | 'speaking' | 'ended';

interface Transcript {
  role: 'user' | 'assistant';
  text: string;
  timestamp: number;
}

interface TimeInfo {
  remaining: number;
  elapsed: number;
}

interface UseVoiceAgentReturn {
  state: AgentState;
  transcripts: Transcript[];
  timeInfo: TimeInfo | null;
  error: string | null;
  start: () => Promise<void>;
  stop: () => void;
  interrupt: () => void;
}

// Audio constants
const SAMPLE_RATE = 16000;
const BUFFER_SIZE = 4096;

export const useVoiceAgent = (): UseVoiceAgentReturn => {
  const [state, setState] = useState<AgentState>('idle');
  const [transcripts, setTranscripts] = useState<Transcript[]>([]);
  const [timeInfo, setTimeInfo] = useState<TimeInfo | null>(null);
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioWorkletRef = useRef<AudioWorkletNode | null>(null);
  const isPlayingRef = useRef(false);
  const playbackSourceRef = useRef<AudioBufferSourceNode | null>(null);

  // Initialize audio context
  const initAudioContext = useCallback(async () => {
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate: SAMPLE_RATE,
      });
    }
    if (audioContextRef.current.state === 'suspended') {
      await audioContextRef.current.resume();
    }
  }, []);

  // Setup audio worklet for processing microphone input
  const setupAudioWorklet = useCallback(async () => {
    if (!audioContextRef.current) return;

    const audioContext = audioContextRef.current;

    // Create audio worklet processor code as a blob
    const workletCode = `
      class PCMProcessor extends AudioProcessor {
        constructor() {
          super();
          this.buffer = new Float32Array(0);
        }

        process(inputs, outputs, parameters) {
          const input = inputs[0];
          if (input && input[0]) {
            const channelData = input[0];
            
            // Convert Float32 to Int16
            const int16Data = new Int16Array(channelData.length);
            for (let i = 0; i < channelData.length; i++) {
              const sample = Math.max(-1, Math.min(1, channelData[i]));
              int16Data[i] = Math.round(sample * 32767);
            }
            
            // Send to main thread
            this.port.postMessage(int16Data.buffer, [int16Data.buffer]);
          }
          return true;
        }
      }

      registerProcessor('pcm-processor', PCMProcessor);
    `;

    const blob = new Blob([workletCode], { type: 'application/javascript' });
    const url = URL.createObjectURL(blob);

    try {
      await audioContext.audioWorklet.addModule(url);
    } catch (e) {
      console.warn('AudioWorklet not supported, falling back to ScriptProcessor');
    }

    URL.revokeObjectURL(url);
  }, []);

  // Start microphone capture
  const startMicrophone = useCallback(async () => {
    try {
      await initAudioContext();

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: SAMPLE_RATE,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      mediaStreamRef.current = stream;

      if (!audioContextRef.current) return;

      const source = audioContextRef.current.createMediaStreamSource(stream);

      // Try AudioWorklet first, fallback to ScriptProcessor
      if (audioContextRef.current.audioWorklet) {
        await setupAudioWorklet();
        const worklet = new AudioWorkletNode(audioContextRef.current, 'pcm-processor');
        
        worklet.port.onmessage = (event) => {
          if (wsRef.current?.readyState === WebSocket.OPEN && state !== 'speaking') {
            const int16Data = new Int16Array(event.data);
            const bytes = new Uint8Array(int16Data.buffer);
            wsRef.current.send(bytes);
          }
        };

        source.connect(worklet);
        audioWorkletRef.current = worklet;
      } else {
        // Fallback to ScriptProcessorNode
        const processor = audioContextRef.current.createScriptProcessor(BUFFER_SIZE, 1, 1);
        
        processor.onaudioprocess = (e) => {
          if (wsRef.current?.readyState === WebSocket.OPEN && state !== 'speaking') {
            const channelData = e.inputBuffer.getChannelData(0);
            const int16Data = new Int16Array(channelData.length);
            
            for (let i = 0; i < channelData.length; i++) {
              const sample = Math.max(-1, Math.min(1, channelData[i]));
              int16Data[i] = Math.round(sample * 32767);
            }
            
            wsRef.current.send(int16Data.buffer);
          }
        };

        source.connect(processor);
        processor.connect(audioContextRef.current.destination);
        audioWorkletRef.current = processor as any;
      }
    } catch (e) {
      setError('Microphone access denied. Please allow microphone permissions.');
      throw e;
    }
  }, [initAudioContext, setupAudioWorklet, state]);

  // Stop microphone capture
  const stopMicrophone = useCallback(() => {
    if (audioWorkletRef.current) {
      audioWorkletRef.current.disconnect();
      audioWorkletRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }
  }, []);

  // Play audio from PCM data
  const playAudio = useCallback(async (pcmData: ArrayBuffer) => {
    if (!audioContextRef.current) return;

    const int16Data = new Int16Array(pcmData);
    const floatData = new Float32Array(int16Data.length);
    
    for (let i = 0; i < int16Data.length; i++) {
      floatData[i] = int16Data[i] / 32768;
    }

    const audioBuffer = audioContextRef.current.createBuffer(1, floatData.length, 24000);
    audioBuffer.getChannelData(0).set(floatData);

    const source = audioContextRef.current.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioContextRef.current.destination);
    
    playbackSourceRef.current = source;
    
    source.onended = () => {
      isPlayingRef.current = false;
      playbackSourceRef.current = null;
    };

    isPlayingRef.current = true;
    source.start();
  }, []);

  // Stop current playback
  const stopPlayback = useCallback(() => {
    if (playbackSourceRef.current) {
      playbackSourceRef.current.stop();
      playbackSourceRef.current = null;
      isPlayingRef.current = false;
    }
  }, []);

  // Start the conversation
  const start = useCallback(async () => {
    try {
      setError(null);
      setState('connecting');

      // Initialize WebSocket
      // For local dev: ws://localhost:8000/ws/chat
      // For Fly.io: wss://lucy-voice-ai.fly.dev/ws/chat
      const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'ws://localhost:8000';
      const wsUrl = `${BACKEND_URL}/ws/chat`;
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      await new Promise<void>((resolve, reject) => {
        ws.onopen = () => resolve();
        ws.onerror = () => reject(new Error('WebSocket connection failed'));
      });

      // Setup WebSocket handlers
      ws.onmessage = async (event) => {
        if (event.data instanceof Blob) {
          // Binary audio data
          const arrayBuffer = await event.data.arrayBuffer();
          playAudio(arrayBuffer);
        } else {
          // JSON message
          const msg = JSON.parse(event.data);
          
          switch (msg.type) {
            case 'state':
              setState(msg.state);
              break;
            case 'transcript':
              setTranscripts(prev => [...prev, {
                role: msg.role,
                text: msg.text,
                timestamp: Date.now(),
              }]);
              break;
            case 'time_update':
              setTimeInfo({
                remaining: msg.remaining,
                elapsed: msg.elapsed,
              });
              break;
            case 'ended':
              setState('ended');
              stopMicrophone();
              break;
            case 'error':
              setError(msg.message);
              break;
          }
        }
      };

      ws.onclose = () => {
        setState('idle');
        stopMicrophone();
      };

      ws.onerror = () => {
        setError('Connection error. Please try again.');
        setState('idle');
      };

      // Start microphone
      await startMicrophone();

      // Send start message to trigger AI opener
      ws.send(JSON.stringify({ type: 'start' }));

    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to start');
      setState('idle');
    }
  }, [startMicrophone, playAudio, stopMicrophone]);

  // Stop the conversation
  const stop = useCallback(() => {
    stopMicrophone();
    stopPlayback();
    
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    
    setState('idle');
    setTranscripts([]);
    setTimeInfo(null);
  }, [stopMicrophone, stopPlayback]);

  // Interrupt AI speaking
  const interrupt = useCallback(() => {
    stopPlayback();
    
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'interrupt' }));
    }
  }, [stopPlayback]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stop();
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, [stop]);

  return {
    state,
    transcripts,
    timeInfo,
    error,
    start,
    stop,
    interrupt,
  };
};

export default useVoiceAgent;
