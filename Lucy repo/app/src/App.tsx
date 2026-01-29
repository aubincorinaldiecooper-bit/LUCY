import { useEffect, useRef } from 'react';
import { Mic, MicOff, PhoneOff, Clock } from 'lucide-react';
import { Orb, type AgentState } from './components/Orb';
import { useVoiceAgent } from './hooks/useVoiceAgent';
import './App.css';

// Format seconds to MM:SS
const formatTime = (seconds: number): string => {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}:${secs.toString().padStart(2, '0')}`;
};

// Get state label
const getStateLabel = (state: string): string => {
  switch (state) {
    case 'idle':
      return 'Ready';
    case 'connecting':
      return 'Connecting...';
    case 'listening':
      return 'Listening';
    case 'thinking':
      return 'Thinking';
    case 'speaking':
      return 'Speaking';
    case 'ended':
      return 'Goodbye';
    default:
      return 'Ready';
  }
};

// Orb color pairs [primary, secondary] - soft, muted tones
const ORB_COLORS: [string, string] = ["#E8E4E0", "#D4CFC8"]

function App() {
  const {
    state,
    transcripts,
    timeInfo,
    error,
    start,
    stop,
    interrupt,
  } = useVoiceAgent();

  const transcriptsRef = useRef<HTMLDivElement>(null);
  const stateLabel = getStateLabel(state);

  // Auto-scroll transcripts
  useEffect(() => {
    if (transcriptsRef.current) {
      transcriptsRef.current.scrollTop = transcriptsRef.current.scrollHeight;
    }
  }, [transcripts]);

  // Handle start button click
  const handleStart = async () => {
    await start();
  };

  // Handle orb click (interrupt when AI is speaking)
  const handleOrbClick = () => {
    if (state === 'speaking') {
      interrupt();
    }
  };

  // Map internal state to Orb agentState
  const getOrbState = (): AgentState => {
    if (state === 'listening') return 'listening';
    if (state === 'thinking' || state === 'speaking') return 'talking';
    return null; // idle, connecting, ended
  };

  const isActive = state !== 'idle' && state !== 'ended';

  return (
    <div className="min-h-screen bg-[#FAF9F7] text-[#2D2D2D] flex flex-col items-center justify-center p-6 font-sans">
      {/* Header */}
      <header className="absolute top-0 left-0 right-0 p-8 flex justify-between items-center">
        <div className="flex items-center gap-3">
          <span className="text-sm tracking-[0.2em] uppercase text-[#8B8680]">[lucy]</span>
        </div>
        
        {timeInfo && (
          <div className="flex items-center gap-2 text-[#8B8680]">
            <Clock className="w-4 h-4" strokeWidth={1.5} />
            <span className="text-sm font-light tracking-wide">
              {formatTime(timeInfo.remaining)}
            </span>
          </div>
        )}
      </header>

      {/* Main Content */}
      <main className="flex flex-col items-center gap-10 w-full max-w-xl">
        {/* Orb Visualizer */}
        <div 
          className={`relative transition-all duration-500 ${isActive ? 'cursor-pointer' : ''}`}
          onClick={handleOrbClick}
        >
          <div className="relative w-64 h-64 rounded-full p-2 bg-white shadow-[0_4px_24px_rgba(0,0,0,0.04)]">
            <div className="w-full h-full rounded-full overflow-hidden bg-[#F5F4F2]">
              <Orb
                colors={ORB_COLORS}
                seed={1000}
                agentState={getOrbState()}
              />
            </div>
          </div>
          
          {/* Subtle state indicator */}
          {isActive && (
            <div 
              className="absolute inset-0 rounded-full pointer-events-none transition-opacity duration-500"
              style={{
                boxShadow: state === 'listening' 
                  ? '0 0 40px 8px rgba(59, 130, 246, 0.15)' 
                  : state === 'speaking'
                  ? '0 0 40px 8px rgba(139, 92, 246, 0.15)'
                  : '0 0 40px 8px rgba(251, 146, 60, 0.1)',
              }}
            />
          )}
        </div>

        {/* State & Description */}
        <div className="text-center space-y-3">
          <p className="text-xs tracking-[0.25em] uppercase text-[#8B8680]">
            {stateLabel}
          </p>
          <p className="text-[#5A5652] text-sm font-light leading-relaxed max-w-xs mx-auto">
            {state === 'idle' && "A space to be heard. 15 minutes of genuine conversation."}
            {state === 'connecting' && "Finding a quiet moment..."}
            {state === 'listening' && "I'm here. Take your time."}
            {state === 'thinking' && "Sitting with what you shared..."}
            {state === 'speaking' && "..."}
            {state === 'ended' && "Thank you for sharing."}
          </p>
        </div>

        {/* Error Message */}
        {error && (
          <div className="text-[#C45B4A] text-sm text-center max-w-sm">
            {error}
          </div>
        )}

        {/* Control Button */}
        <div>
          {state === 'idle' ? (
            <button
              onClick={handleStart}
              className="group flex items-center gap-3 px-8 py-4 bg-[#2D2D2D] text-white rounded-full text-sm tracking-wide hover:bg-[#1D1D1D] transition-colors duration-300"
            >
              <Mic className="w-4 h-4" strokeWidth={1.5} />
              <span>Start Conversation</span>
            </button>
          ) : state === 'ended' ? (
            <button
              onClick={stop}
              className="group flex items-center gap-3 px-8 py-4 border border-[#D4CFC8] text-[#5A5652] rounded-full text-sm tracking-wide hover:bg-[#F5F4F2] transition-colors duration-300"
            >
              <PhoneOff className="w-4 h-4" strokeWidth={1.5} />
              <span>Close</span>
            </button>
          ) : (
            <button
              onClick={stop}
              className="group flex items-center gap-3 px-6 py-3 border border-[#E8E4E0] text-[#8B8680] rounded-full text-sm tracking-wide hover:border-[#D4CFC8] hover:text-[#5A5652] transition-colors duration-300"
            >
              <MicOff className="w-4 h-4" strokeWidth={1.5} />
              <span>End</span>
            </button>
          )}
        </div>

        {/* Transcripts */}
        {transcripts.length > 0 && (
          <div className="w-full max-w-md mt-4">
            <div className="bg-white rounded-2xl shadow-[0_2px_16px_rgba(0,0,0,0.03)] overflow-hidden">
              <div 
                ref={transcriptsRef}
                className="max-h-56 overflow-y-auto p-5 space-y-4 scrollbar-thin"
              >
                {transcripts.map((t, i) => (
                  <div 
                    key={i} 
                    className={`flex ${t.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div 
                      className={`max-w-[85%] px-4 py-3 text-sm leading-relaxed ${
                        t.role === 'user'
                          ? 'bg-[#2D2D2D] text-white rounded-2xl rounded-br-md'
                          : 'bg-[#F5F4F2] text-[#2D2D2D] rounded-2xl rounded-bl-md'
                      }`}
                    >
                      {t.text}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="absolute bottom-0 left-0 right-0 p-8 text-center">
        <p className="text-[#B8B3AC] text-xs tracking-wide">
          15-minute sessions
        </p>
      </footer>
    </div>
  );
}

export default App;
