# Lucy — A 15-Minute Conversation

A psychologically intelligent voice AI that creates genuine human space for conversation.

## 🌐 Live Demo

**Frontend:** https://n6c3g6qdtqs6o.ok.kimi.link (static, needs backend)

## 🏗️ Architecture

```
┌─────────────────┐      WebSocket       ┌─────────────────┐
│   React Frontend│ ◄──────────────────► │  FastAPI Backend│
│   (Static Host) │   wss://.../ws/chat  │   (Fly.io)      │
└─────────────────┘                      └─────────────────┘
```

## 🚀 Deployment Options

### Option 1: Fly.io (Recommended for Live Demo)

#### Step 1: Deploy Backend to Fly.io

```bash
# Install flyctl (if not already installed)
curl -L https://fly.io/install.sh | sh

# Login to Fly.io
flyctl auth login

# Deploy backend
cd backend
flyctl launch --name lucy-voice-ai --region iad
flyctl deploy
```

Or use the provided script:
```bash
./deploy-flyio.sh
```

After deployment, your backend will be at:
- **HTTPS:** `https://lucy-voice-ai.fly.dev`
- **WebSocket:** `wss://lucy-voice-ai.fly.dev/ws/chat`

#### Step 2: Update Frontend Config

Edit `.env` in the project root:
```bash
VITE_BACKEND_URL=wss://lucy-voice-ai.fly.dev
```

#### Step 3: Build & Deploy Frontend

```bash
npm run build
```

Deploy the `dist/` folder to any static host:
- **Vercel:** `vercel --prod`
- **Netlify:** Drag `dist/` to Netlify Drop
- **GitHub Pages:** Push `dist/` to gh-pages branch
- **Kimi Link:** Already deployed at https://n6c3g6qdtqs6o.ok.kimi.link

---

### Option 2: Run Locally (Development)

```bash
# Terminal 1 - Backend
cd backend
pip install -r requirements.txt
python main.py
# Runs on http://localhost:8000

# Terminal 2 - Frontend
cd ..
npm install
npm run dev
# Runs on http://localhost:3000
```

---

## 📁 Project Structure

```
├── backend/
│   ├── main.py                 # FastAPI WebSocket server
│   ├── audio_processor.py      # VAD (400ms silence detection)
│   ├── personaplex_client.py   # AI API client (mock mode)
│   ├── requirements.txt        # Python dependencies
│   ├── Dockerfile              # Fly.io container config
│   └── fly.toml               # Fly.io deployment config
├── src/
│   ├── components/
│   │   └── Orb.tsx            # Animated voice visualizer
│   ├── hooks/
│   │   └── useVoiceAgent.ts   # WebSocket & audio logic
│   ├── App.tsx                # Main UI
│   └── App.css                # Styles
├── .env.example               # Environment variables template
├── deploy-flyio.sh            # Fly.io deployment script
└── dist/                      # Built frontend
```

---

## 🎨 Features

| Feature | Status |
|---------|--------|
| **15-Minute Timer** | ✅ Tracks session, graceful closing |
| **VAD (400ms silence)** | ✅ Energy-based detection |
| **16kHz → 24kHz PCM** | ✅ Correct audio pipeline |
| **Orb Visualizer** | ✅ Organic blob animation |
| **Interruption** | ✅ Click orb or send interrupt |
| **Mock AI Mode** | ✅ Works without API keys |

---

## 🔌 WebSocket Protocol

**Client → Server:**
- `{type: "start"}` - Begin conversation
- `{type: "interrupt"}` - Stop AI speaking
- Binary: PCM audio (16kHz, 16-bit)

**Server → Client:**
- `{type: "state", state: "listening|thinking|speaking"}`
- `{type: "transcript", role: "user\|assistant", text: "..."}`
- `{type: "time_update", remaining: 900, elapsed: 0}`
- `{type: "ended", message: "..."}`
- Binary: PCM audio (24kHz, 16-bit)

---

## 📝 To Use Real AI API

Edit `backend/personaplex_client.py`:
```python
def __init__(
    self,
    api_key: Optional[str] = None,
    base_url: str = "https://api.personaplex.io",
    mock_mode: False  # Change this
):
```

Set API key via Fly.io secrets:
```bash
flyctl secrets set PERSONAPLEX_API_KEY=your_key_here
```

---

## 🐛 Troubleshooting

### "Failed to construct WebSocket"
The frontend is HTTPS but trying to connect to unencrypted WS. Update `.env` to use `wss://` (secure WebSocket).

### "WebSocket connection failed"
- Check backend is running: `flyctl status`
- Verify WebSocket URL in frontend config
- Check Fly.io logs: `flyctl logs`

### No audio output
- Allow microphone permissions in browser
- Ensure AudioContext is unlocked (user interaction required)

---

## 📄 License

MIT License
