#!/bin/bash

# Lucy Voice AI - Fly.io Deployment Script
# This script deploys the backend to Fly.io

set -e

echo "🚀 Deploying Lucy Voice AI to Fly.io"
echo "====================================="

# Check if flyctl is installed
if ! command -v flyctl &> /dev/null; then
    echo "❌ flyctl not found. Installing..."
    curl -L https://fly.io/install.sh | sh
    export PATH="$HOME/.fly/bin:$PATH"
fi

# Check if user is logged in
if ! flyctl auth whoami &> /dev/null; then
    echo "🔑 Please login to Fly.io:"
    flyctl auth login
fi

cd backend

# Check if app already exists
if flyctl apps list | grep -q "lucy-voice-ai"; then
    echo "📦 App exists, deploying updates..."
    flyctl deploy
else
    echo "🆕 Creating new app..."
    flyctl launch --name lucy-voice-ai --region iad --no-deploy
    flyctl deploy
fi

echo ""
echo "✅ Backend deployed!"
echo ""
echo "Backend URL: https://lucy-voice-ai.fly.dev"
echo "WebSocket:   wss://lucy-voice-ai.fly.dev/ws/chat"
echo ""
echo "Next steps:"
echo "1. Update .env file with: VITE_BACKEND_URL=wss://lucy-voice-ai.fly.dev"
echo "2. Run: npm run build"
echo "3. Deploy frontend to your preferred static host"
