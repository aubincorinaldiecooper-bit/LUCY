#!/bin/bash

# PersonaPlex Voice AI - Start Script
# This script starts both the backend and frontend

echo "🎙️  Starting PersonaPlex Voice AI Demo"
echo "========================================"

# Check if we're in the right directory
if [ ! -d "backend" ] || [ ! -f "package.json" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Function to cleanup processes on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup INT TERM

# Start backend
echo "📡 Starting backend on port 8000..."
cd backend
python main.py &
BACKEND_PID=$!
cd ..

# Wait for backend to start
sleep 2

# Check if backend started successfully
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "❌ Backend failed to start. Check backend/main.py for errors."
    exit 1
fi

echo "✅ Backend running on http://localhost:8000"
echo ""

# Start frontend
echo "🎨 Starting frontend on port 3000..."
npm run dev &
FRONTEND_PID=$!

# Wait for frontend to start
sleep 3

# Check if frontend started successfully
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    echo "❌ Frontend failed to start. Check npm install was run."
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

echo "✅ Frontend running on http://localhost:3000"
echo ""
echo "========================================"
echo "🚀 PersonaPlex Voice AI is ready!"
echo ""
echo "Open your browser and navigate to:"
echo "   http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop both services"
echo "========================================"

# Wait for both processes
wait
