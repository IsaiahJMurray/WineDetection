#!/bin/bash

# Startup script for Wine Inference API
# Starts ngrok tunnel and uvicorn server
#
# Usage:
#   1. Make script executable: chmod +x start.sh
#   2. Run: ./start.sh
#   3. Press Ctrl+C to stop both services
#
# Requirements:
#   - ngrok installed and in PATH
#   - uvicorn installed (pip install uvicorn)
#   - Python dependencies installed (pip install -r requirements.txt)

set -e  # Exit on error

# Configuration
PORT=8000
HOST="0.0.0.0"
NGROK_REGION="us"  # Change to your preferred region (us, eu, ap, au, sa, jp, in)
RELOAD=""  # Set to "--reload" for development (auto-reload on code changes)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down...${NC}"
    if [ ! -z "$NGROK_PID" ]; then
        echo "Stopping ngrok (PID: $NGROK_PID)"
        kill $NGROK_PID 2>/dev/null || true
    fi
    if [ ! -z "$UVICORN_PID" ]; then
        echo "Stopping uvicorn (PID: $UVICORN_PID)"
        kill $UVICORN_PID 2>/dev/null || true
    fi
    exit 0
}

# Trap Ctrl+C and cleanup
trap cleanup SIGINT SIGTERM

# Check if ngrok is installed
if ! command -v ngrok &> /dev/null; then
    echo -e "${RED}Error: ngrok is not installed or not in PATH${NC}"
    echo "Install ngrok from: https://ngrok.com/download"
    exit 1
fi

# Check if uvicorn is installed
if ! command -v uvicorn &> /dev/null; then
    echo -e "${RED}Error: uvicorn is not installed${NC}"
    echo "Install with: pip install uvicorn"
    exit 1
fi

# Check if app.py exists
if [ ! -f "app.py" ]; then
    echo -e "${RED}Error: app.py not found in current directory${NC}"
    exit 1
fi

echo -e "${GREEN}Starting Wine Inference API...${NC}"
echo "Port: $PORT"
echo ""

# Start ngrok tunnel
echo -e "${YELLOW}Starting ngrok tunnel...${NC}"
ngrok http $PORT --region=$NGROK_REGION > /dev/null 2>&1 &
NGROK_PID=$!

# Wait a moment for ngrok to start
sleep 2

# Get ngrok URL (try to extract from ngrok API)
NGROK_URL=""
for i in {1..10}; do
    sleep 1
    # Use Python to parse JSON and extract URL (more reliable)
    if command -v python3 &> /dev/null; then
        NGROK_URL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null | \
            python3 -c "import sys, json; data = json.load(sys.stdin); \
            tunnels = data.get('tunnels', []); \
            print(tunnels[0]['public_url'] if tunnels and 'public_url' in tunnels[0] else '')" 2>/dev/null || echo "")
    else
        # Fallback: try simple grep (less reliable)
        NGROK_URL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null | \
            grep -o 'https://[^"]*ngrok[^"]*' | head -1 || echo "")
    fi
    if [ ! -z "$NGROK_URL" ]; then
        break
    fi
done

if [ -z "$NGROK_URL" ]; then
    echo -e "${YELLOW}Warning: Could not automatically detect ngrok URL${NC}"
    echo "Check ngrok dashboard at: http://localhost:4040"
    echo "Or manually get URL from: curl http://localhost:4040/api/tunnels"
    echo ""
else
    echo -e "${GREEN}Ngrok tunnel active:${NC} $NGROK_URL"
    echo -e "${GREEN}API endpoint:${NC} $NGROK_URL/predict"
    echo ""
fi

echo -e "${YELLOW}Ngrok web interface:${NC} http://localhost:4040"
echo ""

# Start uvicorn server
echo -e "${YELLOW}Starting uvicorn server on port $PORT...${NC}"
if [ ! -z "$RELOAD" ]; then
    echo "  (with auto-reload enabled)"
fi
uvicorn app:app --host $HOST --port $PORT $RELOAD &
UVICORN_PID=$!

# Wait a moment for uvicorn to start
sleep 2

# Check if uvicorn started successfully
if ! kill -0 $UVICORN_PID 2>/dev/null; then
    echo -e "${RED}Error: uvicorn failed to start${NC}"
    cleanup
    exit 1
fi

echo -e "${GREEN}Uvicorn server started (PID: $UVICORN_PID)${NC}"
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Server is running!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Local URL: http://$HOST:$PORT"
if [ ! -z "$NGROK_URL" ]; then
    echo "Public URL: $NGROK_URL"
    echo ""
    echo "Update index.html API URL to: $NGROK_URL/predict"
fi
echo ""
echo "Press Ctrl+C to stop both services"
echo ""

# Wait for uvicorn to finish (or until interrupted)
wait $UVICORN_PID

