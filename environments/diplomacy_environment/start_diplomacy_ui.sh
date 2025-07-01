#!/bin/bash
# Script to start the AI Diplomacy visualization interface

echo "Starting AI Diplomacy Animation Interface..."
echo "==========================================="

cd AI_Diplomacy/ai_animation

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

echo ""
echo "Starting development server..."
echo "The interface will be available at http://localhost:5173"
echo ""
echo "To visualize a game:"
echo "1. Load a game JSON file from the interface"
echo "2. Use Play button to start animated playback"
echo "3. Or use Next/Previous buttons for manual control"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

npm run dev