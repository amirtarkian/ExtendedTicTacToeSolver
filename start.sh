#!/bin/bash

# Start script for Extended Tic-Tac-Toe Solver
# This script starts both the backend API and frontend

echo "Starting Extended Tic-Tac-Toe Solver..."
echo ""

# Check if Python dependencies are installed
if ! python3 -c "import flask" 2>/dev/null; then
    echo "Installing Python dependencies..."
    pip3 install -r requirements.txt
fi

# Check if Node modules are installed
if [ ! -d "frontend/node_modules" ]; then
    echo "Installing Node.js dependencies..."
    cd frontend
    npm install
    cd ..
fi

echo "Starting backend API on http://localhost:5000"
python3 api.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 2

echo "Starting frontend on http://localhost:3000"
cd frontend
npm start &
FRONTEND_PID=$!

echo ""
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo ""
echo "To stop the servers, run: kill $BACKEND_PID $FRONTEND_PID"
echo "Or press Ctrl+C and then run: pkill -f 'api.py|react-scripts'"

# Wait for user interrupt
wait

