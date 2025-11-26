# Extended Tic-Tac-Toe Solver

An infinite tic-tac-toe game implementation with 5-in-a-row win condition, designed for Monte Carlo Tree Search (MCTS) and future machine learning enhancements.

## Features

- **Infinite Board**: The game board has no size limits - it expands dynamically as players make moves
- **5-in-a-Row Win Condition**: Players must get 5 consecutive pieces (horizontally, vertically, or diagonally) to win
- **MCTS-Ready**: Full implementation of Monte Carlo Tree Search algorithm
- **Sparse Representation**: Efficient memory usage with dictionary-based board representation
- **React Frontend**: Beautiful, interactive web interface for playing the game
- **REST API**: Flask backend API for game logic and MCTS integration
- **Clean Interface**: Well-structured code ready for ML model integration

## Game Rules

1. Players alternate turns (X goes first)
2. Players place their piece on any empty position adjacent to existing pieces
3. First player to get 5 pieces in a row (horizontal, vertical, or diagonal) wins
4. The board expands infinitely as needed

## Installation

### Backend Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install Node.js dependencies:
```bash
npm install
```

## Running the Application

### Option 1: Web Interface (Recommended)

1. **Start the backend API** (in one terminal):
```bash
python api.py
```
The API will run on `http://localhost:5000`

2. **Start the React frontend** (in another terminal):
```bash
cd frontend
npm start
```
The frontend will open automatically at `http://localhost:3000`

### Option 2: Command Line Interface

```bash
# Run the game
python main.py
```

## Usage

### Human vs MCTS (Default)
```bash
python main.py
# or
python main.py mcts [simulations]
```

### Human vs Human
```bash
python main.py human
```

### MCTS vs MCTS (Testing)
```bash
python main.py test [simulations] [num_games]
```

### Examples
```bash
# Human vs MCTS with 2000 simulations per move
python main.py mcts 2000

# Run 5 MCTS vs MCTS games with 1000 simulations
python main.py test 1000 5
```

## Code Structure

### Backend

#### `game.py`
- `GameState`: Core game logic and state management
- `Player`: Enum for X and O players
- Methods for MCTS compatibility:
  - `get_legal_moves()`: Returns all valid moves
  - `is_terminal()`: Checks if game is over
  - `get_result()`: Returns game result from current player's perspective
  - `make_move()`: Returns new state without modifying current
  - `copy()`: Deep copy for tree search

#### `mcts.py`
- `MCTSNode`: Tree node with UCB1 selection
- `MCTS`: Full Monte Carlo Tree Search implementation
- Ready for enhancement with neural network policies

#### `api.py`
- Flask REST API server
- Endpoints for game management and moves
- MCTS integration endpoints

#### `main.py`
- Command-line interactive game interface
- Support for different game modes
- Testing utilities

### Frontend

#### `frontend/src/App.js`
- Main React application component
- Game state management
- Coordinates between components

#### `frontend/src/components/GameBoard.js`
- Visual board representation
- Handles cell clicks and move input
- Displays legal moves

#### `frontend/src/components/GameStatus.js`
- Shows current player and game status
- Displays winner/draw messages

#### `frontend/src/components/GameControls.js`
- New game button
- MCTS play controls

#### `frontend/src/api.js`
- API client for backend communication
- Axios-based HTTP requests

## MCTS Integration

The game is designed to work seamlessly with MCTS:

```python
from game import GameState
from mcts import MCTS

# Create initial state
state = GameState()

# Initialize MCTS
mcts = MCTS(simulations=1000)

# Get best move
best_move = mcts.search(state)

# Make the move
new_state = state.make_move(*best_move)
```

## Future ML Enhancements

The codebase is structured to easily integrate:

1. **Neural Network Policies**: Replace random simulation with neural network evaluation
2. **AlphaZero-style Training**: Add self-play training loop
3. **Value Networks**: Use neural networks to estimate position values
4. **Policy Networks**: Guide MCTS expansion with learned policies

## Key Design Decisions

1. **Sparse Board Representation**: Uses dictionary `{(row, col): Player}` instead of 2D array for infinite board
2. **Immutable States**: `make_move()` returns new state, enabling safe tree search
3. **Adjacent Move Constraint**: Only considers moves adjacent to existing pieces (prevents infinite move space)
4. **Efficient Win Detection**: Checks only relevant lines after each move

## API Endpoints

The Flask API provides the following endpoints:

- `POST /api/game/new` - Create a new game
- `GET /api/game/<game_id>` - Get current game state
- `POST /api/game/<game_id>/move` - Make a human move (requires `row` and `col` in JSON body)
- `POST /api/game/<game_id>/mcts` - Get MCTS move recommendation (without playing)
- `POST /api/game/<game_id>/mcts/play` - Make MCTS play a move
- `POST /api/game/<game_id>/reset` - Reset game to initial state

## Testing

### Command Line Testing

Run MCTS vs MCTS to test the implementation:

```bash
python main.py test 1000 10
```

This will run 10 games with 1000 simulations per move and show the results.

### Web Interface Testing

1. Start both backend and frontend (see Installation section)
2. Open `http://localhost:3000` in your browser
3. Click on highlighted cells to make moves
4. MCTS will automatically play as player O

## License

This project is open source and available for educational and research purposes.

