"""
Flask API server for the infinite tic-tac-toe game.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from game import GameState, Player
from minimax import Minimax
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Store game sessions (in production, use a proper database)
games = {}


def game_state_to_dict(state: GameState) -> dict:
    """Convert GameState to dictionary for JSON serialization."""
    return {
        'board': {f"{r},{c}": player.name for (r, c), player in state.board.items()},
        'current_player': state.current_player.name,
        'is_terminal': state.is_terminal(),
        'winner': state.get_winner().name if state.get_winner() else None,
        'legal_moves': [{'row': r, 'col': c} for r, c in state.get_legal_moves()],
    }


@app.route('/api/game/new', methods=['POST'])
def new_game():
    """Create a new game."""
    data = request.json or {}
    # Ensure data is a dictionary
    if not isinstance(data, dict):
        data = {}
    game_id = str(len(games))
    # Start with X (AI) so AI plays first
    state = GameState(current_player=Player.X)
    
    # Get Minimax params from request or use defaults
    minimax_params = {
        'depth': data.get('depth', 4)
    }
    
    games[game_id] = {
        'state': state,
        'minimax': Minimax(**minimax_params),
        'minimax_progress': None
    }
    return jsonify({
        'game_id': game_id,
        'state': game_state_to_dict(state),
        'minimax_params': minimax_params
    })


@app.route('/api/game/<game_id>', methods=['GET'])
def get_game(game_id):
    """Get current game state."""
    if game_id not in games:
        return jsonify({'error': 'Game not found'}), 404
    
    state = games[game_id]['state']
    return jsonify({
        'game_id': game_id,
        'state': game_state_to_dict(state)
    })


@app.route('/api/game/<game_id>/move', methods=['POST'])
def make_move(game_id):
    """Make a move in the game."""
    if game_id not in games:
        return jsonify({'error': 'Game not found'}), 404
    
    data = request.json
    row = data.get('row')
    col = data.get('col')
    
    if row is None or col is None:
        return jsonify({'error': 'Row and col are required'}), 400
    
    state = games[game_id]['state']
    
    # Validate move
    legal_moves = state.get_legal_moves()
    if (row, col) not in legal_moves:
        return jsonify({'error': 'Invalid move'}), 400
    
    # Make move
    new_state = state.make_move(row, col)
    games[game_id]['state'] = new_state
    
    return jsonify({
        'game_id': game_id,
        'state': game_state_to_dict(new_state)
    })


@app.route('/api/game/<game_id>/minimax', methods=['POST'])
def minimax_move(game_id):
    """Get Minimax move recommendation."""
    if game_id not in games:
        return jsonify({'error': 'Game not found'}), 404
    
    state = games[game_id]['state']
    
    if state.is_terminal():
        return jsonify({'error': 'Game is already over'}), 400
    
    logger.info(f"Computing Minimax move for game {game_id}")
    minimax = games[game_id]['minimax']
    move = minimax.search(state)
    logger.info(f"Minimax move computed: ({move[0]}, {move[1]})")
    
    return jsonify({
        'move': {'row': move[0], 'col': move[1]}
    })


@app.route('/api/game/<game_id>/minimax/play', methods=['POST'])
def minimax_play(game_id):
    """Make Minimax play a move."""
    if game_id not in games:
        return jsonify({'error': 'Game not found'}), 404
    
    state = games[game_id]['state']
    
    if state.is_terminal():
        return jsonify({'error': 'Game is already over'}), 400
    
    logger.info(f"Computing Minimax move for game {game_id} (auto-play)")
    minimax = games[game_id]['minimax']
    
    # Initialize progress state
    games[game_id]['minimax_progress'] = {
        'current_move': 0,
        'total_moves': 0,
        'current_depth': minimax.depth,
        'best_value': None,
        'best_move': None,
        'is_complete': False
    }
    
    # Define progress callback
    def progress_callback(progress_data):
        games[game_id]['minimax_progress'].update({
            'current_move': progress_data.get('current_move', 0),
            'total_moves': progress_data.get('total_moves', 0),
            'current_depth': progress_data.get('current_depth', minimax.depth),
            'best_value': progress_data.get('best_value'),
            'best_move': progress_data.get('best_move')
        })
    
    try:
        move = minimax.search(state, progress_callback=progress_callback)
        logger.info(f"Minimax move computed and played: ({move[0]}, {move[1]})")
        
        # Make the move
        new_state = state.make_move(*move)
        games[game_id]['state'] = new_state
        
        # Mark progress as complete
        games[game_id]['minimax_progress']['is_complete'] = True
        games[game_id]['minimax_progress']['best_move'] = move
        
        return jsonify({
            'game_id': game_id,
            'move': {'row': move[0], 'col': move[1]},
            'state': game_state_to_dict(new_state)
        })
    except Exception as e:
        # Clear progress on error
        games[game_id]['minimax_progress'] = None
        raise e


@app.route('/api/game/<game_id>/reset', methods=['POST'])
def reset_game(game_id):
    """Reset a game to initial state."""
    if game_id not in games:
        return jsonify({'error': 'Game not found'}), 404
    
    state = GameState()
    games[game_id]['state'] = state
    games[game_id]['minimax_progress'] = None
    
    return jsonify({
        'game_id': game_id,
        'state': game_state_to_dict(state)
    })


@app.route('/api/game/<game_id>/minimax/params', methods=['GET'])
def get_minimax_params(game_id):
    """Get current Minimax parameters."""
    if game_id not in games:
        return jsonify({'error': 'Game not found'}), 404
    
    minimax = games[game_id]['minimax']
    return jsonify({
        'game_id': game_id,
        'minimax_params': minimax.get_params()
    })


@app.route('/api/game/<game_id>/minimax/params', methods=['POST'])
def set_minimax_params(game_id):
    """Update Minimax parameters."""
    if game_id not in games:
        return jsonify({'error': 'Game not found'}), 404
    
    data = request.json or {}
    minimax = games[game_id]['minimax']
    
    # Update only provided params
    minimax.set_params(
        depth=data.get('depth')
    )
    
    return jsonify({
        'game_id': game_id,
        'minimax_params': minimax.get_params()
    })


@app.route('/api/game/<game_id>/minimax/progress', methods=['GET'])
def get_minimax_progress(game_id):
    """Get current Minimax search progress."""
    if game_id not in games:
        return jsonify({'error': 'Game not found'}), 404
    
    progress = games[game_id].get('minimax_progress')
    if progress is None:
        return jsonify({
            'game_id': game_id,
            'progress': None
        })
    
    # Convert best_move tuple to dict if present
    best_move = progress.get('best_move')
    best_move_dict = None
    if best_move is not None:
        best_move_dict = {'row': best_move[0], 'col': best_move[1]}
    
    progress_dict = {
        'current_move': progress.get('current_move', 0),
        'total_moves': progress.get('total_moves', 0),
        'current_depth': progress.get('current_depth', 0),
        'best_value': progress.get('best_value'),
        'best_move': best_move_dict,
        'is_complete': progress.get('is_complete', False)
    }
    
    return jsonify({
        'game_id': game_id,
        'progress': progress_dict
    })


# Backward compatibility endpoints (MCTS -> Minimax)
@app.route('/api/game/<game_id>/mcts', methods=['POST'])
def mcts_move(game_id):
    """Backward compatibility: Get Minimax move recommendation (was MCTS)."""
    return minimax_move(game_id)


@app.route('/api/game/<game_id>/mcts/play', methods=['POST'])
def mcts_play(game_id):
    """Backward compatibility: Make Minimax play a move (was MCTS)."""
    return minimax_play(game_id)


@app.route('/api/game/<game_id>/mcts/params', methods=['GET'])
def get_mcts_params(game_id):
    """Backward compatibility: Get Minimax parameters (was MCTS)."""
    return get_minimax_params(game_id)


@app.route('/api/game/<game_id>/mcts/params', methods=['POST'])
def set_mcts_params(game_id):
    """Backward compatibility: Update Minimax parameters (was MCTS)."""
    return set_minimax_params(game_id)


if __name__ == '__main__':
    app.run(debug=True, port=5001)

