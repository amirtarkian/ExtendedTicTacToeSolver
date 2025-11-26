"""
Minimax algorithm implementation with alpha-beta pruning.
Uses a heuristic value function that evaluates lines and diagonals with open space.
"""

from typing import Optional, Tuple, List, Callable
from game import GameState, Player
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Minimax:
    """
    Minimax algorithm with alpha-beta pruning for infinite tic-tac-toe.
    Uses a heuristic value function that values lines/diagonals with open space.
    """
    
    def __init__(self, depth: int = 4):
        """
        Initialize Minimax.
        
        Args:
            depth: Maximum search depth (default: 4)
        """
        self.depth = depth
        logger.info(f"Minimax initialized with depth={depth}")
    
    def get_params(self) -> dict:
        """Get current Minimax parameters."""
        return {
            'depth': self.depth
        }
    
    def set_params(self, depth: int = None):
        """Update Minimax parameters."""
        if depth is not None:
            self.depth = depth
    
    def search(self, state: GameState, progress_callback: Optional[Callable] = None) -> Tuple[int, int]:
        """
        Run minimax search with alpha-beta pruning to find the best move.
        
        Args:
            state: Current game state
            progress_callback: Optional callback function to report progress updates.
                             Called with dict containing: current_move, total_moves, 
                             current_depth, best_value, best_move
            
        Returns:
            Best move as (row, col) tuple
        """
        logger.info(f"Starting minimax search with depth={self.depth}")
        
        legal_moves = list(state.get_legal_moves())
        if not legal_moves:
            raise ValueError("No legal moves available")
        
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        total_moves = len(legal_moves)
        best_move = None
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        # Try each legal move and find the one with highest minimax value
        # Use alpha-beta pruning at root level for efficiency
        for move_idx, move in enumerate(legal_moves):
            # Report progress at root level
            if progress_callback:
                progress_callback({
                    'current_move': move_idx + 1,
                    'total_moves': total_moves,
                    'current_depth': self.depth,
                    'best_value': best_value if best_move is not None else None,
                    'best_move': best_move
                })
            
            new_state = state.make_move(*move)
            value = self._minimax(new_state, self.depth - 1, alpha, beta, False, progress_callback, move_idx + 1, total_moves)
            
            if value > best_value:
                best_value = value
                best_move = move
            
            # Update alpha for alpha-beta pruning
            alpha = max(alpha, value)
            if beta <= alpha:
                # Pruning: opponent won't choose this branch
                break
        
        if best_move is None:
            # Fallback: return first legal move
            logger.warning("No best move found, returning first legal move")
            return legal_moves[0]
        
        logger.info(f"Best move selected: {best_move} (value: {best_value:.3f})")
        return best_move
    
    def _minimax(self, state: GameState, depth: int, alpha: float, beta: float, maximizing: bool,
                 progress_callback: Optional[Callable] = None, root_move_idx: int = 0, total_root_moves: int = 0) -> float:
        """
        Recursive minimax with alpha-beta pruning.
        
        Args:
            state: Current game state
            depth: Remaining search depth
            alpha: Alpha value for alpha-beta pruning
            beta: Beta value for alpha-beta pruning
            maximizing: True if maximizing player, False if minimizing
            progress_callback: Optional callback function to report progress
            root_move_idx: Current move index at root level (for progress reporting)
            total_root_moves: Total moves at root level (for progress reporting)
            
        Returns:
            Minimax value of the position
        """
        # Terminal state check
        if state.is_terminal():
            winner = state.get_winner()
            if winner is None:
                # Draw
                return 0.0
            # X win = positive infinity, O win = negative infinity
            if winner == Player.X:
                return float('inf')
            else:  # winner == Player.O
                return float('-inf')
        
        # Depth limit reached - use heuristic evaluation
        if depth == 0:
            return self._evaluate(state, state.current_player.other())
        
        legal_moves = list(state.get_legal_moves())
        if not legal_moves:
            return 0.0
        
        if maximizing:
            max_value = float('-inf')
            for move in legal_moves:
                new_state = state.make_move(*move)
                value = self._minimax(new_state, depth - 1, alpha, beta, False, progress_callback, root_move_idx, total_root_moves)
                max_value = max(max_value, value)
                alpha = max(alpha, value)
                if beta <= alpha:
                    break  # Alpha-beta pruning
            return max_value
        else:
            min_value = float('inf')
            for move in legal_moves:
                new_state = state.make_move(*move)
                value = self._minimax(new_state, depth - 1, alpha, beta, True, progress_callback, root_move_idx, total_root_moves)
                min_value = min(min_value, value)
                beta = min(beta, value)
                if beta <= alpha:
                    break  # Alpha-beta pruning
            return min_value
    
    def _evaluate(self, state: GameState, player: Player) -> float:
        """
        Evaluate a game state using a heuristic that values lines/diagonals with open space.
        
        Args:
            state: Game state to evaluate
            player: Player to evaluate for (from whose perspective to score)
            
        Returns:
            Evaluation score (positive favors player, negative favors opponent)
        """
        opponent = player.other()
        score = 0.0
        
        # Directions: horizontal, vertical, diagonal \, diagonal /
        directions = [
            (0, 1),   # Horizontal
            (1, 0),   # Vertical
            (1, 1),   # Diagonal \
            (1, -1),  # Diagonal /
        ]
        
        # Get all positions for both players
        player_positions = {pos for pos, p in state.board.items() if p == player}
        opponent_positions = {pos for pos, p in state.board.items() if p == opponent}
        
        # Evaluate lines for the player
        for pos in player_positions:
            r, c = pos
            for dr, dc in directions:
                line_score = self._evaluate_line(state, r, c, dr, dc, player)
                score += line_score
        
        # Evaluate lines for the opponent (subtract their threats)
        for pos in opponent_positions:
            r, c = pos
            for dr, dc in directions:
                line_score = self._evaluate_line(state, r, c, dr, dc, opponent)
                score -= line_score
        
        return score
    
    def _evaluate_line(self, state: GameState, start_row: int, start_col: int,
                       delta_row: int, delta_col: int, player: Player) -> float:
        """
        Evaluate a line starting from a position in a given direction.
        Values longer lines with open space more highly.
        
        Args:
            state: Game state
            start_row: Starting row
            start_col: Starting column
            delta_row: Row direction
            delta_col: Column direction
            player: Player to evaluate for
            
        Returns:
            Score for this line
        """
        # Count consecutive pieces going forward
        forward_count = 0
        forward_pos = (start_row, start_col)
        for i in range(5):
            r = start_row + i * delta_row
            c = start_col + i * delta_col
            if state.get(r, c) == player:
                forward_count += 1
            else:
                break
        
        # Count consecutive pieces going backward
        backward_count = 0
        for i in range(1, 5):
            r = start_row - i * delta_row
            c = start_col - i * delta_col
            if state.get(r, c) == player:
                backward_count += 1
            else:
                break
        
        # Total consecutive pieces in this line
        total_count = forward_count + backward_count
        
        if total_count == 0:
            return 0.0
        
        # Check for open space on both ends
        # Forward end
        forward_end_r = start_row + forward_count * delta_row
        forward_end_c = start_col + forward_count * delta_col
        forward_open = state.get(forward_end_r, forward_end_c) is None
        
        # Backward end
        backward_end_r = start_row - backward_count * delta_row
        backward_end_c = start_col - backward_count * delta_col
        backward_open = state.get(backward_end_r, backward_end_c) is None
        
        # Score based on line length and open space
        # Longer lines with open space are more valuable
        base_score = 0.0
        
        if total_count >= 4:
            # 4-in-a-row: very threatening (almost winning)
            base_score = 1000.0
        elif total_count == 3:
            # 3-in-a-row: strong threat
            base_score = 100.0
        elif total_count == 2:
            # 2-in-a-row: developing threat
            base_score = 10.0
        elif total_count == 1:
            # Single piece: minimal value
            base_score = 1.0
        
        # Multiply by number of open ends (can extend the line)
        open_ends = sum([forward_open, backward_open])
        if open_ends > 0:
            return base_score * (1 + 0.5 * open_ends)
        else:
            # No open space - line is blocked
            return base_score * 0.1

