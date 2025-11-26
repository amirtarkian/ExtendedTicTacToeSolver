"""
Infinite Tic-Tac-Toe Game with 5-in-a-row win condition.
Designed to be compatible with Monte Carlo Tree Search (MCTS).
"""

from typing import Set, Tuple, Optional, List
from copy import deepcopy
from enum import Enum


class Player(Enum):
    """Represents the two players in the game."""
    X = 1
    O = 2
    
    def other(self) -> 'Player':
        """Returns the other player."""
        return Player.O if self == Player.X else Player.X
    
    def __str__(self) -> str:
        return self.name


class GameState:
    """
    Represents the state of an infinite tic-tac-toe game.
    Uses a sparse dictionary representation for the board.
    """
    
    def __init__(self, board: Optional[dict] = None, current_player: Player = Player.X):
        """
        Initialize game state.
        
        Args:
            board: Dictionary mapping (row, col) tuples to Player values.
                   If None, creates an empty board.
            current_player: The player whose turn it is.
        """
        self.board = board if board is not None else {}
        self.current_player = current_player
        self._winner: Optional[Player] = None
        self._cached_legal_moves: Optional[Set[Tuple[int, int]]] = None
    
    def get(self, row: int, col: int) -> Optional[Player]:
        """Get the player at the given position, or None if empty."""
        return self.board.get((row, col))
    
    def make_move(self, row: int, col: int) -> 'GameState':
        """
        Make a move at the given position.
        Returns a new GameState without modifying the current one.
        
        Args:
            row: Row coordinate
            col: Column coordinate
            
        Returns:
            New GameState after the move
        """
        if (row, col) in self.board:
            raise ValueError(f"Position ({row}, {col}) is already occupied")
        
        new_board = self.board.copy()
        new_board[(row, col)] = self.current_player
        
        new_state = GameState(new_board, self.current_player.other())
        return new_state
    
    def get_legal_moves(self) -> Set[Tuple[int, int]]:
        """
        Get all legal moves (empty positions within 3 squares of played positions).
        For an infinite board, we only consider positions near existing moves.
        
        Returns:
            Set of (row, col) tuples representing legal moves
        """
        if self._cached_legal_moves is not None:
            return self._cached_legal_moves
        
        if not self.board:
            # First move - return center position
            legal_moves = {(0, 0)}
        else:
            legal_moves = set()
            # Consider all positions within 3 squares of existing moves
            for (r, c) in self.board.keys():
                for dr in range(-3, 4):
                    for dc in range(-3, 4):
                        # Skip the position itself
                        if dr == 0 and dc == 0:
                            continue
                        # Check if within 3 squares (Manhattan distance)
                        if abs(dr) + abs(dc) <= 3:
                            new_pos = (r + dr, c + dc)
                            if new_pos not in self.board:
                                legal_moves.add(new_pos)
        
        self._cached_legal_moves = legal_moves
        return legal_moves
    
    def get_legal_moves_within_bounds(self) -> Set[Tuple[int, int]]:
        """
        Get legal moves only within the existing board bounds (no expansion).
        Used during MCTS rollout to prevent board expansion.
        
        Returns:
            Set of (row, col) tuples representing legal moves within bounds
        """
        if not self.board:
            # First move - return center position
            return {(0, 0)}
        
        # Get board bounds
        rows = [r for r, _ in self.board.keys()]
        cols = [c for _, c in self.board.keys()]
        
        if not rows or not cols:
            return set()
        
        min_row, max_row = min(rows), max(rows)
        min_col, max_col = min(cols), max(cols)
        
        legal_moves = set()
        # Only consider positions within existing bounds
        for r in range(min_row, max_row + 1):
            for c in range(min_col, max_col + 1):
                pos = (r, c)
                if pos not in self.board:
                    legal_moves.add(pos)
        
        return legal_moves
    
    def _min_distance_to_token(self, row: int, col: int) -> int:
        """
        Calculate minimum Manhattan distance from a position to any existing token.
        
        Args:
            row: Row coordinate
            col: Column coordinate
            
        Returns:
            Minimum Manhattan distance to nearest token, or float('inf') if no tokens
        """
        if not self.board:
            return float('inf')
        
        min_dist = float('inf')
        for (r, c) in self.board.keys():
            dist = abs(row - r) + abs(col - c)
            min_dist = min(min_dist, dist)
        
        return min_dist
    
    def _check_line(self, start_row: int, start_col: int, 
                    delta_row: int, delta_col: int, player: Player) -> bool:
        """
        Check if there are 5 consecutive pieces in a line starting from a position.
        
        Args:
            start_row: Starting row
            start_col: Starting column
            delta_row: Row direction (e.g., 0 for horizontal, 1 for diagonal)
            delta_col: Column direction (e.g., 1 for horizontal, 1 for diagonal)
            player: Player to check for
            
        Returns:
            True if 5 in a row found
        """
        count = 0
        for i in range(5):
            r = start_row + i * delta_row
            c = start_col + i * delta_col
            if self.get(r, c) == player:
                count += 1
            else:
                break
        
        return count >= 5
    
    def _has_win(self, player: Player) -> bool:
        """
        Check if the given player has 5 in a row anywhere on the board.
        
        Args:
            player: Player to check for
            
        Returns:
            True if player has 5 in a row
        """
        if not self.board:
            return False
        
        # Get all positions for this player
        player_positions = {pos for pos, p in self.board.items() if p == player}
        
        if len(player_positions) < 5:
            return False
        
        directions = [
            (0, 1),   # Horizontal
            (1, 0),   # Vertical
            (1, 1),   # Diagonal \
            (1, -1),  # Diagonal /
        ]
        
        # Check each position as a potential start of a line
        for (r, c) in player_positions:
            for dr, dc in directions:
                # Check if there's a line of 5 starting from this position
                if self._check_line(r, c, dr, dc, player):
                    return True
        
        return False
    
    def is_terminal(self) -> bool:
        """
        Check if the game is in a terminal state (win or draw).
        
        Returns:
            True if the game is over
        """
        if self._winner is not None:
            return True
        
        # Check if the last move (by the previous player) resulted in a win
        if not self.board:
            return False
        
        # The current player is the one who just moved
        previous_player = self.current_player.other()
        
        # Check if the previous player has won
        if self._has_win(previous_player):
            self._winner = previous_player
            return True
        
        # Check for draw (no legal moves)
        if len(self.get_legal_moves()) == 0:
            return True
        
        return False
    
    def get_result(self) -> Optional[float]:
        """
        Get the result from the perspective of the current player.
        Used by MCTS.
        
        Returns:
            1.0 if current player wins,
            -1.0 if current player loses,
            0.0 if draw,
            None if game not terminal
        """
        if not self.is_terminal():
            return None
        
        if self._winner is None:
            return 0.0  # Draw
        
        # From current player's perspective
        if self._winner == self.current_player:
            return 1.0
        else:
            return -1.0
    
    def get_winner(self) -> Optional[Player]:
        """Get the winner of the game, or None if no winner yet."""
        if self.is_terminal():
            return self._winner
        return None
    
    def copy(self) -> 'GameState':
        """Create a deep copy of the game state."""
        return GameState(deepcopy(self.board), self.current_player)
    
    def __str__(self) -> str:
        """String representation of the board."""
        if not self.board:
            return "Empty board"
        
        rows = [r for r, _ in self.board.keys()]
        cols = [c for _, c in self.board.keys()]
        
        min_row, max_row = min(rows), max(rows)
        min_col, max_col = min(cols), max(cols)
        
        lines = []
        lines.append(f"Current player: {self.current_player}")
        lines.append(f"Board bounds: rows [{min_row}, {max_row}], cols [{min_col}, {max_col}]")
        lines.append("")
        
        for r in range(min_row, max_row + 1):
            line = []
            for c in range(min_col, max_col + 1):
                player = self.get(r, c)
                if player == Player.X:
                    line.append("X")
                elif player == Player.O:
                    line.append("O")
                else:
                    line.append(".")
            lines.append(" ".join(line))
        
        return "\n".join(lines)
    
    def __eq__(self, other):
        """Check equality of game states."""
        if not isinstance(other, GameState):
            return False
        return (self.board == other.board and 
                self.current_player == other.current_player)
    
    def __hash__(self):
        """Hash for game state (for use in sets/dicts)."""
        return hash((frozenset(self.board.items()), self.current_player))

