"""
Minimax algorithm implementation with alpha-beta pruning.
Uses a heuristic value function that evaluates lines and diagonals with open space.
Optimized with transposition tables, move ordering, iterative deepening, and caching.
"""

from typing import Optional, Tuple, List, Callable, Dict
from game import GameState, Player
import logging
from collections import OrderedDict
from enum import IntEnum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TTFlag(IntEnum):
    """Transposition table entry flags."""
    EXACT = 0      # Exact value
    LOWER = 1      # Lower bound (alpha cutoff)
    UPPER = 2      # Upper bound (beta cutoff)


class TranspositionTable:
    """LRU cache for storing evaluated game positions."""
    
    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self.table: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: int) -> Optional[Tuple[float, int, TTFlag, Optional[Tuple[int, int]]]]:
        """Get cached entry: (value, depth, flag, best_move)."""
        if key in self.table:
            self.hits += 1
            self.table.move_to_end(key)
            return self.table[key]
        self.misses += 1
        return None
    
    def put(self, key: int, value: float, depth: int, flag: TTFlag, best_move: Optional[Tuple[int, int]] = None):
        """Store entry in cache."""
        if key in self.table:
            self.table.move_to_end(key)
        self.table[key] = (value, depth, flag, best_move)
        if len(self.table) > self.max_size:
            self.table.popitem(last=False)
    
    def clear(self):
        """Clear the transposition table."""
        self.table.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {'size': len(self.table), 'hits': self.hits, 'misses': self.misses, 'hit_rate': f'{hit_rate:.1%}'}


class Minimax:
    """
    Minimax algorithm with alpha-beta pruning for infinite tic-tac-toe.
    Uses a heuristic value function that values lines/diagonals with open space.
    """
    
    def __init__(self, depth: int = 4, use_iterative_deepening: bool = True, tt_size: int = 100000, adaptive_depth: bool = True):
        """
        Initialize Minimax.
        
        Args:
            depth: Maximum search depth (default: 4)
            use_iterative_deepening: Whether to use iterative deepening (default: True)
            tt_size: Transposition table size (default: 100000)
            adaptive_depth: Whether to use deeper search in early game (default: True)
        """
        self.depth = depth
        self.use_iterative_deepening = use_iterative_deepening
        self.adaptive_depth = adaptive_depth
        self.tt = TranspositionTable(max_size=tt_size)
        self.eval_cache: Dict[int, float] = {}
        self.killer_moves: Dict[int, List[Tuple[int, int]]] = {}  # killer_moves[depth] = [move1, move2]
        logger.info(f"Minimax initialized with depth={depth}, iterative_deepening={use_iterative_deepening}, adaptive_depth={adaptive_depth}")
    
    def get_params(self) -> dict:
        """Get current Minimax parameters."""
        return {
            'depth': self.depth,
            'use_iterative_deepening': self.use_iterative_deepening,
            'adaptive_depth': self.adaptive_depth,
            'tt_stats': self.tt.get_stats()
        }
    
    def set_params(self, depth: int = None, use_iterative_deepening: bool = None, adaptive_depth: bool = None):
        """Update Minimax parameters."""
        if depth is not None:
            self.depth = depth
        if use_iterative_deepening is not None:
            self.use_iterative_deepening = use_iterative_deepening
        if adaptive_depth is not None:
            self.adaptive_depth = adaptive_depth
    
    def _get_effective_depth(self, state: GameState) -> int:
        """
        Calculate effective search depth based on game stage.
        Early game = slightly deeper search (fewer legal moves).
        """
        if not self.adaptive_depth:
            return self.depth
        
        num_pieces = len(state.board)
        
        # More conservative depth boosts
        if num_pieces <= 4:
            effective_depth = self.depth + 2
        elif num_pieces <= 10:
            effective_depth = self.depth + 1
        else:
            effective_depth = self.depth
        
        # Cap at reasonable maximum
        return min(effective_depth, 7)
    
    def clear_caches(self):
        """Clear all caches (transposition table and evaluation cache)."""
        self.tt.clear()
        self.eval_cache.clear()
        self.killer_moves.clear()
    
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
        effective_depth = self._get_effective_depth(state)
        num_pieces = len(state.board)
        logger.info(f"Starting minimax search with depth={effective_depth} (base={self.depth}, pieces={num_pieces})")
        
        legal_moves = list(state.get_legal_moves())
        if not legal_moves:
            raise ValueError("No legal moves available")
        
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        # Early win/loss detection with urgency-based threat analysis
        threats = self._find_urgent_threats(state)
        
        # Check for immediate win (urgency 3)
        for move, urgency in threats:
            if urgency == 3:
                logger.info(f"Immediate winning move found: {move}")
                return move
        
        # Check for must-block threats (urgency 2 = 4-in-a-row threat)
        must_block = [m for m, u in threats if u == 2]
        if must_block:
            logger.info(f"Must block opponent 4-in-a-row threat: {must_block[0]}")
            return must_block[0]
        
        # Log if there are open-3 threats to consider
        open_3_threats = [m for m, u in threats if u == 1]
        if open_3_threats:
            logger.info(f"Found {len(open_3_threats)} open 3-in-a-row threats to consider")
        
        # Clear killer moves for new search
        self.killer_moves.clear()
        
        if self.use_iterative_deepening:
            return self._iterative_deepening_search(state, effective_depth, progress_callback)
        else:
            return self._fixed_depth_search(state, effective_depth, progress_callback)
    
    def _iterative_deepening_search(self, state: GameState, max_depth: int, progress_callback: Optional[Callable] = None) -> Tuple[int, int]:
        """Search with iterative deepening for better move ordering."""
        best_move = None
        
        for current_depth in range(1, max_depth + 1):
            move = self._fixed_depth_search(state, current_depth, progress_callback)
            if move:
                best_move = move
        
        return best_move
    
    def _fixed_depth_search(self, state: GameState, depth: int, progress_callback: Optional[Callable] = None) -> Tuple[int, int]:
        """Search to a fixed depth."""
        legal_moves = list(state.get_legal_moves())
        total_moves = len(legal_moves)
        
        # Order moves for better pruning
        ordered_moves = self._order_moves(state, legal_moves, depth)
        
        best_move = None
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        for move_idx, move in enumerate(ordered_moves):
            if progress_callback:
                progress_callback({
                    'current_move': move_idx + 1,
                    'total_moves': total_moves,
                    'current_depth': depth,
                    'best_value': best_value if best_move is not None else None,
                    'best_move': best_move
                })
            
            new_state = state.make_move(*move)
            
            # Principal Variation Search: search first move with full window
            if move_idx == 0:
                value = self._minimax(new_state, depth - 1, alpha, beta, False)
            else:
                # Null window search for remaining moves
                value = self._minimax(new_state, depth - 1, alpha, alpha + 1, False)
                if alpha < value < beta:
                    # Re-search with full window if null window fails
                    value = self._minimax(new_state, depth - 1, alpha, beta, False)
            
            if value > best_value:
                best_value = value
                best_move = move
            
            alpha = max(alpha, value)
            if beta <= alpha:
                break
        
        if best_move is None:
            logger.warning("No best move found, returning first legal move")
            return legal_moves[0]
        
        logger.info(f"Depth {depth}: Best move {best_move} (value: {best_value:.3f})")
        return best_move
    
    def _minimax(self, state: GameState, depth: int, alpha: float, beta: float, maximizing: bool) -> float:
        """
        Recursive minimax with alpha-beta pruning and transposition table.
        
        Args:
            state: Current game state
            depth: Remaining search depth
            alpha: Alpha value for alpha-beta pruning
            beta: Beta value for alpha-beta pruning
            maximizing: True if maximizing player, False if minimizing
            
        Returns:
            Minimax value of the position
        """
        alpha_orig = alpha
        
        # Transposition table lookup
        state_hash = hash(state)
        tt_entry = self.tt.get(state_hash)
        if tt_entry is not None:
            tt_value, tt_depth, tt_flag, tt_best_move = tt_entry
            if tt_depth >= depth:
                if tt_flag == TTFlag.EXACT:
                    return tt_value
                elif tt_flag == TTFlag.LOWER:
                    alpha = max(alpha, tt_value)
                elif tt_flag == TTFlag.UPPER:
                    beta = min(beta, tt_value)
                if alpha >= beta:
                    return tt_value
        
        # Terminal state check
        if state.is_terminal():
            winner = state.get_winner()
            if winner is None:
                return 0.0
            # X win = positive, O win = negative (large but finite for comparison)
            if winner == Player.X:
                return 100000.0 + depth  # Prefer faster wins
            else:
                return -100000.0 - depth
        
        # Depth limit reached - use cached heuristic evaluation
        if depth == 0:
            return self._evaluate_cached(state)
        
        legal_moves = list(state.get_legal_moves())
        if not legal_moves:
            return 0.0
        
        # Limit moves at deeper search levels to reduce branching factor
        # At shallow depth (close to leaves), only consider best moves
        move_limit = 0  # 0 means no limit
        if depth <= 2:
            move_limit = 12  # Only top 12 moves at shallow depths
        elif depth <= 3:
            move_limit = 18  # Top 18 moves at medium depths
        
        # Order moves for better pruning
        ordered_moves = self._order_moves(state, legal_moves, depth, limit=move_limit)
        
        best_move = None
        if maximizing:
            value = float('-inf')
            for move in ordered_moves:
                new_state = state.make_move(*move)
                child_value = self._minimax(new_state, depth - 1, alpha, beta, False)
                if child_value > value:
                    value = child_value
                    best_move = move
                alpha = max(alpha, value)
                if beta <= alpha:
                    self._record_killer(depth, move)
                    break
        else:
            value = float('inf')
            for move in ordered_moves:
                new_state = state.make_move(*move)
                child_value = self._minimax(new_state, depth - 1, alpha, beta, True)
                if child_value < value:
                    value = child_value
                    best_move = move
                beta = min(beta, value)
                if beta <= alpha:
                    self._record_killer(depth, move)
                    break
        
        # Store in transposition table
        if value <= alpha_orig:
            flag = TTFlag.UPPER
        elif value >= beta:
            flag = TTFlag.LOWER
        else:
            flag = TTFlag.EXACT
        self.tt.put(state_hash, value, depth, flag, best_move)
        
        return value
    
    def _record_killer(self, depth: int, move: Tuple[int, int]):
        """Record a killer move (caused beta cutoff) at this depth."""
        if depth not in self.killer_moves:
            self.killer_moves[depth] = []
        killers = self.killer_moves[depth]
        if move not in killers:
            killers.insert(0, move)
            if len(killers) > 2:
                killers.pop()
    
    def _evaluate_cached(self, state: GameState) -> float:
        """Cached evaluation to avoid redundant calculations."""
        state_hash = hash(state)
        if state_hash in self.eval_cache:
            return self.eval_cache[state_hash]
        score = self._evaluate(state, state.current_player.other())
        # Add early game positional bonus
        score += self._early_game_bonus(state, state.current_player.other())
        self.eval_cache[state_hash] = score
        return score
    
    def _early_game_bonus(self, state: GameState, player: Player) -> float:
        """
        Fast early game positional bonus.
        Prioritizes center control.
        """
        num_pieces = len(state.board)
        if num_pieces > 10:
            return 0.0  # Only apply in early game
        
        bonus = 0.0
        for (r, c), p in state.board.items():
            # Center control bonus - being near (0,0) is valuable
            distance = abs(r) + abs(c)
            value = max(0, 4 - distance) * 3.0
            if p == player:
                bonus += value
            else:
                bonus -= value
        
        return bonus
    
    def _order_moves(self, state: GameState, moves: List[Tuple[int, int]], depth: int, limit: int = 0) -> List[Tuple[int, int]]:
        """
        Order moves for better alpha-beta pruning.
        Priority: TT best > wins > blocks > killer moves > threats > center > near pieces.
        
        Args:
            limit: If > 0, only return top N moves (reduces branching factor)
        """
        move_scores = []
        state_hash = hash(state)
        
        # Get TT best move if available
        tt_entry = self.tt.get(state_hash)
        tt_best = tt_entry[3] if tt_entry else None
        
        # Get killer moves for this depth
        killers = self.killer_moves.get(depth, [])
        
        # Pre-compute threat scores for all moves
        threat_map = {}
        for move in moves:
            threat_map[move] = self._threat_score_for_ordering(state, move)
        
        for move in moves:
            score = 0.0
            r, c = move
            
            # Highest priority: TT best move
            if move == tt_best:
                score += 100000
            
            # Very high priority: winning moves and blocking moves
            our_threat, opp_threat = threat_map[move]
            if our_threat >= 4:
                score += 50000  # Winning move
            if opp_threat >= 4:
                score += 40000  # Must block
            if our_threat >= 3:
                score += 8000   # Creating strong threat
            if opp_threat >= 3:
                score += 7000   # Blocking opponent's threat
            
            # High priority: killer moves
            if move in killers:
                score += 5000 - killers.index(move) * 100
            
            # Medium priority: smaller threats
            score += our_threat * 200
            score += opp_threat * 150  # Also value defensive moves
            
            # Prefer center moves (lower distance = higher score)
            distance_from_center = abs(r) + abs(c)
            score -= distance_from_center * 10
            
            # Prefer moves near existing pieces
            for (br, bc) in state.board.keys():
                dist = abs(r - br) + abs(c - bc)
                if dist <= 2:
                    score += (3 - dist) * 20
            
            move_scores.append((score, move))
        
        move_scores.sort(reverse=True, key=lambda x: x[0])
        ordered = [move for _, move in move_scores]
        
        # Limit moves if requested (prune low-value moves)
        if limit > 0 and len(ordered) > limit:
            return ordered[:limit]
        return ordered
    
    def _threat_score_for_ordering(self, state: GameState, move: Tuple[int, int]) -> Tuple[int, int]:
        """
        Calculate threat scores for move ordering.
        Returns (our_threat_score, opponent_threat_score).
        """
        r, c = move
        current = state.current_player
        opponent = current.other()
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        max_our = 0
        max_opp = 0
        
        for dr, dc in directions:
            # Count our pieces
            our_count = 0
            for sign in [1, -1]:
                for i in range(1, 5):
                    if state.get(r + sign*i*dr, c + sign*i*dc) == current:
                        our_count += 1
                    else:
                        break
            max_our = max(max_our, our_count)
            
            # Count opponent pieces
            opp_count = 0
            for sign in [1, -1]:
                for i in range(1, 5):
                    if state.get(r + sign*i*dr, c + sign*i*dc) == opponent:
                        opp_count += 1
                    else:
                        break
            max_opp = max(max_opp, opp_count)
        
        return (max_our, max_opp)
    
    def _find_immediate_win(self, state: GameState) -> Optional[Tuple[int, int]]:
        """Check if current player can win immediately."""
        current_player = state.current_player
        for move in state.get_legal_moves():
            test_state = state.make_move(*move)
            if test_state.is_terminal() and test_state.get_winner() == current_player:
                return move
        return None
    
    def _find_immediate_block(self, state: GameState) -> Optional[Tuple[int, int]]:
        """Check if opponent can win next turn and return a blocking move."""
        opponent = state.current_player.other()
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        # Check all legal moves for blocking 4-in-a-row threats
        for move in state.get_legal_moves():
            r, c = move
            for dr, dc in directions:
                # Count opponent pieces on both sides of this position
                count_pos = 0
                count_neg = 0
                for i in range(1, 5):
                    nr, nc = r + i * dr, c + i * dc
                    if state.get(nr, nc) == opponent:
                        count_pos += 1
                    else:
                        break
                for i in range(1, 5):
                    nr, nc = r - i * dr, c - i * dc
                    if state.get(nr, nc) == opponent:
                        count_neg += 1
                    else:
                        break
                if count_pos + count_neg >= 4:
                    return move
        return None
    
    def _find_urgent_threats(self, state: GameState) -> List[Tuple[Tuple[int, int], int]]:
        """
        Find all urgent threats that need to be addressed.
        Returns list of (move, urgency) where higher urgency = more critical.
        Urgency levels: 3=win, 2=block4, 1=block open 3
        """
        threats = []
        current = state.current_player
        opponent = current.other()
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for move in state.get_legal_moves():
            r, c = move
            max_urgency = 0
            
            for dr, dc in directions:
                # Count our pieces (potential win)
                our_count_pos, our_count_neg = 0, 0
                for i in range(1, 5):
                    if state.get(r + i*dr, c + i*dc) == current:
                        our_count_pos += 1
                    else:
                        break
                for i in range(1, 5):
                    if state.get(r - i*dr, c - i*dc) == current:
                        our_count_neg += 1
                    else:
                        break
                
                if our_count_pos + our_count_neg >= 4:
                    max_urgency = max(max_urgency, 3)  # Win!
                
                # Count opponent pieces (need to block)
                opp_count_pos, opp_count_neg = 0, 0
                open_pos, open_neg = False, False
                for i in range(1, 5):
                    cell = state.get(r + i*dr, c + i*dc)
                    if cell == opponent:
                        opp_count_pos += 1
                    elif cell is None:
                        open_pos = True
                        break
                    else:
                        break
                for i in range(1, 5):
                    cell = state.get(r - i*dr, c - i*dc)
                    if cell == opponent:
                        opp_count_neg += 1
                    elif cell is None:
                        open_neg = True
                        break
                    else:
                        break
                
                total_opp = opp_count_pos + opp_count_neg
                if total_opp >= 4:
                    max_urgency = max(max_urgency, 2)  # Must block 4-in-a-row
                elif total_opp >= 3 and (open_pos or open_neg):
                    max_urgency = max(max_urgency, 1)  # Should block open 3-in-a-row
            
            if max_urgency > 0:
                threats.append((move, max_urgency))
        
        # Sort by urgency (highest first)
        threats.sort(key=lambda x: -x[1])
        return threats
    
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

