"""
Monte Carlo Tree Search (MCTS) implementation with MPS acceleration.
Uses Apple's Metal Performance Shaders for GPU-accelerated simulations.
"""

from typing import Optional, List, Tuple
from game import GameState, Player
import math
import random
from tqdm import tqdm
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import MPS accelerator
try:
    from mps_accelerator import MPSAccelerator, create_accelerator
    MPS_AVAILABLE = True
except ImportError:
    MPS_AVAILABLE = False
    logger.warning("MPS accelerator not available - install PyTorch for GPU acceleration")


class MCTSNode:
    """
    A node in the Monte Carlo Tree Search tree.
    Each node represents a game state.
    """
    
    def __init__(self, state: GameState, parent: Optional['MCTSNode'] = None, 
                 move: Optional[Tuple[int, int]] = None):
        """
        Initialize an MCTS node.
        
        Args:
            state: The game state at this node
            parent: Parent node (None for root)
            move: The move that led to this state (None for root)
        """
        self.state = state
        self.parent = parent
        self.move = move
        self.children: List['MCTSNode'] = []
        self.visits = 0
        self.wins = 0.0
        self.untried_moves = set(state.get_legal_moves())
    
    def is_fully_expanded(self) -> bool:
        """Check if all legal moves have been tried."""
        return len(self.untried_moves) == 0
    
    def is_terminal(self) -> bool:
        """Check if this node represents a terminal state."""
        return self.state.is_terminal()
    
    def ucb1_value(self, exploration_constant: float = 1.414) -> float:
        """
        Calculate UCB1 value for this node.
        Only valid if node has been visited.
        
        Args:
            exploration_constant: Exploration parameter (default sqrt(2))
            
        Returns:
            UCB1 value
        """
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.wins / self.visits
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration
    
    def select_child(self) -> 'MCTSNode':
        """
        Select the best child according to UCB1.
        
        Returns:
            Best child node
        """
        return max(self.children, key=lambda child: child.ucb1_value())
    
    def expand(self) -> 'MCTSNode':
        """
        Expand the tree by adding a new child node.
        
        Returns:
            New child node
        """
        move = self.untried_moves.pop()
        new_state = self.state.make_move(*move)
        child = MCTSNode(new_state, parent=self, move=move)
        self.children.append(child)
        return child
    
    def update(self, result: float):
        """
        Update node statistics after a simulation.
        
        Args:
            result: Result from the perspective of the node's player
        """
        self.visits += 1
        self.wins += result


class MCTS:
    """
    Monte Carlo Tree Search algorithm with MPS acceleration.
    Uses Apple's Metal Performance Shaders for GPU-accelerated batch simulations.
    """
    
    def __init__(self, exploration_constant: float = 1.414, simulations: int = 1000,
                 max_simulation_depth: int = 1000, use_mps: bool = True, 
                 batch_size: int = 32):
        """
        Initialize MCTS.
        
        Args:
            exploration_constant: UCB1 exploration parameter (higher = more exploration)
            simulations: Number of simulations per move
            max_simulation_depth: Maximum depth for random rollouts
            use_mps: Whether to use MPS acceleration (requires PyTorch and Apple Silicon)
            batch_size: Number of simulations to batch together for MPS acceleration
        """
        self.exploration_constant = exploration_constant
        self.simulations = simulations
        self.max_simulation_depth = max_simulation_depth
        self.use_mps = use_mps and MPS_AVAILABLE
        self.batch_size = batch_size
        
        # Initialize MPS accelerator if available
        if self.use_mps:
            try:
                self.accelerator = create_accelerator(batch_size=batch_size, use_mps=True)
                logger.info(f"MCTS initialized with MPS acceleration (batch_size={batch_size})")
                # Verify MPS is actually being used
                if self.accelerator:
                    status = self.accelerator.verify_mps_usage()
                    logger.info(f"MPS verification: {status['verification']}")
                    logger.info(f"  Device: {status['device']}, Tensor on MPS: {status.get('tensor_on_mps', False)}")
                    logger.info(f"  PyTorch threads: {status['torch_threads']}, Interop threads: {status['torch_interop_threads']}")
            except Exception as e:
                logger.warning(f"Failed to initialize MPS accelerator: {e}. Falling back to CPU.")
                self.accelerator = None
                self.use_mps = False
        else:
            self.accelerator = None
            if use_mps and not MPS_AVAILABLE:
                logger.info("MCTS initialized without MPS (PyTorch not installed or MPS unavailable)")
            else:
                logger.info("MCTS initialized without MPS acceleration")
    
    def get_params(self) -> dict:
        """Get current MCTS parameters."""
        return {
            'exploration_constant': self.exploration_constant,
            'simulations': self.simulations,
            'max_simulation_depth': self.max_simulation_depth,
            'use_mps': self.use_mps,
            'batch_size': self.batch_size,
            'mps_available': self.accelerator.is_available() if self.accelerator else False
        }
    
    def set_params(self, exploration_constant: float = None, simulations: int = None,
                   max_simulation_depth: int = None, use_mps: bool = None,
                   batch_size: int = None):
        """Update MCTS parameters."""
        if exploration_constant is not None:
            self.exploration_constant = exploration_constant
        if simulations is not None:
            self.simulations = simulations
        if max_simulation_depth is not None:
            self.max_simulation_depth = max_simulation_depth
        if use_mps is not None and MPS_AVAILABLE:
            self.use_mps = use_mps
            if use_mps and not self.accelerator:
                try:
                    self.accelerator = create_accelerator(batch_size=self.batch_size, use_mps=True)
                except Exception as e:
                    logger.warning(f"Failed to initialize MPS accelerator: {e}")
                    self.use_mps = False
        if batch_size is not None:
            self.batch_size = batch_size
            if self.accelerator:
                self.accelerator.batch_size = batch_size
    
    def search(self, state: GameState) -> Tuple[int, int]:
        """
        Run MCTS to find the best move.
        
        Args:
            state: Current game state
            
        Returns:
            Best move as (row, col) tuple
        """
        logger.info(f"Starting MCTS search with {self.simulations} simulations")
        root = MCTSNode(state)
        
        # Track rollout outcomes
        rollout_wins = 0
        rollout_losses = 0
        rollout_draws = 0
        total_rollouts = 0
        
        # Use batched simulations if MPS accelerator is available
        if self.use_mps and self.accelerator:
            # Collect nodes for batched simulation
            nodes_to_simulate = []
            
            # Use tqdm to show progress
            with tqdm(total=self.simulations, desc="MCTS simulations (MPS)", unit="sim") as pbar:
                for sim_num in range(self.simulations):
                    # Selection
                    node = self._select(root)
                    
                    # Expansion
                    if not node.is_terminal() and not node.is_fully_expanded():
                        node = node.expand()
                    
                    # Collect node for batched simulation
                    nodes_to_simulate.append(node)
                    
                    # Run batched simulations when batch is full or at the end
                    if len(nodes_to_simulate) >= self.batch_size or sim_num == self.simulations - 1:
                        # Extract states for batch simulation
                        states_to_simulate = [n.state for n in nodes_to_simulate]
                        
                        # Run batched simulations
                        results = self.accelerator.batch_simulate(
                            states_to_simulate, 
                            self.max_simulation_depth
                        )
                        
                        # Track rollout outcomes
                        for result in results:
                            total_rollouts += 1
                            if result == 1.0:
                                rollout_wins += 1
                            elif result == -1.0:
                                rollout_losses += 1
                            else:
                                rollout_draws += 1
                        
                        # Backpropagate results
                        for node, result in zip(nodes_to_simulate, results):
                            self._backpropagate(node, result)
                        
                        # Clear batch
                        nodes_to_simulate = []
                    
                    # Update progress bar with additional info
                    if (sim_num + 1) % max(1, self.simulations // 10) == 0 or sim_num == 0:
                        if root.children:
                            best_child = max(root.children, key=lambda child: child.visits)
                            win_rate = best_child.wins / best_child.visits if best_child.visits > 0 else 0.0
                            win_pct = (rollout_wins / total_rollouts * 100) if total_rollouts > 0 else 0.0
                            pbar.set_postfix({
                                'nodes': len(root.children),
                                'best_visits': best_child.visits,
                                'win_rate': f"{win_rate:.2f}",
                                'wins%': f"{win_pct:.1f}%",
                                'mps': 'ON' if self.use_mps else 'OFF'
                            })
                    
                    pbar.update(1)
        else:
            # Standard sequential simulation (CPU fallback)
            desc = "MCTS simulations"
            if self.use_mps:
                desc += " (CPU fallback)"
            with tqdm(total=self.simulations, desc=desc, unit="sim") as pbar:
                for sim_num in range(self.simulations):
                    # Selection
                    node = self._select(root)
                    
                    # Expansion
                    if not node.is_terminal() and not node.is_fully_expanded():
                        node = node.expand()
                    
                    # Simulation
                    result = self._simulate(node.state)
                    
                    # Track rollout outcomes
                    total_rollouts += 1
                    if result == 1.0:
                        rollout_wins += 1
                    elif result == -1.0:
                        rollout_losses += 1
                    else:
                        rollout_draws += 1
                    
                    # Backpropagate
                    self._backpropagate(node, result)
                    
                    # Update progress bar with additional info
                    if (sim_num + 1) % max(1, self.simulations // 10) == 0 or sim_num == 0:
                        if root.children:
                            best_child = max(root.children, key=lambda child: child.visits)
                            win_rate = best_child.wins / best_child.visits if best_child.visits > 0 else 0.0
                            win_pct = (rollout_wins / total_rollouts * 100) if total_rollouts > 0 else 0.0
                            pbar.set_postfix({
                                'nodes': len(root.children),
                                'best_visits': best_child.visits,
                                'win_rate': f"{win_rate:.2f}",
                                'wins%': f"{win_pct:.1f}%"
                            })
                    
                    pbar.update(1)
        
        # Return best move
        if not root.children:
            # Fallback: return random legal move
            legal_moves = list(state.get_legal_moves())
            logger.warning("No children found, returning random move")
            return random.choice(legal_moves) if legal_moves else None
        
        best_child = max(root.children, key=lambda child: child.visits)
        
        # Log final rollout statistics
        if total_rollouts > 0:
            win_pct = (rollout_wins / total_rollouts * 100)
            loss_pct = (rollout_losses / total_rollouts * 100)
            draw_pct = (rollout_draws / total_rollouts * 100)
            logger.info(f"Rollout statistics: {total_rollouts} total rollouts")
            logger.info(f"  Wins: {rollout_wins} ({win_pct:.1f}%)")
            logger.info(f"  Losses: {rollout_losses} ({loss_pct:.1f}%)")
            logger.info(f"  Draws: {rollout_draws} ({draw_pct:.1f}%)")
        
        logger.info(f"Best move selected: {best_child.move} (visits: {best_child.visits}, win_rate: {best_child.wins/best_child.visits:.3f})")
        return best_child.move
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Select a node to expand using UCB1.
        
        Args:
            node: Starting node
            
        Returns:
            Selected node
        """
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return node
            else:
                if node.children:
                    node = self._select_child_with_exploration(node)
                else:
                    return node
        return node
    
    def _select_child_with_exploration(self, node: MCTSNode) -> MCTSNode:
        """Select best child using exploration constant."""
        return max(node.children, 
                   key=lambda child: child.ucb1_value(self.exploration_constant))
    
    def _weighted_move_selection(self, state: GameState, moves: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Select a move with weights favoring moves closer to existing tokens.
        Uses exponential decay: weight = exp(-distance / scale)
        
        Args:
            state: Current game state
            moves: List of legal moves
            
        Returns:
            Selected move tuple
        """
        if not moves:
            raise ValueError("No moves available")
        
        if len(moves) == 1:
            return moves[0]
        
        # Calculate weights based on distance to nearest token
        weights = []
        for move in moves:
            dist = state._min_distance_to_token(*move)
            # Exponential decay: closer moves have higher weights
            # Scale factor of 2.0 means moves 1 space away are ~1.65x more likely than 2 spaces away
            weight = math.exp(-dist / 2.0)
            weights.append(weight)
        
        # Normalize weights to probabilities
        total_weight = sum(weights)
        if total_weight == 0:
            # Fallback to uniform if all weights are 0
            return random.choice(moves)
        
        probabilities = [w / total_weight for w in weights]
        
        # Sample according to probabilities
        return moves[np.random.choice(len(moves), p=probabilities)]
    
    def _simulate(self, state: GameState) -> float:
        """
        Simulate a random game from the given state.
        Uses weighted sampling to prefer moves closer to existing tokens.
        Does not expand the board during rollout.
        
        Args:
            state: Game state to simulate from
            
        Returns:
            Result from the perspective of the state's current player
        """
        current_state = state.copy()
        depth = 0
        
        while not current_state.is_terminal() and depth < self.max_simulation_depth:
            # Use moves within bounds only (no expansion during rollout)
            legal_moves = list(current_state.get_legal_moves_within_bounds())
            if not legal_moves:
                break
            # Use weighted selection to prefer moves closer to tokens
            move = self._weighted_move_selection(current_state, legal_moves)
            current_state = current_state.make_move(*move)
            depth += 1
        
        result = current_state.get_result()
        if result is None:
            return 0.0
        
        # Result is from the perspective of the original state's current player
        return result
    
    def _backpropagate(self, node: MCTSNode, result: float):
        """
        Backpropagate simulation results up the tree.
        
        Args:
            node: Node to start backpropagation from
            result: Result from the perspective of the node's player
        """
        while node is not None:
            node.update(result)
            # Flip result for parent (opponent's perspective)
            result = -result
            node = node.parent

