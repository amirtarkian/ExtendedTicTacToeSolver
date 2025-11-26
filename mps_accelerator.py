"""
Metal Performance Shaders (MPS) accelerator for MCTS simulations.
Uses PyTorch with MPS backend to accelerate batched game simulations.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
from game import GameState, Player
import random
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import math

logger = logging.getLogger(__name__)

# Check if MPS is available
MPS_AVAILABLE = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
DEVICE = torch.device("mps" if MPS_AVAILABLE else "cpu")

if MPS_AVAILABLE:
    logger.info("Metal Performance Shaders (MPS) is available - using GPU acceleration")
else:
    logger.info("MPS not available - using CPU (MPS requires macOS 12.3+ and Apple Silicon)")


class MPSAccelerator:
    """
    Accelerates MCTS simulations using Metal Performance Shaders.
    Batches multiple simulations together for parallel execution.
    """
    
    def __init__(self, batch_size: int = 32, use_mps: bool = True, num_workers: Optional[int] = None):
        """
        Initialize MPS accelerator.
        
        Args:
            batch_size: Number of simulations to batch together
            use_mps: Whether to use MPS (will fallback to CPU if unavailable)
            num_workers: Number of parallel workers for simulations (None = auto-detect)
        """
        self.batch_size = batch_size
        self.use_mps = use_mps and MPS_AVAILABLE
        self.device = DEVICE if self.use_mps else torch.device("cpu")
        
        # Determine number of workers for parallel processing
        if num_workers is None:
            # Use number of CPU cores, but cap at batch_size
            self.num_workers = min(os.cpu_count() or 4, batch_size)
        else:
            self.num_workers = num_workers
        
        if self.use_mps:
            logger.info(f"MPS accelerator initialized with batch_size={batch_size} on device {self.device}")
            logger.info(f"Using {self.num_workers} parallel workers for simulations")
            # Verify MPS is actually available
            test_tensor = torch.rand(10, device=self.device)
            if test_tensor.device.type == 'mps':
                logger.info("MPS device verified - GPU acceleration active")
            else:
                logger.warning(f"MPS initialization failed - tensor created on {test_tensor.device.type} instead")
        else:
            logger.info(f"CPU accelerator initialized with batch_size={batch_size}")
            logger.info(f"Using {self.num_workers} parallel workers for simulations")
    
    def batch_simulate(self, states: List[GameState], max_depth: int = 50) -> List[float]:
        """
        Run batched simulations using GPU acceleration.
        
        Args:
            states: List of game states to simulate from
            max_depth: Maximum simulation depth
            
        Returns:
            List of simulation results (from each state's current player perspective)
        """
        if not states:
            return []
        
        results = []
        
        # Process in batches
        for i in range(0, len(states), self.batch_size):
            batch = states[i:i + self.batch_size]
            batch_results = self._simulate_batch(batch, max_depth)
            results.extend(batch_results)
        
        return results
    
    def _simulate_batch(self, states: List[GameState], max_depth: int) -> List[float]:
        """
        Simulate a batch of game states in parallel.
        Uses GPU-accelerated random number generation when possible.
        
        Args:
            states: Batch of game states
            max_depth: Maximum simulation depth
            
        Returns:
            List of simulation results (in same order as input states)
        """
        if not states:
            return []
        
        # Generate random numbers on GPU if using MPS
        if self.use_mps:
            # Use GPU for random number generation
            # We'll still do the game logic on CPU but batch the random choices
            num_random_values = len(states) * max_depth
            random_tensor = torch.rand(num_random_values, device=self.device)
            # Verify tensor is on MPS device
            if random_tensor.device.type != 'mps':
                logger.warning(f"Expected MPS device but got {random_tensor.device.type}")
            random_values = random_tensor.cpu().numpy().tolist()
        else:
            random_values = None
        
        # Run simulations in parallel using ThreadPoolExecutor
        results = [None] * len(states)  # Pre-allocate to maintain order
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all simulation tasks
            future_to_index = {}
            for idx, state in enumerate(states):
                random_start_idx = idx * max_depth if random_values else 0
                future = executor.submit(
                    self._simulate_single, 
                    state, 
                    max_depth, 
                    random_values, 
                    random_start_idx
                )
                future_to_index[future] = idx
            
            # Collect results as they complete (maintain order)
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Simulation failed for state {idx}: {e}")
                    results[idx] = 0.0
        
        return results
    
    def _weighted_move_selection(self, state: GameState, moves: List[Tuple[int, int]], 
                                 random_val: float) -> Tuple[int, int]:
        """
        Select a move with weights favoring moves closer to existing tokens.
        Uses exponential decay: weight = exp(-distance / scale)
        
        Args:
            state: Current game state
            moves: List of legal moves
            random_val: Random value [0, 1) for weighted sampling
            
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
            return moves[int(random_val * len(moves))]
        
        probabilities = [w / total_weight for w in weights]
        
        # Sample according to probabilities using the provided random value
        cumulative = 0.0
        for i, prob in enumerate(probabilities):
            cumulative += prob
            if random_val < cumulative:
                return moves[i]
        
        # Fallback to last move (shouldn't happen, but safety)
        return moves[-1]
    
    def _simulate_single(self, state: GameState, max_depth: int, 
                        random_values: Optional[List[float]] = None,
                        random_start_idx: int = 0) -> float:
        """
        Simulate a single game state.
        Uses weighted sampling to prefer moves closer to existing tokens.
        Does not expand the board during rollout.
        
        Args:
            state: Game state to simulate from
            max_depth: Maximum simulation depth
            random_values: Pre-generated random values (for batching)
            random_start_idx: Starting index in random_values
            
        Returns:
            Simulation result from state's current player perspective
        """
        current_state = state.copy()
        depth = 0
        
        while not current_state.is_terminal() and depth < max_depth:
            # Use moves within bounds only (no expansion during rollout)
            legal_moves = list(current_state.get_legal_moves_within_bounds())
            if not legal_moves:
                break
            
            # Use pre-generated random value if available, otherwise generate new one
            if random_values and random_start_idx + depth < len(random_values):
                rand_val = random_values[random_start_idx + depth]
            else:
                rand_val = random.random()
            
            # Use weighted selection to prefer moves closer to tokens
            move = self._weighted_move_selection(current_state, legal_moves, rand_val)
            current_state = current_state.make_move(*move)
            depth += 1
        
        result = current_state.get_result()
        if result is None:
            return 0.0
        
        return result
    
    def generate_random_moves_batch(self, num_moves: int, max_legal_moves: int = 100) -> torch.Tensor:
        """
        Generate random move indices in batch using GPU.
        
        Args:
            num_moves: Number of random moves to generate
            max_legal_moves: Maximum number of legal moves (for sizing)
            
        Returns:
            Tensor of random move indices
        """
        if self.use_mps:
            # Generate random values on GPU
            random_vals = torch.rand(num_moves, device=self.device)
            # Verify tensor is on MPS device
            if random_vals.device.type != 'mps':
                logger.warning(f"Expected MPS device but got {random_vals.device.type}")
            # Convert to move indices (0 to max_legal_moves-1)
            move_indices = (random_vals * max_legal_moves).long()
            return move_indices.cpu()
        else:
            # CPU fallback
            return torch.randint(0, max_legal_moves, (num_moves,))
    
    def is_available(self) -> bool:
        """Check if MPS acceleration is available."""
        return self.use_mps
    
    def verify_mps_usage(self) -> dict:
        """
        Verify that MPS is actually being used and return status information.
        
        Returns:
            Dictionary with verification results
        """
        status = {
            'mps_available': MPS_AVAILABLE,
            'use_mps': self.use_mps,
            'device': str(self.device),
            'torch_threads': torch.get_num_threads(),
            'torch_interop_threads': torch.get_num_interop_threads(),
        }
        
        if self.use_mps:
            # Test creating a tensor on MPS
            try:
                test_tensor = torch.rand(100, device=self.device)
                status['tensor_device'] = str(test_tensor.device)
                status['tensor_on_mps'] = test_tensor.device.type == 'mps'
                status['verification'] = 'PASS' if test_tensor.device.type == 'mps' else 'FAIL'
            except Exception as e:
                status['verification'] = f'ERROR: {e}'
                status['tensor_on_mps'] = False
        else:
            status['verification'] = 'SKIP (MPS not enabled)'
            status['tensor_on_mps'] = False
        
        return status


def create_accelerator(batch_size: int = 32, use_mps: bool = True, num_workers: Optional[int] = None) -> MPSAccelerator:
    """
    Factory function to create an MPS accelerator.
    
    Args:
        batch_size: Number of simulations to batch together
        use_mps: Whether to use MPS (will fallback to CPU if unavailable)
        num_workers: Number of parallel workers for simulations (None = auto-detect)
        
    Returns:
        MPSAccelerator instance
    """
    return MPSAccelerator(batch_size=batch_size, use_mps=use_mps, num_workers=num_workers)

