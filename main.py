"""
Main file for playing the infinite tic-tac-toe game.
Supports human vs human, human vs Minimax, and Minimax vs Minimax.
"""

from game import GameState, Player
from minimax import Minimax


def print_board(state: GameState):
    """Print the game board in a readable format."""
    print("\n" + "=" * 50)
    print(state)
    print("=" * 50 + "\n")


def get_human_move(state: GameState) -> tuple:
    """Get a move from a human player."""
    legal_moves = state.get_legal_moves()
    
    print(f"Legal moves (row, col): {sorted(legal_moves)}")
    
    while True:
        try:
            move_input = input(f"Enter move for {state.current_player} (row col): ").strip()
            if not move_input:
                continue
            
            parts = move_input.split()
            if len(parts) != 2:
                print("Please enter two numbers separated by a space (e.g., '0 0')")
                continue
            
            row, col = int(parts[0]), int(parts[1])
            move = (row, col)
            
            if move not in legal_moves:
                print(f"Invalid move. Please choose from legal moves.")
                continue
            
            return move
        
        except ValueError:
            print("Please enter valid integers.")
        except KeyboardInterrupt:
            print("\nGame interrupted.")
            return None


def play_game(human_vs_human: bool = False, minimax_depth: int = 4):
    """
    Play a game of infinite tic-tac-toe.
    
    Args:
        human_vs_human: If True, both players are human. If False, player X is human, O is Minimax.
        minimax_depth: Minimax search depth
    """
    state = GameState()
    minimax = Minimax(depth=minimax_depth)
    
    print("=" * 50)
    print("Infinite Tic-Tac-Toe (5 in a row to win)")
    print("=" * 50)
    
    if human_vs_human:
        print("Mode: Human vs Human")
    else:
        print("Mode: Human (X) vs Minimax (O)")
        print(f"Minimax search depth: {minimax_depth}")
    
    print_board(state)
    
    while not state.is_terminal():
        current_player = state.current_player
        
        if human_vs_human or current_player == Player.X:
            # Human turn
            move = get_human_move(state)
            if move is None:
                return
        else:
            # Minimax turn
            print(f"Minimax ({current_player}) is thinking...")
            move = minimax.search(state)
            print(f"Minimax plays: {move}")
        
        state = state.make_move(*move)
        print_board(state)
        
        if state.is_terminal():
            winner = state.get_winner()
            if winner:
                print(f"\n🎉 {winner} wins!")
            else:
                print("\n🤝 It's a draw!")
            break


def test_minimax_vs_minimax(depth: int = 4, num_games: int = 1):
    """
    Test Minimax vs Minimax.
    
    Args:
        depth: Minimax search depth
        num_games: Number of games to play
    """
    print(f"Running Minimax vs Minimax ({num_games} game(s))")
    print(f"Search depth: {depth}")
    
    for game_num in range(num_games):
        print(f"\n{'=' * 50}")
        print(f"Game {game_num + 1}/{num_games}")
        print(f"{'=' * 50}")
        
        state = GameState()
        minimax = Minimax(depth=depth)
        move_count = 0
        
        while not state.is_terminal():
            move = minimax.search(state)
            state = state.make_move(*move)
            move_count += 1
            
            if move_count % 10 == 0:
                print(f"Move {move_count}: {state.current_player.other()} plays {move}")
        
        winner = state.get_winner()
        if winner:
            print(f"\nGame {game_num + 1} result: {winner} wins in {move_count} moves")
        else:
            print(f"\nGame {game_num + 1} result: Draw in {move_count} moves")
        
        print_board(state)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            # Minimax vs Minimax test
            depth = int(sys.argv[2]) if len(sys.argv) > 2 else 4
            num_games = int(sys.argv[3]) if len(sys.argv) > 3 else 1
            test_minimax_vs_minimax(depth, num_games)
        elif sys.argv[1] == "human":
            # Human vs Human
            play_game(human_vs_human=True)
        elif sys.argv[1] == "minimax":
            # Human vs Minimax
            depth = int(sys.argv[2]) if len(sys.argv) > 2 else 4
            play_game(human_vs_human=False, minimax_depth=depth)
        else:
            print("Usage:")
            print("  python main.py              - Human vs Minimax (default)")
            print("  python main.py human         - Human vs Human")
            print("  python main.py minimax [depth] - Human vs Minimax")
            print("  python main.py test [depth] [num_games] - Minimax vs Minimax")
    else:
        # Default: Human vs Minimax
        play_game(human_vs_human=False, minimax_depth=4)

