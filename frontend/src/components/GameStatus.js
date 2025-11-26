import React from 'react';
import './GameStatus.css';

function GameStatus({ gameState, loading }) {
  if (!gameState) {
    return (
      <div className="game-status">
        <div className="status-loading">Loading game status...</div>
      </div>
    );
  }

  const getStatusMessage = () => {
    if (gameState.is_terminal) {
      if (gameState.winner) {
        return `🎉 ${gameState.winner} wins!`;
      }
      return "🤝 It's a draw!";
    }
    return `Current player: ${gameState.current_player}`;
  };

  const getStatusClass = () => {
    if (gameState.is_terminal) {
      return gameState.winner ? 'status-winner' : 'status-draw';
    }
    return `status-${gameState.current_player.toLowerCase()}`;
  };

  return (
    <div className="game-status">
      <div className={`status-message ${getStatusClass()}`}>
        {loading && <span className="loading-spinner">⏳</span>}
        {getStatusMessage()}
      </div>
      {!gameState.is_terminal && (
        <div className="status-hint">
          {gameState.current_player === 'O' 
            ? 'Click on a highlighted cell to make your move'
            : 'Waiting for Minimax to play...'}
        </div>
      )}
    </div>
  );
}

export default GameStatus;

