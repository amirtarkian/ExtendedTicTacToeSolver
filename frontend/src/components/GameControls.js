import React, { useState } from 'react';
import './GameControls.css';

function GameControls({ onNewGame, onMinimaxPlay, gameState, loading, minimaxParams, onMinimaxParamChange }) {
  const [showParams, setShowParams] = useState(false);

  return (
    <div className="game-controls">
      <div className="button-row">
        <button
          className="btn btn-primary"
          onClick={onNewGame}
          disabled={loading}
        >
          New Game
        </button>
        {gameState && !gameState.is_terminal && gameState.current_player === 'X' && (
          <button
            className="btn btn-secondary"
            onClick={onMinimaxPlay}
            disabled={loading}
          >
            Let Minimax Play
          </button>
        )}
        <button
          className="btn btn-toggle"
          onClick={() => setShowParams(!showParams)}
        >
          {showParams ? 'Hide' : 'Show'} Minimax Settings
        </button>
      </div>

      {showParams && minimaxParams && (
        <div className="minimax-params">
          <h3>Minimax Parameters</h3>
          
          <div className="param-group">
            <label htmlFor="depth">
              Search Depth: <span className="param-value">{minimaxParams.depth}</span>
            </label>
            <input
              type="range"
              id="depth"
              min="2"
              max="8"
              step="1"
              value={minimaxParams.depth}
              onChange={(e) => onMinimaxParamChange('depth', parseInt(e.target.value))}
              disabled={loading}
            />
            <span className="param-hint">Higher = stronger but slower (2-8 recommended)</span>
          </div>
        </div>
      )}
    </div>
  );
}

export default GameControls;

