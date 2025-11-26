import React from 'react';
import './MinimaxProgress.css';

function MinimaxProgress({ progress }) {
  if (!progress || progress.is_complete) {
    return null;
  }

  const { current_move, total_moves, current_depth, best_value, best_move } = progress;
  
  // Calculate progress percentage
  const progressPercent = total_moves > 0 
    ? Math.min(100, (current_move / total_moves) * 100) 
    : 0;

  // Format best value
  const formatValue = (value) => {
    if (value === null || value === undefined) {
      return '—';
    }
    return value.toFixed(3);
  };

  // Format best move
  const formatMove = (move) => {
    if (!move) {
      return '—';
    }
    return `(${move.row}, ${move.col})`;
  };

  return (
    <div className="minimax-progress">
      <div className="progress-header">
        <h3>Minimax Search Progress</h3>
      </div>
      
      <div className="progress-content">
        {/* Moves Progress Bar */}
        <div className="progress-section">
          <div className="progress-label">
            <span>Evaluating Moves</span>
            <span className="progress-count">
              {current_move} / {total_moves}
            </span>
          </div>
          <div className="progress-bar-container">
            <div 
              className="progress-bar-fill" 
              style={{ width: `${progressPercent}%` }}
            />
          </div>
        </div>

        {/* Depth Indicator */}
        <div className="progress-section">
          <div className="progress-label">
            <span>Search Depth</span>
            <span className="depth-indicator">
              {current_depth}
            </span>
          </div>
        </div>

        {/* Value Function Display */}
        <div className="progress-section">
          <div className="value-display">
            <div className="value-item">
              <span className="value-label">Best Value:</span>
              <span className="value-number">{formatValue(best_value)}</span>
            </div>
            <div className="value-item">
              <span className="value-label">Best Move:</span>
              <span className="value-number">{formatMove(best_move)}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default MinimaxProgress;

