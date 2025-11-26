import React from 'react';
import './GameBoard.css';

function GameBoard({ gameState, onCellClick, loading }) {
  if (!gameState) {
    return <div className="game-board-loading">Loading board...</div>;
  }

  // Extract board positions
  const boardPositions = {};
  Object.entries(gameState.board).forEach(([key, player]) => {
    const [row, col] = key.split(',').map(Number);
    boardPositions[`${row},${col}`] = player;
  });

  // Find bounds
  const positions = Object.keys(boardPositions).map(key => {
    const [r, c] = key.split(',').map(Number);
    return { row: r, col: c };
  });

  if (positions.length === 0) {
    // Empty board - show center cell
    return (
      <div className="game-board-container">
        <div className="game-board">
          <div
            className={`cell empty legal ${loading ? 'disabled' : ''}`}
            onClick={() => !loading && onCellClick(0, 0)}
          >
            {boardPositions['0,0'] || ''}
          </div>
        </div>
      </div>
    );
  }

  const rows = positions.map(p => p.row);
  const cols = positions.map(p => p.col);
  const minRow = Math.min(...rows);
  const maxRow = Math.max(...rows);
  const minCol = Math.min(...cols);
  const maxCol = Math.max(...cols);

  // Add padding around the board
  const padding = 2;
  const displayMinRow = minRow - padding;
  const displayMaxRow = maxRow + padding;
  const displayMinCol = minCol - padding;
  const displayMaxCol = maxCol + padding;

  // Create set of legal moves for quick lookup
  const legalMovesSet = new Set(
    gameState.legal_moves.map(m => `${m.row},${m.col}`)
  );

  const cells = [];
  for (let row = displayMinRow; row <= displayMaxRow; row++) {
    for (let col = displayMinCol; col <= displayMaxCol; col++) {
      const key = `${row},${col}`;
      const player = boardPositions[key];
      const isLegal = legalMovesSet.has(key);
      const isEmpty = !player;

      cells.push(
        <div
          key={key}
          className={`cell ${isEmpty ? 'empty' : 'filled'} ${isLegal ? 'legal' : ''} ${loading ? 'disabled' : ''} ${player ? `player-${player.toLowerCase()}` : ''}`}
          onClick={() => !loading && isLegal && isEmpty && onCellClick(row, col)}
          title={isLegal && isEmpty ? `Click to place ${gameState.current_player}` : ''}
          data-player={player}
        >
          {player || (isLegal ? '·' : '')}
        </div>
      );
    }
  }

  // Calculate grid size
  const gridCols = displayMaxCol - displayMinCol + 1;

  return (
    <div className="game-board-container">
      <div className="game-board" style={{ gridTemplateColumns: `repeat(${gridCols}, 1fr)` }}>
        {cells}
      </div>
      <div className="board-info">
        <p>Board bounds: rows [{minRow}, {maxRow}], cols [{minCol}, {maxCol}]</p>
        <p>Legal moves: {gameState.legal_moves.length}</p>
      </div>
    </div>
  );
}

export default GameBoard;

