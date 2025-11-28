import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import GameBoard from './components/GameBoard';
import GameControls from './components/GameControls';
import GameStatus from './components/GameStatus';
import MinimaxProgress from './components/MinimaxProgress';
import { createGame, getGameState, makeMove, minimaxPlay, setMinimaxParams, getMinimaxProgress } from './api';

function App() {
  const [gameId, setGameId] = useState(null);
  const [gameState, setGameState] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [minimaxParams, setMinimaxParams] = useState({
    depth: 4
  });
  const [minimaxProgress, setMinimaxProgress] = useState(null);
  const [lastMove, setLastMove] = useState(null);
  const progressPollIntervalRef = useRef(null);

  useEffect(() => {
    startNewGame();
    
    // Cleanup: clear progress polling on unmount
    return () => {
      if (progressPollIntervalRef.current) {
        clearInterval(progressPollIntervalRef.current);
        progressPollIntervalRef.current = null;
      }
    };
  }, []);

  const startNewGame = async () => {
    try {
      setLoading(true);
      setError(null);
      setLastMove(null);
      const response = await createGame(minimaxParams);
      setGameId(response.game_id);
      setGameState(response.state);
      if (response.minimax_params) {
        setMinimaxParams(response.minimax_params);
      }
      
      // If game starts with X (AI), auto-play the first move
      if (response.state && !response.state.is_terminal && response.state.current_player === 'X') {
        setTimeout(async () => {
          await handleMinimaxPlayWithProgress(response.game_id);
        }, 500);
      } else {
        setLoading(false);
      }
    } catch (err) {
      setError('Failed to create game: ' + err.message);
      setLoading(false);
    }
  };

  const handleMinimaxParamChange = async (param, value) => {
    const newParams = { ...minimaxParams, [param]: value };
    setMinimaxParams(newParams);
    
    if (gameId) {
      try {
        await setMinimaxParams(gameId, { [param]: value });
      } catch (err) {
        console.error('Failed to update Minimax params:', err);
      }
    }
  };

  const handleCellClick = async (row, col) => {
    if (!gameId || !gameState) return;
    if (gameState.is_terminal) return;
    if (gameState.current_player !== 'O') return; // Only allow O (human) to click
    
    // Check if move is legal
    const isLegal = gameState.legal_moves.some(
      move => move.row === row && move.col === col
    );
    if (!isLegal) return;

    try {
      setLoading(true);
      setError(null);
      const response = await makeMove(gameId, row, col);
      setGameState(response.state);
      setLastMove({ row, col });

      // If game continues and it's X's turn, let Minimax play
      if (!response.state.is_terminal && response.state.current_player === 'X') {
        setTimeout(async () => {
          await handleMinimaxPlayWithProgress(gameId);
        }, 500);
      } else {
        setLoading(false);
      }
    } catch (err) {
      setError('Move failed: ' + err.message);
      setLoading(false);
    }
  };

  const startProgressPolling = (gameId) => {
    // Clear any existing interval
    if (progressPollIntervalRef.current) {
      clearInterval(progressPollIntervalRef.current);
    }
    
    // Poll for progress every 150ms
    progressPollIntervalRef.current = setInterval(async () => {
      try {
        const progressData = await getMinimaxProgress(gameId);
        if (progressData.progress) {
          setMinimaxProgress(progressData.progress);
          
          // Stop polling if complete
          if (progressData.progress.is_complete) {
            if (progressPollIntervalRef.current) {
              clearInterval(progressPollIntervalRef.current);
              progressPollIntervalRef.current = null;
            }
            // Clear progress after a short delay
            setTimeout(() => {
              setMinimaxProgress(null);
            }, 500);
          }
        } else {
          // No progress data, stop polling
          if (progressPollIntervalRef.current) {
            clearInterval(progressPollIntervalRef.current);
            progressPollIntervalRef.current = null;
          }
          setMinimaxProgress(null);
        }
      } catch (err) {
        console.error('Failed to get minimax progress:', err);
        // Stop polling on error
        if (progressPollIntervalRef.current) {
          clearInterval(progressPollIntervalRef.current);
          progressPollIntervalRef.current = null;
        }
        setMinimaxProgress(null);
      }
    }, 150);
  };

  const handleMinimaxPlayWithProgress = async (targetGameId) => {
    if (!targetGameId) return;

    try {
      setLoading(true);
      setError(null);
      setMinimaxProgress(null);
      
      // Start polling for progress
      startProgressPolling(targetGameId);
      
      // Make the minimax play request
      const response = await minimaxPlay(targetGameId);
      setGameState(response.state);
      
      // Update last move if move is included in response
      if (response.move) {
        setLastMove({ row: response.move.row, col: response.move.col });
      }
      
      // Stop polling and clear progress
      if (progressPollIntervalRef.current) {
        clearInterval(progressPollIntervalRef.current);
        progressPollIntervalRef.current = null;
      }
      
      // Get final progress before clearing
      try {
        const finalProgress = await getMinimaxProgress(targetGameId);
        if (finalProgress.progress) {
          setMinimaxProgress(finalProgress.progress);
          setTimeout(() => {
            setMinimaxProgress(null);
          }, 1000);
        }
      } catch (err) {
        // Ignore errors getting final progress
      }
    } catch (err) {
      setError('Minimax move failed: ' + err.message);
      // Stop polling on error
      if (progressPollIntervalRef.current) {
        clearInterval(progressPollIntervalRef.current);
        progressPollIntervalRef.current = null;
      }
      setMinimaxProgress(null);
    } finally {
      setLoading(false);
    }
  };

  const handleMinimaxPlay = async () => {
    if (!gameId || !gameState) return;
    if (gameState.is_terminal) return;

    await handleMinimaxPlayWithProgress(gameId);
  };

  return (
    <div className="App">
      <div className="container">
        <header className="header">
          <h1>Extended Tic-Tac-Toe</h1>
          <p className="subtitle">5 in a row to win • Infinite board</p>
        </header>

        {error && (
          <div className="error-message">
            {error}
          </div>
        )}

        <GameStatus gameState={gameState} loading={loading} />

        {minimaxProgress && (
          <MinimaxProgress progress={minimaxProgress} />
        )}

        <GameBoard
          gameState={gameState}
          onCellClick={handleCellClick}
          loading={loading}
          lastMove={lastMove}
        />

        <GameControls
          onNewGame={startNewGame}
          onMinimaxPlay={handleMinimaxPlay}
          gameState={gameState}
          loading={loading}
          minimaxParams={minimaxParams}
          onMinimaxParamChange={handleMinimaxParamChange}
        />
      </div>
    </div>
  );
}

export default App;

