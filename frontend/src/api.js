/**
 * API client for communicating with the Flask backend
 */

import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5001/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Create a new game
 */
export const createGame = async (minimaxParams = {}) => {
  const response = await api.post('/game/new', minimaxParams);
  return response.data;
};

/**
 * Get current game state
 */
export const getGameState = async (gameId) => {
  const response = await api.get(`/game/${gameId}`);
  return response.data;
};

/**
 * Make a move in the game
 */
export const makeMove = async (gameId, row, col) => {
  const response = await api.post(`/game/${gameId}/move`, {
    row,
    col,
  });
  return response.data;
};

/**
 * Get Minimax move recommendation (without playing it)
 */
export const getMinimaxMove = async (gameId) => {
  const response = await api.post(`/game/${gameId}/minimax`);
  return response.data;
};

/**
 * Make Minimax play a move
 */
export const minimaxPlay = async (gameId) => {
  const response = await api.post(`/game/${gameId}/minimax/play`);
  return response.data;
};

// Backward compatibility aliases
export const getMCTSMove = getMinimaxMove;
export const mctsPlay = minimaxPlay;

/**
 * Reset a game
 */
export const resetGame = async (gameId) => {
  const response = await api.post(`/game/${gameId}/reset`);
  return response.data;
};

/**
 * Get Minimax parameters
 */
export const getMinimaxParams = async (gameId) => {
  const response = await api.get(`/game/${gameId}/minimax/params`);
  return response.data;
};

/**
 * Update Minimax parameters
 */
export const setMinimaxParams = async (gameId, params) => {
  const response = await api.post(`/game/${gameId}/minimax/params`, params);
  return response.data;
};

/**
 * Get Minimax search progress
 */
export const getMinimaxProgress = async (gameId) => {
  const response = await api.get(`/game/${gameId}/minimax/progress`);
  return response.data;
};

// Backward compatibility aliases
export const getMCTSParams = getMinimaxParams;
export const setMCTSParams = setMinimaxParams;

