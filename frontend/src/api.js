const API_BASE = '/api/v1';

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
    this.name = 'ApiError';
  }
}

async function request(url, options = {}) {
  const response = await fetch(`${API_BASE}${url}`, {
    headers: {
      'Content-Type': 'application/json',
    },
    ...options,
  });

  const data = await response.json();

  if (!response.ok) {
    throw new ApiError(data.error || `Request failed with status ${response.status}`, response.status);
  }

  return data;
}

export async function getPairs() {
  return request('/pairs');
}

export async function getHistory(pair) {
  return request(`/pairs/${encodeURIComponent(pair)}/history`);
}

export async function trainModel(pair, timeframe = '1h') {
  return request(`/pairs/${encodeURIComponent(pair)}/train?timeframe=${encodeURIComponent(timeframe)}`, { method: 'POST' });
}

export async function getPredictions(pair) {
  return request(`/pairs/${encodeURIComponent(pair)}/predict`);
}

export async function getTradeSetup(pair, timeframe = '1h') {
  return request(`/pairs/${encodeURIComponent(pair)}/trade-setup?timeframe=${encodeURIComponent(timeframe)}`);
}

export async function getForecast(pair, horizon = 30, timeframe = '1h') {
  return request(`/pairs/${encodeURIComponent(pair)}/forecast?horizon=${encodeURIComponent(horizon)}&timeframe=${encodeURIComponent(timeframe)}`);
}

export async function getModels() {
  return request('/models');
}

export { ApiError };