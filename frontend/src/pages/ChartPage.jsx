import { useState, useEffect, useCallback, useRef } from 'react';
import { getPairs, getAccuracyPredictions, createChartWebSocket } from '../api';
import Spinner from '../components/Spinner';
import ErrorMessage from '../components/ErrorMessage';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const TIMEFRAMES = [
  { value: '5m', label: '5m' },
  { value: '15m', label: '15m' },
  { value: '1h', label: '1h' },
  { value: '4h', label: '4h' },
];

const SUB_VIEWS = [
  { value: 'predictions', label: 'Predictions' },
  { value: 'forecast', label: 'Forecast' },
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatPair(ticker) {
  return ticker.replace('=X', '');
}

function formatDate(dateStr) {
  const d = new Date(dateStr);
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

function formatTime(dateStr) {
  const d = new Date(dateStr);
  return d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
}

function isDarkMode() {
  return document.documentElement.classList.contains('dark');
}

function formatDrift(value) {
  if (value == null) return '—';
  const sign = value >= 0 ? '+' : '';
  return `${sign}${(value * 100).toFixed(3)}%`;
}

function formatDriftAbs(value) {
  if (value == null) return '—';
  const sign = value >= 0 ? '+' : '';
  return `${sign}${value.toFixed(4)}%`;
}

// ---------------------------------------------------------------------------
// Canvas chart drawing — Predictions view
// ---------------------------------------------------------------------------

function drawPredictionsChart(canvas, predictions, theme) {
  if (!canvas || !predictions || predictions.length < 2) {
    drawEmptyChart(canvas, theme, 'Need at least 2 evaluated predictions');
    return;
  }

  const dark = theme ?? isDarkMode();
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();

  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const W = rect.width;
  const H = rect.height;
  const PAD = { top: 30, right: 70, bottom: 45, left: 15 };

  const chartW = W - PAD.left - PAD.right;
  const chartH = H - PAD.top - PAD.bottom;

  // Filter to evaluated predictions with both prices
  const evaluated = predictions.filter(
    (p) => p.target_price != null && p.actual_price != null && p.predicted_at
  );

  if (evaluated.length < 2) {
    drawEmptyChart(canvas, dark, 'Need at least 2 evaluated predictions');
    return;
  }

  const points = evaluated
    .sort((a, b) => new Date(a.predicted_at) - new Date(b.predicted_at))
    .map((p) => ({
      date: new Date(p.predicted_at),
      predicted: p.target_price,
      actual: p.actual_price,
      drift: p.percentage_drift,
    }));

  const minDate = points[0].date.getTime();
  const maxDate = points[points.length - 1].date.getTime();
  const allPrices = points.flatMap((p) => [p.predicted, p.actual]);
  const minPrice = Math.min(...allPrices) * 0.9995;
  const maxPrice = Math.max(...allPrices) * 1.0005;

  const xScale = (date) =>
    PAD.left + ((date.getTime() - minDate) / (maxDate - minDate)) * chartW;
  const yScale = (price) =>
    PAD.top + chartH - ((price - minPrice) / (maxPrice - minPrice)) * chartH;

  // Clear & background
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = dark ? '#111827' : '#ffffff';
  ctx.fillRect(0, 0, W, H);

  // Grid lines
  const gridColor = dark ? '#374151' : '#f3f4f6';
  const textColor = dark ? '#9ca3af' : '#9ca3af';
  ctx.strokeStyle = gridColor;
  ctx.lineWidth = 1;
  const numGridLines = 5;
  for (let i = 0; i <= numGridLines; i++) {
    const y = PAD.top + (chartH / numGridLines) * i;
    ctx.beginPath();
    ctx.moveTo(PAD.left, y);
    ctx.lineTo(W - PAD.right, y);
    ctx.stroke();

    const price = maxPrice - ((maxPrice - minPrice) / numGridLines) * i;
    ctx.fillStyle = textColor;
    ctx.font = '10px Inter, system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(price.toFixed(4), W - PAD.right + 8, y + 4);
  }

  // Predicted price line (dashed orange)
  ctx.beginPath();
  ctx.strokeStyle = '#f97316';
  ctx.lineWidth = 2;
  ctx.lineJoin = 'round';
  ctx.setLineDash([5, 3]);
  points.forEach((p, i) => {
    const x = xScale(p.date);
    const y = yScale(p.predicted);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.setLineDash([]);

  // Actual price line (solid blue)
  ctx.beginPath();
  ctx.strokeStyle = '#2563eb';
  ctx.lineWidth = 2;
  ctx.lineJoin = 'round';
  points.forEach((p, i) => {
    const x = xScale(p.date);
    const y = yScale(p.actual);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Drift area (subtle fill between the two lines)
  ctx.beginPath();
  points.forEach((p, i) => {
    const x = xScale(p.date);
    const y = yScale(p.predicted);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  for (let i = points.length - 1; i >= 0; i--) {
    const x = xScale(points[i].date);
    const y = yScale(points[i].actual);
    ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.fillStyle = dark ? 'rgba(139, 92, 246, 0.08)' : 'rgba(139, 92, 246, 0.06)';
  ctx.fill();

  // Dots at last points
  const lastPoint = points[points.length - 1];
  const dotStroke = dark ? '#111827' : '#ffffff';

  ctx.beginPath();
  ctx.arc(xScale(lastPoint.date), yScale(lastPoint.predicted), 4, 0, Math.PI * 2);
  ctx.fillStyle = '#f97316';
  ctx.fill();
  ctx.strokeStyle = dotStroke;
  ctx.lineWidth = 2;
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(xScale(lastPoint.date), yScale(lastPoint.actual), 4, 0, Math.PI * 2);
  ctx.fillStyle = '#2563eb';
  ctx.fill();
  ctx.strokeStyle = dotStroke;
  ctx.lineWidth = 2;
  ctx.stroke();

  // Date labels
  ctx.fillStyle = textColor;
  ctx.font = '10px Inter, system-ui, sans-serif';
  ctx.textAlign = 'center';
  const dateStep = Math.max(1, Math.floor(points.length / 8));
  for (let i = 0; i < points.length; i += dateStep) {
    ctx.fillText(formatDate(points[i].date.toISOString()), xScale(points[i].date), H - PAD.bottom + 20);
  }
  ctx.fillText(formatDate(lastPoint.date.toISOString()), xScale(lastPoint.date), H - PAD.bottom + 20);

  // Y-axis label
  ctx.save();
  ctx.translate(12, PAD.top + chartH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = textColor;
  ctx.font = '11px Inter, system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Price', 0, 0);
  ctx.restore();
}

// ---------------------------------------------------------------------------
// Drift chart (shows percentage drift over time for each data point)
// ---------------------------------------------------------------------------

function drawDriftChart(canvas, points, theme) {
  if (!canvas || !points || points.length < 2) {
    drawEmptyChart(canvas, theme, 'Need at least 2 data points');
    return;
  }

  const dark = theme ?? isDarkMode();
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();

  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const W = rect.width;
  const H = rect.height;
  const PAD = { top: 20, right: 50, bottom: 45, left: 15 };

  const chartW = W - PAD.left - PAD.right;
  const chartH = H - PAD.top - PAD.bottom;

  const validPoints = points.filter((p) => p.drift != null);
  if (validPoints.length < 2) {
    drawEmptyChart(canvas, dark, 'Not enough drift data to display');
    return;
  }

  const drifts = validPoints.map((p) => p.drift);
  const maxAbsDrift = Math.max(...drifts.map(Math.abs), 0.001);
  const minDrift = -maxAbsDrift;
  const maxDrift = maxAbsDrift;

  const xScale = (i) => PAD.left + (i / (validPoints.length - 1)) * chartW;
  const yScale = (val) =>
    PAD.top + chartH - ((val - minDrift) / (maxDrift - minDrift)) * chartH;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = dark ? '#111827' : '#ffffff';
  ctx.fillRect(0, 0, W, H);

  // Grid lines
  const gridColor = dark ? '#374151' : '#f3f4f6';
  const textColor = dark ? '#9ca3af' : '#9ca3af';
  ctx.strokeStyle = gridColor;
  ctx.lineWidth = 1;
  const numGridLines = 5;
  for (let i = 0; i <= numGridLines; i++) {
    const y = PAD.top + (chartH / numGridLines) * i;
    ctx.beginPath();
    ctx.moveTo(PAD.left, y);
    ctx.lineTo(W - PAD.right, y);
    ctx.stroke();

    const val = maxDrift - ((maxDrift - minDrift) / numGridLines) * i;
    ctx.fillStyle = textColor;
    ctx.font = '10px Inter, system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`${val >= 0 ? '+' : ''}${val.toFixed(3)}%`, W - PAD.right + 5, y + 4);
  }

  // Zero line
  const y0 = yScale(0);
  ctx.beginPath();
  ctx.setLineDash([4, 4]);
  ctx.strokeStyle = dark ? '#6b7280' : '#d1d5db';
  ctx.lineWidth = 1;
  ctx.moveTo(PAD.left, y0);
  ctx.lineTo(W - PAD.right, y0);
  ctx.stroke();
  ctx.setLineDash([]);

  // Fill area
  ctx.beginPath();
  validPoints.forEach((p, i) => {
    const x = xScale(i);
    const y = yScale(p.drift);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.lineTo(xScale(validPoints.length - 1), y0);
  ctx.lineTo(PAD.left, y0);
  ctx.closePath();

  // Gradient fill based on positive/negative
  const gradient = ctx.createLinearGradient(0, PAD.top, 0, PAD.top + chartH);
  gradient.addColorStop(0, 'rgba(16, 185, 129, 0.12)');
  gradient.addColorStop(0.5, 'rgba(16, 185, 129, 0.02)');
  gradient.addColorStop(1, 'rgba(239, 68, 68, 0.12)');
  ctx.fillStyle = gradient;
  ctx.fill();

  // Line
  ctx.beginPath();
  ctx.strokeStyle = '#8b5cf6';
  ctx.lineWidth = 2;
  ctx.lineJoin = 'round';
  validPoints.forEach((p, i) => {
    const x = xScale(i);
    const y = yScale(p.drift);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Dots
  const dotStroke = dark ? '#111827' : '#ffffff';
  validPoints.forEach((p, i) => {
    ctx.beginPath();
    ctx.arc(xScale(i), yScale(p.drift), 3, 0, Math.PI * 2);
    ctx.fillStyle = p.drift >= 0 ? '#10b981' : '#ef4444';
    ctx.fill();
    ctx.strokeStyle = dotStroke;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  });

  // X-axis labels
  ctx.fillStyle = textColor;
  ctx.font = '10px Inter, system-ui, sans-serif';
  ctx.textAlign = 'center';
  const labelStep = Math.max(1, Math.floor(validPoints.length / 8));
  for (let i = 0; i < validPoints.length; i += labelStep) {
    if (validPoints[i].date) {
      ctx.fillText(formatDate(validPoints[i].date), xScale(i), H - PAD.bottom + 20);
    }
  }

  // Y-axis label
  ctx.save();
  ctx.translate(12, PAD.top + chartH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = textColor;
  ctx.font = '11px Inter, system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('% Drift', 0, 0);
  ctx.restore();
}

// ---------------------------------------------------------------------------
// Canvas chart drawing — Forecast view (live via WebSocket)
// ---------------------------------------------------------------------------

function drawForecastChart(canvas, dataPoints, theme) {
  if (!canvas || !dataPoints || dataPoints.length < 2) {
    drawEmptyChart(canvas, theme, 'Waiting for live data…');
    return;
  }

  const dark = theme ?? isDarkMode();
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();

  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const W = rect.width;
  const H = rect.height;
  const PAD = { top: 30, right: 70, bottom: 45, left: 15 };

  const chartW = W - PAD.left - PAD.right;
  const chartH = H - PAD.top - PAD.bottom;

  // Separate live and forecast prices
  const livePoints = dataPoints.filter((d) => d.live_price != null);
  const forecastPoints = dataPoints.filter((d) => d.forecast_price != null);

  if (livePoints.length < 2) {
    drawEmptyChart(canvas, dark, 'Waiting for live data…');
    return;
  }

  // All prices for y-axis range
  const allPrices = dataPoints.flatMap((d) => [
    d.live_price,
    d.forecast_price,
  ]).filter((p) => p != null);

  if (allPrices.length < 2) {
    drawEmptyChart(canvas, dark, 'Waiting for price data…');
    return;
  }

  const minPrice = Math.min(...allPrices) * 0.9995;
  const maxPrice = Math.max(...allPrices) * 1.0005;

  const minDate = new Date(dataPoints[0].timestamp).getTime();
  const maxDate = new Date(dataPoints[dataPoints.length - 1].timestamp).getTime();
  const dateRange = maxDate - minDate || 1;

  const xScale = (date) =>
    PAD.left + ((date.getTime() - minDate) / dateRange) * chartW;
  const yScale = (price) =>
    PAD.top + chartH - ((price - minPrice) / (maxPrice - minPrice)) * chartH;

  // Clear & background
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = dark ? '#111827' : '#ffffff';
  ctx.fillRect(0, 0, W, H);

  // Grid lines
  const gridColor = dark ? '#374151' : '#f3f4f6';
  const textColor = dark ? '#9ca3af' : '#9ca3af';
  ctx.strokeStyle = gridColor;
  ctx.lineWidth = 1;
  const numGridLines = 5;
  for (let i = 0; i <= numGridLines; i++) {
    const y = PAD.top + (chartH / numGridLines) * i;
    ctx.beginPath();
    ctx.moveTo(PAD.left, y);
    ctx.lineTo(W - PAD.right, y);
    ctx.stroke();

    const price = maxPrice - ((maxPrice - minPrice) / numGridLines) * i;
    ctx.fillStyle = textColor;
    ctx.font = '10px Inter, system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(price.toFixed(4), W - PAD.right + 8, y + 4);
  }

  // Drift fill between forecast and live
  if (forecastPoints.length >= 2) {
    ctx.beginPath();
    dataPoints.forEach((d, i) => {
      if (d.forecast_price != null && d.live_price != null) {
        const x = xScale(new Date(d.timestamp));
        const y = yScale(d.forecast_price);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
    });
    for (let i = dataPoints.length - 1; i >= 0; i--) {
      const d = dataPoints[i];
      if (d.forecast_price != null && d.live_price != null) {
        const x = xScale(new Date(d.timestamp));
        const y = yScale(d.live_price);
        ctx.lineTo(x, y);
      }
    }
    ctx.closePath();
    ctx.fillStyle = dark ? 'rgba(139, 92, 246, 0.08)' : 'rgba(139, 92, 246, 0.06)';
    ctx.fill();
  }

  // Forecast price line (dashed orange)
  const forecastPts = dataPoints.filter((d) => d.forecast_price != null);
  if (forecastPts.length >= 2) {
    ctx.beginPath();
    ctx.strokeStyle = '#f97316';
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    ctx.setLineDash([5, 3]);
    forecastPts.forEach((d, i) => {
      const x = xScale(new Date(d.timestamp));
      const y = yScale(d.forecast_price);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Live price line (solid blue)
  ctx.beginPath();
  ctx.strokeStyle = '#2563eb';
  ctx.lineWidth = 2;
  ctx.lineJoin = 'round';
  livePoints.forEach((d, i) => {
    const x = xScale(new Date(d.timestamp));
    const y = yScale(d.live_price);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Dots at last points
  const dotStroke = dark ? '#111827' : '#ffffff';
  const lastLive = livePoints[livePoints.length - 1];
  ctx.beginPath();
  ctx.arc(xScale(new Date(lastLive.timestamp)), yScale(lastLive.live_price), 4, 0, Math.PI * 2);
  ctx.fillStyle = '#2563eb';
  ctx.fill();
  ctx.strokeStyle = dotStroke;
  ctx.lineWidth = 2;
  ctx.stroke();

  if (forecastPts.length > 0) {
    const lastForecast = forecastPts[forecastPts.length - 1];
    ctx.beginPath();
    ctx.arc(xScale(new Date(lastForecast.timestamp)), yScale(lastForecast.forecast_price), 4, 0, Math.PI * 2);
    ctx.fillStyle = '#f97316';
    ctx.fill();
    ctx.strokeStyle = dotStroke;
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  // Time labels on x-axis
  ctx.fillStyle = textColor;
  ctx.font = '10px Inter, system-ui, sans-serif';
  ctx.textAlign = 'center';
  const labelStep = Math.max(1, Math.floor(dataPoints.length / 8));
  for (let i = 0; i < dataPoints.length; i += labelStep) {
    const d = dataPoints[i];
    ctx.fillText(formatTime(d.timestamp), xScale(new Date(d.timestamp)), H - PAD.bottom + 20);
  }
  const lastD = dataPoints[dataPoints.length - 1];
  ctx.fillText(formatTime(lastD.timestamp), xScale(new Date(lastD.timestamp)), H - PAD.bottom + 20);

  // Y-axis label
  ctx.save();
  ctx.translate(12, PAD.top + chartH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = textColor;
  ctx.font = '11px Inter, system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Price', 0, 0);
  ctx.restore();
}

// ---------------------------------------------------------------------------
// Empty chart helper
// ---------------------------------------------------------------------------

function drawEmptyChart(canvas, theme, message) {
  if (!canvas) return;
  const dark = theme ?? isDarkMode();
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();

  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  ctx.fillStyle = dark ? '#111827' : '#ffffff';
  ctx.fillRect(0, 0, rect.width, rect.height);
  ctx.fillStyle = dark ? '#9ca3af' : '#6b7280';
  ctx.font = '13px Inter, system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(message || 'No data', rect.width / 2, rect.height / 2);
}

// ---------------------------------------------------------------------------
// Main Chart Page component
// ---------------------------------------------------------------------------

export default function ChartPage() {
  const [pairs, setPairs] = useState([]);
  const [selectedPair, setSelectedPair] = useState('');
  const [selectedTimeframe, setSelectedTimeframe] = useState('1h');
  const [subView, setSubView] = useState('predictions');
  const [pairsLoading, setPairsLoading] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Predictions state
  const [predictions, setPredictions] = useState(null);

  // Forecast / WebSocket state
  const [wsConnected, setWsConnected] = useState(false);
  const [wsData, setWsData] = useState([]);
  const [wsError, setWsError] = useState(null);
  const [forecastMeta, setForecastMeta] = useState(null);
  const wsRef = useRef(null);

  const predCanvasRef = useRef(null);
  const predDriftCanvasRef = useRef(null);
  const forecastCanvasRef = useRef(null);
  const forecastDriftCanvasRef = useRef(null);

  // Fetch pairs on mount
  useEffect(() => {
    let cancelled = false;
    async function fetchPairs() {
      try {
        const data = await getPairs();
        if (!cancelled) {
          setPairs(data.pairs || []);
          if (data.pairs?.length > 0) {
            setSelectedPair(data.pairs[0]);
          }
        }
      } catch (err) {
        if (!cancelled) setError(err.message);
      } finally {
        if (!cancelled) setPairsLoading(false);
      }
    }
    fetchPairs();
    return () => { cancelled = true; };
  }, []);

  // Fetch predictions data for the Predictions sub-view
  const fetchPredictionsData = useCallback(async (pair, tf) => {
    setLoading(true);
    setError(null);
    setPredictions(null);
    try {
      const data = await getAccuracyPredictions(
        pair || undefined,
        tf || undefined,
        undefined,
        1,
        200
      );
      setPredictions(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  // When sub-view or pair/timeframe changes, fetch or connect
  useEffect(() => {
    if (!selectedPair) return;

    // Close any existing WebSocket
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setWsConnected(false);
    setWsData([]);
    setWsError(null);
    setForecastMeta(null);

    if (subView === 'predictions') {
      fetchPredictionsData(selectedPair, selectedTimeframe);
    }
    // Forecast WebSocket is connected via a separate effect below
  }, [subView, selectedPair, selectedTimeframe, fetchPredictionsData]);

  // WebSocket connection for Forecast sub-view
  useEffect(() => {
    if (subView !== 'forecast' || !selectedPair) return;

    // Close existing
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    const ws = createChartWebSocket(selectedPair, selectedTimeframe);
    wsRef.current = ws;

    ws.onopen = () => {
      setWsConnected(true);
      setWsError(null);
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'tick' && msg.data) {
          setWsData((prev) => {
            const next = [...prev, msg.data];
            // Keep max 200 points
            return next.length > 200 ? next.slice(-200) : next;
          });
          if (msg.data.forecast_direction) {
            setForecastMeta({
              direction: msg.data.forecast_direction,
              confidence: msg.data.forecast_confidence,
            });
          }
        } else if (msg.type === 'error') {
          setWsError(msg.message);
        }
      } catch {
        // ignore malformed messages
      }
    };

    ws.onerror = () => {
      setWsConnected(false);
      setWsError('WebSocket connection error');
    };

    ws.onclose = () => {
      setWsConnected(false);
    };

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [subView, selectedPair, selectedTimeframe]);

  // ---- Draw Predictions charts ----
  useEffect(() => {
    if (predictions?.predictions && predCanvasRef.current) {
      drawPredictionsChart(predCanvasRef.current, predictions.predictions);
    }
  }, [predictions]);

  useEffect(() => {
    if (predictions?.predictions) {
      const driftPoints = predictions.predictions
        .filter((p) => p.predicted_at && p.percentage_drift != null)
        .sort((a, b) => new Date(a.predicted_at) - new Date(b.predicted_at))
        .map((p) => ({
          date: formatDate(p.predicted_at),
          drift: p.percentage_drift * 100,
        }));
      drawDriftChart(predDriftCanvasRef.current, driftPoints);
    }
  }, [predictions]);

  // ---- Draw Forecast charts ----
  useEffect(() => {
    if (wsData.length > 0 && forecastCanvasRef.current) {
      drawForecastChart(forecastCanvasRef.current, wsData);
    } else if (forecastCanvasRef.current) {
      drawEmptyChart(forecastCanvasRef.current, null, 'Connecting to live feed…');
    }
  }, [wsData]);

  useEffect(() => {
    if (wsData.length > 0) {
      const driftPoints = wsData
        .filter((d) => d.drift_pct != null)
        .map((d) => ({
          date: formatTime(d.timestamp),
          drift: d.drift_pct,
        }));
      drawDriftChart(forecastDriftCanvasRef.current, driftPoints);
    } else if (forecastDriftCanvasRef.current) {
      drawEmptyChart(forecastDriftCanvasRef.current, null, 'Waiting for drift data…');
    }
  }, [wsData]);

  // ---- Redraw on resize / theme change ----
  useEffect(() => {
    function handleResize() {
      if (predictions?.predictions && predCanvasRef.current) {
        drawPredictionsChart(predCanvasRef.current, predictions.predictions);
      }
      if (predictions?.predictions) {
        const driftPoints = predictions.predictions
          .filter((p) => p.predicted_at && p.percentage_drift != null)
          .sort((a, b) => new Date(a.predicted_at) - new Date(b.predicted_at))
          .map((p) => ({
            date: formatDate(p.predicted_at),
            drift: p.percentage_drift * 100,
          }));
        drawDriftChart(predDriftCanvasRef.current, driftPoints);
      }
      if (wsData.length > 0 && forecastCanvasRef.current) {
        drawForecastChart(forecastCanvasRef.current, wsData);
      }
      if (wsData.length > 0) {
        const driftPoints = wsData
          .filter((d) => d.drift_pct != null)
          .map((d) => ({
            date: formatTime(d.timestamp),
            drift: d.drift_pct,
          }));
        drawDriftChart(forecastDriftCanvasRef.current, driftPoints);
      }
    }
    function handleThemeChange() {
      setTimeout(handleResize, 50);
    }
    window.addEventListener('resize', handleResize);
    const observer = new MutationObserver(handleThemeChange);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
    return () => {
      window.removeEventListener('resize', handleResize);
      observer.disconnect();
    };
  }, [predictions, wsData]);

  // ---- Render ----
  if (pairsLoading) return <Spinner className="py-20" />;

  return (
    <div>
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">Chart</h1>
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
          Visualise prediction accuracy and live forecast performance.
        </p>
      </div>

      {/* Controls */}
      <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-5 shadow-sm mb-6">
        <div className="flex flex-col sm:flex-row gap-4 items-end">
          {/* Sub-view toggle */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              View
            </label>
            <div className="flex rounded-lg overflow-hidden border border-gray-300 dark:border-gray-600">
              {SUB_VIEWS.map((sv) => (
                <button
                  key={sv.value}
                  onClick={() => setSubView(sv.value)}
                  className={`px-4 py-2 text-sm font-medium transition-colors ${
                    subView === sv.value
                      ? 'bg-primary-600 text-white'
                      : 'bg-white dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-600'
                  }`}
                >
                  {sv.label}
                </button>
              ))}
            </div>
          </div>

          {/* Pair selector */}
          <div className="flex-1">
            <label htmlFor="chart-pair-select" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Forex Pair
            </label>
            <select
              id="chart-pair-select"
              value={selectedPair}
              onChange={(e) => setSelectedPair(e.target.value)}
              disabled={loading}
              className="block w-full max-w-xs rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 px-4 py-2.5 text-sm focus:border-primary-500 focus:ring-2 focus:ring-primary-200 dark:focus:ring-primary-800 focus:outline-none disabled:opacity-50"
            >
              {pairs.map((pair) => (
                <option key={pair} value={pair}>
                  {formatPair(pair)}
                </option>
              ))}
            </select>
          </div>

          {/* Timeframe toggle */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Timeframe
            </label>
            <div className="flex rounded-lg overflow-hidden border border-gray-300 dark:border-gray-600">
              {TIMEFRAMES.map((tf) => (
                <button
                  key={tf.value}
                  onClick={() => setSelectedTimeframe(tf.value)}
                  className={`px-3 py-2 text-sm font-medium transition-colors ${
                    selectedTimeframe === tf.value
                      ? 'bg-primary-600 text-white'
                      : 'bg-white dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-600'
                  }`}
                >
                  {tf.label}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Loading banner */}
      {loading && subView === 'predictions' && (
        <div className="rounded-xl border border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-900/20 p-5 mb-6">
          <div className="flex items-center gap-3">
            <Spinner className="py-0" />
            <p className="text-sm text-blue-700 dark:text-blue-200">
              Loading prediction data for {formatPair(selectedPair)}…
            </p>
          </div>
        </div>
      )}

      {/* Error */}
      {error && !loading && <ErrorMessage message={error} />}

      {/* ==================== PREDICTIONS SUB-VIEW ==================== */}
      {subView === 'predictions' && !loading && !error && (
        <>
          {/* Predicted vs Actual Chart */}
          <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 shadow-sm mb-6">
            <div className="px-5 py-4 border-b border-gray-200 dark:border-gray-700 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
              <h2 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
                Predicted vs Actual Prices
              </h2>
              <div className="flex items-center gap-4 text-xs text-gray-500 dark:text-gray-400">
                <span className="flex items-center gap-1.5">
                  <span
                    className="h-0.5 w-4 inline-block"
                    style={{ borderTop: '2px dashed #f97316', background: 'none', height: 0 }}
                  />
                  Predicted
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="h-0.5 w-4 bg-blue-600 inline-block" />
                  Actual
                </span>
              </div>
            </div>
            <div className="p-4">
              <canvas
                ref={predCanvasRef}
                style={{ width: '100%', height: '400px' }}
                aria-label="Chart comparing predicted prices against actual market prices"
                role="img"
              />
            </div>
          </div>

          {/* Drift Chart */}
          <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 shadow-sm mb-6">
            <div className="px-5 py-4 border-b border-gray-200 dark:border-gray-700">
              <h2 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
                Percentage Drift (Predicted vs Actual)
              </h2>
            </div>
            <div className="p-4">
              <canvas
                ref={predDriftCanvasRef}
                style={{ width: '100%', height: '250px' }}
                aria-label="Chart showing percentage drift between predicted and actual prices"
                role="img"
              />
            </div>
          </div>

          {/* Predictions Table */}
          {predictions?.predictions?.length > 0 && (
            <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 shadow-sm overflow-hidden mb-6">
              <div className="px-5 py-4 border-b border-gray-200 dark:border-gray-700">
                <h2 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
                  Prediction Records
                  <span className="ml-2 text-xs font-normal text-gray-500 dark:text-gray-400">
                    {predictions.total?.toLocaleString()} total
                  </span>
                </h2>
              </div>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                  <thead className="bg-gray-50 dark:bg-gray-700/50">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Date</th>
                      <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Predicted</th>
                      <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Actual</th>
                      <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Drift</th>
                      <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                    {predictions.predictions.map((pred) => (
                      <tr key={pred.id} className="hover:bg-gray-50 dark:hover:bg-gray-700/50">
                        <td className="px-4 py-3 text-sm text-gray-900 dark:text-gray-100 whitespace-nowrap">
                          {pred.predicted_at ? new Date(pred.predicted_at).toLocaleString() : '—'}
                        </td>
                        <td className="px-4 py-3 text-sm text-right text-gray-900 dark:text-gray-100 font-mono">
                          {pred.target_price != null ? pred.target_price.toFixed(4) : '—'}
                        </td>
                        <td className="px-4 py-3 text-sm text-right text-gray-900 dark:text-gray-100 font-mono">
                          {pred.actual_price != null ? pred.actual_price.toFixed(4) : '—'}
                        </td>
                        <td className="px-4 py-3 text-sm text-right font-mono">
                          {pred.percentage_drift != null ? (
                            <span className={pred.percentage_drift >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}>
                              {formatDrift(pred.percentage_drift)}
                            </span>
                          ) : (
                            <span className="text-gray-400">—</span>
                          )}
                        </td>
                        <td className="px-4 py-3 text-center whitespace-nowrap">
                          <span
                            className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium ${
                              pred.status === 'correct'
                                ? 'bg-green-50 dark:bg-green-900/30 text-green-700 dark:text-green-400 ring-1 ring-inset ring-green-600/20 dark:ring-green-500/30'
                                : pred.status === 'incorrect'
                                  ? 'bg-red-50 dark:bg-red-900/30 text-red-700 dark:text-red-400 ring-1 ring-inset ring-red-600/20 dark:ring-red-500/30'
                                  : 'bg-yellow-50 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400 ring-1 ring-inset ring-yellow-600/20 dark:ring-yellow-500/30'
                            }`}
                          >
                            <span
                              className={`h-1.5 w-1.5 rounded-full ${
                                pred.status === 'correct'
                                  ? 'bg-green-500 dark:bg-green-400'
                                  : pred.status === 'incorrect'
                                    ? 'bg-red-500 dark:bg-red-400'
                                    : 'bg-yellow-500 dark:bg-yellow-400'
                              }`}
                            />
                            {pred.status}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Empty state for predictions */}
          {predictions && predictions.predictions?.length === 0 && (
            <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-12 shadow-sm text-center">
              <svg xmlns="http://www.w3.org/2000/svg" className="mx-auto h-12 w-12 text-gray-400 dark:text-gray-500" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 0 1 3 19.875v-6.75ZM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125V8.625ZM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125V4.125Z" />
              </svg>
              <h3 className="mt-4 text-sm font-semibold text-gray-900 dark:text-gray-100">No Prediction Data</h3>
              <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
                No prediction records found for {formatPair(selectedPair)} with the selected timeframe.
              </p>
            </div>
          )}
        </>
      )}

      {/* ==================== FORECAST SUB-VIEW ==================== */}
      {subView === 'forecast' && (
        <>
          {/* Connection status */}
          <div className="mb-4 flex items-center gap-3">
            <span className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium ${
              wsConnected
                ? 'bg-green-50 dark:bg-green-900/30 text-green-700 dark:text-green-400 ring-1 ring-inset ring-green-600/20'
                : wsError
                  ? 'bg-red-50 dark:bg-red-900/30 text-red-700 dark:text-red-400 ring-1 ring-inset ring-red-600/20'
                  : 'bg-yellow-50 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400 ring-1 ring-inset ring-yellow-600/20'
            }`}>
              <span className={`h-2 w-2 rounded-full ${wsConnected ? 'bg-green-500 animate-pulse' : wsError ? 'bg-red-500' : 'bg-yellow-500 animate-pulse'}`} />
              {wsConnected ? 'Live' : wsError ? 'Disconnected' : 'Connecting…'}
            </span>
            {forecastMeta && (
              <span className="text-xs text-gray-500 dark:text-gray-400">
                Forecast direction: <span className="font-medium text-gray-900 dark:text-gray-100">{forecastMeta.direction}</span>
                {' · '}Confidence: <span className="font-medium text-gray-900 dark:text-gray-100">{forecastMeta.confidence != null ? `${(forecastMeta.confidence * 100).toFixed(1)}%` : '—'}</span>
              </span>
            )}
            {wsError && (
              <span className="text-xs text-red-600 dark:text-red-400">{wsError}</span>
            )}
          </div>

          {/* Forecast vs Live Chart */}
          <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 shadow-sm mb-6">
            <div className="px-5 py-4 border-b border-gray-200 dark:border-gray-700 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
              <h2 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
                Forecast vs Live Prices
              </h2>
              <div className="flex items-center gap-4 text-xs text-gray-500 dark:text-gray-400">
                <span className="flex items-center gap-1.5">
                  <span className="h-0.5 w-4 bg-blue-600 inline-block" />
                  Live
                </span>
                <span className="flex items-center gap-1.5">
                  <span
                    className="h-0.5 w-4 inline-block"
                    style={{ borderTop: '2px dashed #f97316', background: 'none', height: 0 }}
                  />
                  Forecast
                </span>
              </div>
            </div>
            <div className="p-4">
              <canvas
                ref={forecastCanvasRef}
                style={{ width: '100%', height: '400px' }}
                aria-label="Live chart comparing forecasted prices against live market prices"
                role="img"
              />
            </div>
          </div>

          {/* Live Drift Chart */}
          <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 shadow-sm mb-6">
            <div className="px-5 py-4 border-b border-gray-200 dark:border-gray-700">
              <h2 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
                Percentage Drift (Forecast vs Live)
              </h2>
            </div>
            <div className="p-4">
              <canvas
                ref={forecastDriftCanvasRef}
                style={{ width: '100%', height: '250px' }}
                aria-label="Chart showing percentage drift between forecasted and live prices"
                role="img"
              />
            </div>
          </div>

          {/* Live data points summary */}
          {wsData.length > 0 && (
            <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-5 shadow-sm mb-6">
              <div className="grid gap-4 sm:grid-cols-3">
                <div className="rounded-lg bg-gray-50 dark:bg-gray-900/50 p-4">
                  <p className="text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">Latest Live Price</p>
                  <p className="mt-1 text-xl font-bold text-gray-900 dark:text-gray-100 font-mono">
                    {wsData[wsData.length - 1]?.live_price?.toFixed(4) ?? '—'}
                  </p>
                </div>
                <div className="rounded-lg bg-gray-50 dark:bg-gray-900/50 p-4">
                  <p className="text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">Forecast Price</p>
                  <p className="mt-1 text-xl font-bold text-gray-900 dark:text-gray-100 font-mono">
                    {wsData[wsData.length - 1]?.forecast_price?.toFixed(4) ?? '—'}
                  </p>
                </div>
                <div className="rounded-lg bg-gray-50 dark:bg-gray-900/50 p-4">
                  <p className="text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">Current Drift</p>
                  <p className={`mt-1 text-xl font-bold font-mono ${
                    wsData[wsData.length - 1]?.drift_pct >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
                  }`}>
                    {formatDriftAbs(wsData[wsData.length - 1]?.drift_pct)}
                  </p>
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
