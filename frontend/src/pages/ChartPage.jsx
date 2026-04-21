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
  ctx.fillStyle = dark ? '#020617' : '#ffffff';
  ctx.fillRect(0, 0, W, H);

  // Grid lines
  const gridColor = dark ? '#1e293b' : '#f3f4f6';
  const textColor = dark ? '#64748b' : '#9ca3af';
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

  // Real price line (solid green) — matches mock
  ctx.beginPath();
  ctx.strokeStyle = dark ? '#22c55e' : '#16a34a';
  ctx.lineWidth = 2;
  ctx.lineJoin = 'round';
  points.forEach((p, i) => {
    const x = xScale(p.date);
    const y = yScale(p.actual);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Predicted price line (solid blue) — matches mock
  ctx.beginPath();
  ctx.strokeStyle = dark ? '#3b82f6' : '#2563eb';
  ctx.lineWidth = 2;
  ctx.lineJoin = 'round';
  points.forEach((p, i) => {
    const x = xScale(p.date);
    const y = yScale(p.predicted);
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
  ctx.fillStyle = dark ? 'rgba(99, 102, 241, 0.08)' : 'rgba(59, 130, 246, 0.06)';
  ctx.fill();

  // Dots at last points
  const lastPoint = points[points.length - 1];
  const dotStroke = dark ? '#020617' : '#ffffff';

  ctx.beginPath();
  ctx.arc(xScale(lastPoint.date), yScale(lastPoint.predicted), 4, 0, Math.PI * 2);
  ctx.fillStyle = dark ? '#3b82f6' : '#2563eb';
  ctx.fill();
  ctx.strokeStyle = dotStroke;
  ctx.lineWidth = 2;
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(xScale(lastPoint.date), yScale(lastPoint.actual), 4, 0, Math.PI * 2);
  ctx.fillStyle = dark ? '#22c55e' : '#16a34a';
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
  ctx.fillStyle = dark ? '#020617' : '#ffffff';
  ctx.fillRect(0, 0, W, H);

  // Grid lines
  const gridColor = dark ? '#1e293b' : '#f3f4f6';
  const textColor = dark ? '#64748b' : '#9ca3af';
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
  ctx.strokeStyle = dark ? '#334155' : '#d1d5db';
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
  gradient.addColorStop(0, 'rgba(34, 197, 94, 0.12)');
  gradient.addColorStop(0.5, 'rgba(34, 197, 94, 0.02)');
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
  const dotStroke = dark ? '#020617' : '#ffffff';
  validPoints.forEach((p, i) => {
    ctx.beginPath();
    ctx.arc(xScale(i), yScale(p.drift), 3, 0, Math.PI * 2);
    ctx.fillStyle = p.drift >= 0 ? '#22c55e' : '#ef4444';
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
  ctx.fillStyle = dark ? '#020617' : '#ffffff';
  ctx.fillRect(0, 0, W, H);

  // Grid lines
  const gridColor = dark ? '#1e293b' : '#f3f4f6';
  const textColor = dark ? '#64748b' : '#9ca3af';
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
    ctx.fillStyle = dark ? 'rgba(99, 102, 241, 0.08)' : 'rgba(59, 130, 246, 0.06)';
    ctx.fill();
  }

  // Forecast price line (solid blue)
  const forecastPts = dataPoints.filter((d) => d.forecast_price != null);
  if (forecastPts.length >= 2) {
    ctx.beginPath();
    ctx.strokeStyle = dark ? '#3b82f6' : '#2563eb';
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    forecastPts.forEach((d, i) => {
      const x = xScale(new Date(d.timestamp));
      const y = yScale(d.forecast_price);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  // Live price line (solid green) — matches mock
  ctx.beginPath();
  ctx.strokeStyle = dark ? '#22c55e' : '#16a34a';
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
  const dotStroke = dark ? '#020617' : '#ffffff';
  const lastLive = livePoints[livePoints.length - 1];
  ctx.beginPath();
  ctx.arc(xScale(new Date(lastLive.timestamp)), yScale(lastLive.live_price), 4, 0, Math.PI * 2);
  ctx.fillStyle = dark ? '#22c55e' : '#16a34a';
  ctx.fill();
  ctx.strokeStyle = dotStroke;
  ctx.lineWidth = 2;
  ctx.stroke();

  if (forecastPts.length > 0) {
    const lastForecast = forecastPts[forecastPts.length - 1];
    ctx.beginPath();
    ctx.arc(xScale(new Date(lastForecast.timestamp)), yScale(lastForecast.forecast_price), 4, 0, Math.PI * 2);
    ctx.fillStyle = dark ? '#3b82f6' : '#2563eb';
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

  ctx.fillStyle = dark ? '#020617' : '#ffffff';
  ctx.fillRect(0, 0, rect.width, rect.height);
  ctx.fillStyle = dark ? '#64748b' : '#9ca3af';
  ctx.font = '13px Inter, system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(message || 'No data', rect.width / 2, rect.height / 2);
}

// ---------------------------------------------------------------------------
// Side Panel Cards
// ---------------------------------------------------------------------------

function StatCard({ label, value, valueClass }) {
  return (
    <div className="rounded-xl bg-slate-900/60 dark:bg-slate-800/40 p-4">
      <h3 className="text-xs font-medium uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-2">
        {label}
      </h3>
      <div className={`text-xl font-bold font-mono ${valueClass || 'text-slate-100'}`}>
        {value}
      </div>
    </div>
  );
}

function SignalBadge({ direction, confidence }) {
  const isBuy = direction?.toUpperCase() === 'BUY' || direction?.toUpperCase() === 'UP';
  const isSell = direction?.toUpperCase() === 'SELL' || direction?.toUpperCase() === 'DOWN';

  return (
    <div className="rounded-xl bg-slate-900/60 dark:bg-slate-800/40 p-4">
      <h3 className="text-xs font-medium uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-3">
        Last Signal
      </h3>
      {direction ? (
        <>
          <div className={`text-center py-2.5 px-4 rounded-lg font-bold text-sm uppercase tracking-wide ${
            isBuy
              ? 'bg-green-600/20 text-green-400 ring-1 ring-green-500/30'
              : isSell
                ? 'bg-red-600/20 text-red-400 ring-1 ring-red-500/30'
                : 'bg-slate-700/40 text-slate-300 ring-1 ring-slate-600/30'
          }`}>
            {direction}
          </div>
          {confidence != null && (
            <p className="mt-2 text-center text-xs text-slate-500 dark:text-slate-400">
              Confidence: {(confidence * 100).toFixed(1)}%
            </p>
          )}
        </>
      ) : (
        <p className="text-sm text-slate-500 dark:text-slate-400">No signal</p>
      )}
    </div>
  );
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
        100
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

  // ---- Derived stats ----
  const predStats = (() => {
    if (!predictions?.predictions?.length) return null;
    const preds = predictions.predictions;
    const evaluated = preds.filter(p => p.status === 'correct' || p.status === 'incorrect');
    const accuracy = evaluated.length > 0
      ? ((evaluated.filter(p => p.status === 'correct').length / evaluated.length) * 100).toFixed(1)
      : '—';
    const latestPred = preds[0];
    const lastSignal = latestPred?.status === 'correct' ? 'Correct' : latestPred?.status === 'incorrect' ? 'Incorrect' : 'Pending';
    const lastDrift = latestPred?.percentage_drift;
    return { accuracy, lastSignal, lastDrift, total: predictions.total || preds.length };
  })();

  const forecastStats = (() => {
    if (wsData.length === 0) return null;
    const latest = wsData[wsData.length - 1];
    return {
      livePrice: latest.live_price,
      forecastPrice: latest.forecast_price,
      drift: latest.drift_pct,
      direction: forecastMeta?.direction,
      confidence: forecastMeta?.confidence,
    };
  })();

  // ---- Render ----
  if (pairsLoading) return <Spinner className="py-20" />;

  // Active stats for side panel
  const stats = subView === 'predictions' ? predStats : forecastStats;

  return (
    <div className="flex flex-col h-[calc(100vh-4rem)] -m-4 sm:-m-6 lg:-m-8">
      {/* Top Bar — matches mock's topbar */}
      <div className="flex items-center justify-between px-4 sm:px-6 py-3 bg-slate-900 dark:bg-slate-950 border-b border-slate-700 dark:border-slate-800 shrink-0">
        <h1 className="text-base sm:text-lg font-semibold text-slate-100 dark:text-slate-200">
          Model vs Market
        </h1>
        <div className="flex items-center gap-2 sm:gap-3 overflow-x-auto">
          {/* Sub-view toggle */}
          {SUB_VIEWS.map((sv) => (
            <button
              key={sv.value}
              onClick={() => setSubView(sv.value)}
              className={`px-3 py-1.5 text-xs sm:text-sm font-medium rounded-md transition-colors whitespace-nowrap ${
                subView === sv.value
                  ? 'bg-blue-600 text-white'
                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700 dark:bg-slate-800 dark:text-slate-400 dark:hover:bg-slate-700'
              }`}
            >
              {sv.label}
            </button>
          ))}

          <div className="w-px h-5 bg-slate-700 dark:bg-slate-700 mx-1" />

          {/* Pair buttons — matches mock's pair selector */}
          {pairs.slice(0, 6).map((pair) => (
            <button
              key={pair}
              onClick={() => setSelectedPair(pair)}
              className={`px-3 py-1.5 text-xs sm:text-sm font-medium rounded-md transition-colors whitespace-nowrap ${
                selectedPair === pair
                  ? 'bg-blue-600 text-white'
                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700 dark:bg-slate-800 dark:text-slate-400 dark:hover:bg-slate-700'
              }`}
            >
              {formatPair(pair)}
            </button>
          ))}

          {pairs.length > 6 && (
            <select
              value={selectedPair}
              onChange={(e) => setSelectedPair(e.target.value)}
              className="bg-slate-800 text-slate-300 text-xs sm:text-sm rounded-md px-2 py-1.5 border border-slate-700 focus:border-blue-500 focus:outline-none dark:bg-slate-800 dark:text-slate-400"
            >
              {pairs.map((pair) => (
                <option key={pair} value={pair}>
                  {formatPair(pair)}
                </option>
              ))}
            </select>
          )}

          <div className="w-px h-5 bg-slate-700 dark:bg-slate-700 mx-1" />

          {/* Timeframe buttons */}
          {TIMEFRAMES.map((tf) => (
            <button
              key={tf.value}
              onClick={() => setSelectedTimeframe(tf.value)}
              className={`px-2.5 py-1.5 text-xs sm:text-sm font-medium rounded-md transition-colors whitespace-nowrap ${
                selectedTimeframe === tf.value
                  ? 'bg-blue-600 text-white'
                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700 dark:bg-slate-800 dark:text-slate-400 dark:hover:bg-slate-700'
              }`}
            >
              {tf.label}
            </button>
          ))}
        </div>
      </div>

      {/* Main Layout — chart area + side panel, matches mock's flex layout */}
      <div className="flex flex-1 min-h-0 bg-slate-950 dark:bg-[#020617]">
        {/* Chart Area — flex:3 in mock */}
        <div className="flex-3 p-3 sm:p-5 flex flex-col min-w-0">
          <div className="bg-slate-900 dark:bg-[#020617] rounded-xl flex-1 flex flex-col p-3 sm:p-4 min-h-0 border border-slate-800 dark:border-slate-800/50">
            {/* Chart Header */}
            <div className="flex items-center justify-between mb-2 sm:mb-3 shrink-0">
              <div className="text-sm font-medium text-slate-300 dark:text-slate-400">
                {subView === 'predictions' ? 'Price vs Prediction' : 'Forecast vs Live'}
              </div>
              <div className="flex items-center gap-4 text-xs text-slate-400 dark:text-slate-500">
                <span className="flex items-center gap-1.5">
                  <span className="h-2.5 w-2.5 rounded-full bg-green-500 inline-block" />
                  {subView === 'predictions' ? 'Real Price' : 'Live'}
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="h-2.5 w-2.5 rounded-full bg-blue-500 inline-block" />
                  {subView === 'predictions' ? 'Prediction' : 'Forecast'}
                </span>
              </div>
            </div>

            {/* Loading banner */}
            {loading && subView === 'predictions' && (
              <div className="rounded-lg border border-blue-500/20 bg-blue-500/10 p-3 mb-3 shrink-0">
                <div className="flex items-center gap-2">
                  <Spinner className="py-0" />
                  <p className="text-xs text-blue-300">
                    Loading prediction data for {formatPair(selectedPair)}…
                  </p>
                </div>
              </div>
            )}

            {/* Error */}
            {error && !loading && <ErrorMessage message={error} />}

            {/* WebSocket status */}
            {subView === 'forecast' && (
              <div className="flex items-center gap-2 mb-2 shrink-0">
                <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium ${
                  wsConnected
                    ? 'bg-green-500/10 text-green-400 ring-1 ring-green-500/20'
                    : wsError
                      ? 'bg-red-500/10 text-red-400 ring-1 ring-red-500/20'
                      : 'bg-yellow-500/10 text-yellow-400 ring-1 ring-yellow-500/20'
                }`}>
                  <span className={`h-1.5 w-1.5 rounded-full ${wsConnected ? 'bg-green-400 animate-pulse' : wsError ? 'bg-red-400' : 'bg-yellow-400 animate-pulse'}`} />
                  {wsConnected ? 'Live' : wsError ? 'Disconnected' : 'Connecting…'}
                </span>
                {wsError && (
                  <span className="text-xs text-red-400">{wsError}</span>
                )}
              </div>
            )}

            {/* Canvas — fills remaining space */}
            <div className="flex-1 min-h-0">
              {subView === 'predictions' ? (
                <canvas
                  ref={predCanvasRef}
                  style={{ width: '100%', height: '100%' }}
                  aria-label="Chart comparing predicted prices against actual market prices"
                  role="img"
                />
              ) : (
                <canvas
                  ref={forecastCanvasRef}
                  style={{ width: '100%', height: '100%' }}
                  aria-label="Live chart comparing forecasted prices against live market prices"
                  role="img"
                />
              )}
            </div>
          </div>

          {/* Drift Chart — below main chart */}
          <div className="bg-slate-900 dark:bg-[#020617] rounded-xl mt-3 flex flex-col p-3 sm:p-4 border border-slate-800 dark:border-slate-800/50"
            style={{ height: '180px', minHeight: '140px' }}
          >
            <div className="text-sm font-medium text-slate-300 dark:text-slate-400 mb-2 shrink-0">
              Percentage Drift
            </div>
            <div className="flex-1 min-h-0">
              {subView === 'predictions' ? (
                <canvas
                  ref={predDriftCanvasRef}
                  style={{ width: '100%', height: '100%' }}
                  aria-label="Chart showing percentage drift between predicted and actual prices"
                  role="img"
                />
              ) : (
                <canvas
                  ref={forecastDriftCanvasRef}
                  style={{ width: '100%', height: '100%' }}
                  aria-label="Chart showing percentage drift between forecasted and live prices"
                  role="img"
                />
              )}
            </div>
          </div>
        </div>

        {/* Side Panel — matches mock's side-panel */}
        <div className="w-64 shrink-0 bg-slate-950 dark:bg-[#020617] border-l border-slate-800 dark:border-slate-800/50 p-4 flex flex-col gap-3 overflow-y-auto hidden lg:flex">
          {subView === 'predictions' && stats ? (
            <>
              <StatCard
                label="Prediction Accuracy"
                value={`${stats.accuracy}%`}
                valueClass={Number(stats.accuracy) >= 60 ? 'text-green-400' : Number(stats.accuracy) < 50 ? 'text-red-400' : 'text-slate-100'}
              />
              <StatCard
                label="Total Predictions"
                value={stats.total?.toLocaleString()}
              />
              <SignalBadge direction={stats.lastSignal} />
              <StatCard
                label="Last Deviation"
                value={stats.lastDrift != null ? formatDrift(stats.lastDrift) : '—'}
                valueClass={stats.lastDrift >= 0 ? 'text-green-400' : 'text-red-400'}
              />
            </>
          ) : subView === 'forecast' && forecastStats ? (
            <>
              <StatCard
                label="Latest Live Price"
                value={forecastStats.livePrice?.toFixed(4) ?? '—'}
              />
              <StatCard
                label="Forecast Price"
                value={forecastStats.forecastPrice?.toFixed(4) ?? '—'}
                valueClass="text-blue-400"
              />
              <SignalBadge direction={forecastStats.direction} confidence={forecastStats.confidence} />
              <StatCard
                label="Current Drift"
                value={forecastStats.drift != null ? formatDriftAbs(forecastStats.drift) : '—'}
                valueClass={forecastStats.drift >= 0 ? 'text-green-400' : 'text-red-400'}
              />
            </>
          ) : (
            <>
              <div className="rounded-xl bg-slate-900/60 dark:bg-slate-800/40 p-4">
                <h3 className="text-xs font-medium uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-2">
                  {subView === 'predictions' ? 'Accuracy' : 'Status'}
                </h3>
                <div className="text-slate-500 dark:text-slate-400 text-sm">
                  {subView === 'predictions' ? 'Loading data…' : 'Waiting for live feed…'}
                </div>
              </div>
            </>
          )}

          {/* Predictions Table (in side panel, scrollable) */}
          {subView === 'predictions' && predictions?.predictions?.length > 0 && (
            <div className="flex-1 min-h-0 rounded-xl bg-slate-900/60 dark:bg-slate-800/40 p-3 flex flex-col overflow-hidden mt-auto">
              <h3 className="text-xs font-medium uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-2 shrink-0">
                Recent Records
              </h3>
              <div className="flex-1 overflow-y-auto space-y-1.5">
                {predictions.predictions.slice(0, 20).map((pred) => (
                  <div key={pred.id} className="flex items-center justify-between text-xs py-1.5 px-2 rounded-lg hover:bg-slate-800/40">
                    <span className="text-slate-400 dark:text-slate-500 whitespace-nowrap">
                      {pred.predicted_at ? formatDate(pred.predicted_at) : '—'}
                    </span>
                    <span className={`font-mono font-medium ${pred.percentage_drift >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {formatDrift(pred.percentage_drift)}
                    </span>
                    <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-medium ${
                      pred.status === 'correct'
                        ? 'bg-green-500/10 text-green-400'
                        : pred.status === 'incorrect'
                          ? 'bg-red-500/10 text-red-400'
                          : 'bg-yellow-500/10 text-yellow-400'
                    }`}>
                      <span className={`h-1 w-1 rounded-full ${
                        pred.status === 'correct'
                          ? 'bg-green-400'
                          : pred.status === 'incorrect'
                            ? 'bg-red-400'
                            : 'bg-yellow-400'
                      }`} />
                      {pred.status}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
