import { useState, useEffect, useCallback, useRef } from 'react';
import { useSearchParams } from 'react-router-dom';
import { getPairs, getForecast } from '../api';
import Spinner from '../components/Spinner';
import ErrorMessage from '../components/ErrorMessage';

const HORIZON_OPTIONS = [
  { value: 7, label: '1 Week' },
  { value: 14, label: '2 Weeks' },
  { value: 30, label: '1 Month' },
  { value: 60, label: '2 Months' },
  { value: 90, label: '3 Months' },
];

const TIMEFRAME_OPTIONS = [
  { value: '5m', label: '5 Min' },
  { value: '15m', label: '15 Min' },
  { value: '1h', label: '1 Hour' },
  { value: '4h', label: '4 Hours' },
];

function formatPair(ticker) {
  return ticker.replace('=X', '');
}

function formatDate(dateStr) {
  const d = new Date(dateStr);
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

function isDarkMode() {
  return document.documentElement.classList.contains('dark');
}

function drawChart(canvas, data) {
  if (!canvas || !data) return;

  const dark = isDarkMode();
  const historicalData = data.historical_data || data.historicalData || [];
  const forecastData = data.forecast || [];

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

  // Theme-aware colors
  const colors = {
    bg: dark ? '#111827' : '#ffffff',
    histRegion: dark ? 'rgba(30, 58, 138, 0.15)' : 'rgba(239, 246, 255, 0.4)',
    forecastRegion: dark ? 'rgba(120, 53, 15, 0.15)' : 'rgba(255, 247, 237, 0.5)',
    boundaryLine: dark ? '#4b5563' : '#9ca3af',
    gridLine: dark ? '#374151' : '#f3f4f6',
    gridText: dark ? '#9ca3af' : '#9ca3af',
    histLine: '#2563eb',
    forecastLine: '#f97316',
    confidenceBand: dark ? 'rgba(251, 146, 60, 0.15)' : 'rgba(251, 146, 60, 0.12)',
    upperLine: 'rgba(251, 146, 60, 0.3)',
    dotStroke: dark ? '#111827' : '#ffffff',
    labelText: dark ? '#d1d5db' : '#6b7280',
  };

  // Clear
  ctx.clearRect(0, 0, W, H);

  // Background
  ctx.fillStyle = colors.bg;
  ctx.fillRect(0, 0, W, H);

  // Combine all data points
  const allPoints = [
    ...historicalData.map((d) => ({ date: new Date(d.date), price: d.close, type: 'historical' })),
    ...forecastData.map((d) => ({ date: new Date(d.date), price: d.predicted_price, type: 'forecast', upper: d.upper_bound, lower: d.lower_bound })),
  ];

  if (allPoints.length < 2) return;

  const minDate = allPoints[0].date.getTime();
  const maxDate = allPoints[allPoints.length - 1].date.getTime();

  const allPrices = allPoints.flatMap((p) => [p.price, p.upper || p.price, p.lower || p.price]);
  const minPrice = Math.min(...allPrices) * 0.999;
  const maxPrice = Math.max(...allPrices) * 1.001;

  const xScale = (date) => PAD.left + ((date.getTime() - minDate) / (maxDate - minDate)) * chartW;
  const yScale = (price) => PAD.top + chartH - ((price - minPrice) / (maxPrice - minPrice)) * chartH;

  // Forecast background shading
  const boundaryIdx = historicalData.length;
  if (boundaryIdx > 0 && boundaryIdx < allPoints.length) {
    const boundaryX = xScale(allPoints[boundaryIdx].date);

    // Historical region
    ctx.fillStyle = colors.histRegion;
    ctx.fillRect(PAD.left, PAD.top, boundaryX - PAD.left, chartH);

    // Forecast region
    ctx.fillStyle = colors.forecastRegion;
    ctx.fillRect(boundaryX, PAD.top, W - PAD.right - boundaryX, chartH);

    // Vertical boundary line
    ctx.beginPath();
    ctx.setLineDash([6, 4]);
    ctx.strokeStyle = colors.boundaryLine;
    ctx.lineWidth = 1.5;
    ctx.moveTo(boundaryX, PAD.top);
    ctx.lineTo(boundaryX, PAD.top + chartH);
    ctx.stroke();
    ctx.setLineDash([]);

    // Boundary label
    ctx.fillStyle = colors.labelText;
    ctx.font = '11px Inter, system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Forecast →', boundaryX + (W - PAD.right - boundaryX) / 2, PAD.top - 10);
    ctx.fillText('← Historical', PAD.left + (boundaryX - PAD.left) / 2, PAD.top - 10);
  }

  // Grid lines
  ctx.strokeStyle = colors.gridLine;
  ctx.lineWidth = 1;
  const numGridLines = 5;
  for (let i = 0; i <= numGridLines; i++) {
    const y = PAD.top + (chartH / numGridLines) * i;
    ctx.beginPath();
    ctx.moveTo(PAD.left, y);
    ctx.lineTo(W - PAD.right, y);
    ctx.stroke();

    // Price label
    const price = maxPrice - ((maxPrice - minPrice) / numGridLines) * i;
    ctx.fillStyle = colors.gridText;
    ctx.font = '10px Inter, system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(price.toFixed(4), W - PAD.right + 8, y + 4);
  }

  // Confidence band (forecast only)
  if (forecastData.length > 1) {
    ctx.beginPath();
    forecastData.forEach((p, i) => {
      const x = xScale(new Date(p.date));
      const y = yScale(p.upper_bound);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    for (let i = forecastData.length - 1; i >= 0; i--) {
      const x = xScale(new Date(forecastData[i].date));
      const y = yScale(forecastData[i].lower_bound);
      ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fillStyle = colors.confidenceBand;
    ctx.fill();
  }

  // Historical price line
  if (historicalData.length > 1) {
    ctx.beginPath();
    ctx.strokeStyle = colors.histLine;
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    historicalData.forEach((d, i) => {
      const x = xScale(new Date(d.date));
      const y = yScale(d.close);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  // Forecast price line
  if (forecastData.length > 0) {
    ctx.beginPath();
    ctx.strokeStyle = colors.forecastLine;
    ctx.lineWidth = 2.5;
    ctx.lineJoin = 'round';
    ctx.setLineDash([5, 3]);

    // Connect from last historical point to first forecast point (if historical data exists)
    if (historicalData.length > 0) {
      const lastHist = historicalData[historicalData.length - 1];
      const firstForecast = forecastData[0];
      ctx.moveTo(xScale(new Date(lastHist.date)), yScale(lastHist.close));
      ctx.lineTo(xScale(new Date(firstForecast.date)), yScale(firstForecast.predicted_price));
    } else {
      const firstForecast = forecastData[0];
      ctx.moveTo(xScale(new Date(firstForecast.date)), yScale(firstForecast.predicted_price));
    }

    // Continue through forecast points
    for (let i = 1; i < forecastData.length; i++) {
      const x = xScale(new Date(forecastData[i].date));
      const y = yScale(forecastData[i].predicted_price);
      ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // Upper bound line
    ctx.beginPath();
    ctx.strokeStyle = colors.upperLine;
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    forecastData.forEach((p, i) => {
      const x = xScale(new Date(p.date));
      const y = yScale(p.upper_bound);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Lower bound line
    ctx.beginPath();
    forecastData.forEach((p, i) => {
      const x = xScale(new Date(p.date));
      const y = yScale(p.lower_bound);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Dots on endpoints
  if (historicalData.length > 0) {
    const last = historicalData[historicalData.length - 1];
    ctx.beginPath();
    ctx.arc(xScale(new Date(last.date)), yScale(last.close), 4, 0, Math.PI * 2);
    ctx.fillStyle = colors.histLine;
    ctx.fill();
    ctx.strokeStyle = colors.dotStroke;
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  if (forecastData.length > 0) {
    const last = forecastData[forecastData.length - 1];
    ctx.beginPath();
    ctx.arc(xScale(new Date(last.date)), yScale(last.predicted_price), 4, 0, Math.PI * 2);
    ctx.fillStyle = colors.forecastLine;
    ctx.fill();
    ctx.strokeStyle = colors.dotStroke;
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  // Date labels on x-axis
  ctx.fillStyle = colors.gridText;
  ctx.font = '10px Inter, system-ui, sans-serif';
  ctx.textAlign = 'center';
  const dateStep = Math.max(1, Math.floor(allPoints.length / 8));
  for (let i = 0; i < allPoints.length; i += dateStep) {
    const x = xScale(allPoints[i].date);
    ctx.fillText(formatDate(allPoints[i].date.toISOString()), x, H - PAD.bottom + 20);
  }
  // Always show last date
  const lastPt = allPoints[allPoints.length - 1];
  ctx.fillText(formatDate(lastPt.date.toISOString()), xScale(lastPt.date), H - PAD.bottom + 20);

  // Y-axis label
  ctx.save();
  ctx.translate(12, PAD.top + chartH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = colors.gridText;
  ctx.font = '11px Inter, system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Price', 0, 0);
  ctx.restore();
}

export default function ForecastPage() {
  const [searchParams] = useSearchParams();
  const preselectedPair = searchParams.get('pair') || '';

  const [pairs, setPairs] = useState([]);
  const [selectedPair, setSelectedPair] = useState(preselectedPair);
  const [horizon, setHorizon] = useState(30);
  const [selectedTimeframe, setSelectedTimeframe] = useState('1h');
  const [loading, setLoading] = useState(false);
  const [forecast, setForecast] = useState(null);
  const [error, setError] = useState(null);
  const [pairsLoading, setPairsLoading] = useState(true);

  const canvasRef = useRef(null);

  useEffect(() => {
    if (preselectedPair) {
      setSelectedPair(preselectedPair);
    }
  }, [preselectedPair]);

  useEffect(() => {
    let cancelled = false;
    async function fetchPairs() {
      try {
        const data = await getPairs();
        if (!cancelled) {
          setPairs(data.pairs || []);
          if (!preselectedPair && data.pairs?.length > 0) {
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
  }, [preselectedPair]);

  // Draw chart when forecast data is available
  useEffect(() => {
    if (forecast && canvasRef.current) {
      drawChart(canvasRef.current, forecast);
    }
  }, [forecast]);

  // Redraw chart on resize or theme change
  useEffect(() => {
    function handleResize() {
      if (forecast && canvasRef.current) {
        drawChart(canvasRef.current, forecast);
      }
    }
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [forecast]);

  // Redraw chart when theme changes
  useEffect(() => {
    const observer = new MutationObserver(() => {
      if (forecast && canvasRef.current) {
        drawChart(canvasRef.current, forecast);
      }
    });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
    return () => observer.disconnect();
  }, [forecast]);

  const fetchForecast = useCallback(async () => {
    if (!selectedPair) return;
    setLoading(true);
    setForecast(null);
    setError(null);
    try {
      const data = await getForecast(selectedPair, horizon, selectedTimeframe);
      setForecast(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [selectedPair, horizon, selectedTimeframe]);

  if (pairsLoading) return <Spinner className="py-20" />;

  const directionColor = forecast?.direction === 'UP' ? 'green' : forecast?.direction === 'DOWN' ? 'red' : 'gray';

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">Price Forecast</h1>
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
          Generate future price forecasts using the trained ML model for a selected forex pair.
        </p>
      </div>

      {/* Controls */}
      <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm mb-6 dark:border-gray-700 dark:bg-gray-800">
        <div className="flex flex-col gap-4">
          {/* Pair Selection */}
          <div>
            <label htmlFor="forecast-pair-select" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Select Forex Pair
            </label>
            <select
              id="forecast-pair-select"
              value={selectedPair}
              onChange={(e) => {
                setSelectedPair(e.target.value);
                setForecast(null);
                setError(null);
              }}
              disabled={loading}
              className="block w-full max-w-xs rounded-lg border border-gray-300 bg-white px-4 py-2.5 text-sm text-gray-900 focus:border-primary-500 focus:ring-2 focus:ring-primary-200 focus:outline-none disabled:opacity-50 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
            >
              {pairs.map((pair) => (
                <option key={pair} value={pair}>
                  {formatPair(pair)}
                </option>
              ))}
            </select>
          </div>

          {/* Horizon Selection */}
          <div>
            <label htmlFor="forecast-horizon" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Forecast Horizon
            </label>
            <div className="flex flex-wrap gap-2">
              {HORIZON_OPTIONS.map((opt) => (
                <button
                  key={opt.value}
                  onClick={() => {
                    setHorizon(opt.value);
                    setForecast(null);
                    setError(null);
                  }}
                  disabled={loading}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    horizon === opt.value
                      ? 'bg-primary-600 text-white'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200 disabled:opacity-50 dark:bg-gray-700 dark:text-gray-300 dark:hover:bg-gray-600'
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>

          {/* Time Frame Selection */}
          <div>
            <label htmlFor="forecast-timeframe" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Time Frame
            </label>
            <select
              id="forecast-timeframe"
              value={selectedTimeframe}
              onChange={(e) => {
                setSelectedTimeframe(e.target.value);
                setForecast(null);
                setError(null);
              }}
              disabled={loading}
              className="block w-full max-w-xs rounded-lg border border-gray-300 bg-white px-4 py-2.5 text-sm text-gray-900 focus:border-primary-500 focus:ring-2 focus:ring-primary-200 focus:outline-none disabled:opacity-50 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
            >
              {TIMEFRAME_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>

          {/* Generate Button */}
          <div>
            <button
              onClick={fetchForecast}
              disabled={loading || !selectedPair}
              className="inline-flex items-center justify-center gap-2 rounded-lg bg-primary-600 px-6 py-2.5 text-sm font-medium text-white hover:bg-primary-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <>
                  <div className="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                  Generating Forecast...
                </>
              ) : (
                'Generate Forecast'
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Loading */}
      {loading && (
        <div className="rounded-xl border border-blue-200 bg-blue-50 p-5 mb-6 dark:border-blue-800/50 dark:bg-blue-900/20">
          <div className="flex items-center gap-3">
            <Spinner className="py-0" />
            <p className="text-sm text-blue-700 dark:text-blue-400">
              Generating {horizon}-day forecast for {formatPair(selectedPair)}...
            </p>
          </div>
        </div>
      )}

      {/* Error */}
      {error && !loading && (
        <ErrorMessage
          message={error}
          onRetry={fetchForecast}
        />
      )}

      {/* Forecast Results */}
      {forecast && !loading && (
        <>
          {/* Summary Cards */}
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4 mb-6">
            <div className="rounded-xl bg-gray-50 p-4 dark:bg-gray-700/50">
              <p className="text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">Forecast Direction</p>
              <div className="mt-1 flex items-center gap-2">
                <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium ${
                  directionColor === 'green'
                    ? 'bg-green-50 text-green-700 ring-1 ring-inset ring-green-600/20 dark:bg-green-900/30 dark:text-green-400 dark:ring-green-800/40'
                    : directionColor === 'red'
                      ? 'bg-red-50 text-red-700 ring-1 ring-inset ring-red-600/20 dark:bg-red-900/30 dark:text-red-400 dark:ring-red-800/40'
                      : 'bg-gray-100 text-gray-600 ring-1 ring-inset ring-gray-500/20 dark:bg-gray-600 dark:text-gray-300 dark:ring-gray-500/30'
                }`}>
                  <span className={`h-1.5 w-1.5 rounded-full ${
                    directionColor === 'green' ? 'bg-green-500' : directionColor === 'red' ? 'bg-red-500' : 'bg-gray-400'
                  }`} />
                  {forecast.direction}
                </span>
              </div>
            </div>

            <div className="rounded-xl bg-gray-50 p-4 dark:bg-gray-700/50">
              <p className="text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">Current Price</p>
              <p className="mt-1 text-xl font-bold text-gray-900 dark:text-gray-100 font-mono">
                {forecast.last_historical_price?.toFixed(4)}
              </p>
            </div>

            <div className="rounded-xl bg-gray-50 p-4 dark:bg-gray-700/50">
              <p className="text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">Forecast Period</p>
              <p className="mt-1 text-xl font-bold text-gray-900 dark:text-gray-100">
                {forecast.forecast_horizon ?? horizon} days
              </p>
            </div>

            <div className="rounded-xl bg-gray-50 p-4 dark:bg-gray-700/50">
              <p className="text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">Model Confidence</p>
              <p className="mt-1 text-xl font-bold text-gray-900 dark:text-gray-100">
                {forecast.confidence != null ? `${(forecast.confidence * 100).toFixed(1)}%` : '—'}
              </p>
            </div>
          </div>

          {/* Model Info */}
          <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm mb-6 dark:border-gray-700 dark:bg-gray-800">
            <h2 className="text-sm font-semibold text-gray-900 dark:text-gray-100 mb-3">Model Information</h2>
            <div className="grid gap-3 sm:grid-cols-3 text-sm">
              <div>
                <span className="text-gray-500 dark:text-gray-400">Pair:</span>{' '}
                <span className="font-medium dark:text-gray-100">{forecast.pair ? formatPair(forecast.pair) : '—'}</span>
              </div>
              <div>
                <span className="text-gray-500 dark:text-gray-400">Model:</span>{' '}
                <span className="font-medium font-mono text-xs dark:text-gray-100">{forecast.model_name || 'Unknown'} ({forecast.model_version || '—'})</span>
              </div>
              {forecast.trained_at && (
                <div>
                  <span className="text-gray-500 dark:text-gray-400">Trained At:</span>{' '}
                  <span className="font-medium dark:text-gray-100">{new Date(forecast.trained_at).toLocaleString()}</span>
                </div>
              )}
            </div>
          </div>

          {/* Chart */}
          <div className="rounded-xl border border-gray-200 bg-white shadow-sm mb-6 dark:border-gray-700 dark:bg-gray-800">
            <div className="px-5 py-4 border-b border-gray-200 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 dark:border-gray-700">
              <h2 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
                Price Forecast Chart
              </h2>
              <div className="flex items-center gap-4 text-xs text-gray-500 dark:text-gray-400">
                <span className="flex items-center gap-1.5">
                  <span className="h-0.5 w-4 bg-blue-600 inline-block" />
                  Historical
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="h-0.5 w-4 bg-orange-500 inline-block" style={{ borderTop: '2px dashed #f97316', background: 'none', height: 0 }} />
                  Forecast
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="h-2 w-4 bg-orange-100 rounded inline-block border border-orange-200 dark:bg-orange-900/30 dark:border-orange-800/50" />
                  Confidence Band
                </span>
              </div>
            </div>
            <div className="p-4">
              <canvas
                ref={canvasRef}
                style={{ width: '100%', height: '400px' }}
              />
            </div>
          </div>

          {/* Forecast Table */}
          {forecast.forecast?.length > 0 && (
            <div className="rounded-xl border border-gray-200 bg-white shadow-sm overflow-hidden dark:border-gray-700 dark:bg-gray-800">
              <div className="px-5 py-4 border-b border-gray-200 dark:border-gray-700">
                <h2 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
                  Forecast Details
                  <span className="ml-2 text-xs font-normal text-gray-500 dark:text-gray-400">
                    {forecast.forecast.length} periods
                  </span>
                </h2>
              </div>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                  <thead className="bg-gray-50 dark:bg-gray-700/50">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                        Date
                      </th>
                      <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                        Predicted Price
                      </th>
                      <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                        Upper Bound
                      </th>
                      <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                        Lower Bound
                      </th>
                      <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                        Direction
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                    {forecast.forecast.map((point, idx) => (
                      <tr key={idx} className="hover:bg-gray-50 dark:hover:bg-gray-700/30">
                        <td className="px-4 py-3 text-sm text-gray-900 dark:text-gray-100 whitespace-nowrap">
                          {point.date}
                        </td>
                        <td className="px-4 py-3 text-sm text-right text-gray-900 dark:text-gray-100 font-mono">
                          {point.predicted_price != null ? point.predicted_price.toFixed(4) : '—'}
                        </td>
                        <td className="px-4 py-3 text-sm text-right text-green-700 dark:text-green-400 font-mono">
                          {point.upper_bound != null ? point.upper_bound.toFixed(4) : '—'}
                        </td>
                        <td className="px-4 py-3 text-sm text-right text-red-700 dark:text-red-400 font-mono">
                          {point.lower_bound != null ? point.lower_bound.toFixed(4) : '—'}
                        </td>
                        <td className="px-4 py-3 text-center whitespace-nowrap">
                          <span
                            className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium ${
                              point.direction === 'UP'
                                ? 'bg-green-50 text-green-700 ring-1 ring-inset ring-green-600/20 dark:bg-green-900/30 dark:text-green-400 dark:ring-green-800/40'
                                : point.direction === 'DOWN'
                                  ? 'bg-red-50 text-red-700 ring-1 ring-inset ring-red-600/20 dark:bg-red-900/30 dark:text-red-400 dark:ring-red-800/40'
                                  : 'bg-gray-100 text-gray-600 ring-1 ring-inset ring-gray-500/20 dark:bg-gray-600 dark:text-gray-300 dark:ring-gray-500/30'
                            }`}
                          >
                            <span
                              className={`h-1.5 w-1.5 rounded-full ${
                                point.direction === 'UP' ? 'bg-green-500' : point.direction === 'DOWN' ? 'bg-red-500' : 'bg-gray-400'
                              }`}
                            />
                            {point.direction}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
