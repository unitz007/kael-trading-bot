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

function formatPair(ticker) {
  return ticker.replace('=X', '');
}

function formatDate(dateStr) {
  const d = new Date(dateStr);
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

function drawChart(canvas, data) {
  if (!canvas || !data) return;

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

  // Clear
  ctx.clearRect(0, 0, W, H);

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

  // Background
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, W, H);

  // Forecast background shading
  const boundaryIdx = historicalData.length;
  if (boundaryIdx > 0 && boundaryIdx < allPoints.length) {
    const boundaryX = xScale(allPoints[boundaryIdx].date);

    // Historical region - subtle blue tint
    ctx.fillStyle = 'rgba(239, 246, 255, 0.4)';
    ctx.fillRect(PAD.left, PAD.top, boundaryX - PAD.left, chartH);

    // Forecast region - subtle orange tint
    ctx.fillStyle = 'rgba(255, 247, 237, 0.5)';
    ctx.fillRect(boundaryX, PAD.top, W - PAD.right - boundaryX, chartH);

    // Vertical boundary line
    ctx.beginPath();
    ctx.setLineDash([6, 4]);
    ctx.strokeStyle = '#9ca3af';
    ctx.lineWidth = 1.5;
    ctx.moveTo(boundaryX, PAD.top);
    ctx.lineTo(boundaryX, PAD.top + chartH);
    ctx.stroke();
    ctx.setLineDash([]);

    // Boundary label
    ctx.fillStyle = '#6b7280';
    ctx.font = '11px Inter, system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Forecast →', boundaryX + (W - PAD.right - boundaryX) / 2, PAD.top - 10);
    ctx.fillText('← Historical', PAD.left + (boundaryX - PAD.left) / 2, PAD.top - 10);
  }

  // Grid lines
  ctx.strokeStyle = '#f3f4f6';
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
    ctx.fillStyle = '#9ca3af';
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
    ctx.fillStyle = 'rgba(251, 146, 60, 0.12)';
    ctx.fill();
  }

  // Historical price line
  if (historicalData.length > 1) {
    ctx.beginPath();
    ctx.strokeStyle = '#2563eb';
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
    ctx.strokeStyle = '#f97316';
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
    ctx.strokeStyle = 'rgba(251, 146, 60, 0.3)';
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
    ctx.fillStyle = '#2563eb';
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  if (forecastData.length > 0) {
    const last = forecastData[forecastData.length - 1];
    ctx.beginPath();
    ctx.arc(xScale(new Date(last.date)), yScale(last.predicted_price), 4, 0, Math.PI * 2);
    ctx.fillStyle = '#f97316';
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  // Date labels on x-axis
  ctx.fillStyle = '#9ca3af';
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
  ctx.fillStyle = '#9ca3af';
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

  // Redraw chart on resize
  useEffect(() => {
    function handleResize() {
      if (forecast && canvasRef.current) {
        drawChart(canvasRef.current, forecast);
      }
    }
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [forecast]);

  const fetchForecast = useCallback(async () => {
    if (!selectedPair) return;
    setLoading(true);
    setForecast(null);
    setError(null);
    try {
      const data = await getForecast(selectedPair, horizon);
      setForecast(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [selectedPair, horizon]);

  if (pairsLoading) return <Spinner className="py-20" />;

  const directionColor = forecast?.direction === 'UP' ? 'green' : forecast?.direction === 'DOWN' ? 'red' : 'gray';

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Price Forecast</h1>
        <p className="mt-1 text-sm text-gray-500">
          Generate future price forecasts using the trained ML model for a selected forex pair.
        </p>
      </div>

      {/* Controls */}
      <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm mb-6">
        <div className="flex flex-col gap-4">
          {/* Pair Selection */}
          <div>
            <label htmlFor="forecast-pair-select" className="block text-sm font-medium text-gray-700 mb-2">
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
              className="block w-full max-w-xs rounded-lg border border-gray-300 bg-white px-4 py-2.5 text-sm text-gray-900 focus:border-primary-500 focus:ring-2 focus:ring-primary-200 focus:outline-none disabled:opacity-50"
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
            <label htmlFor="forecast-horizon" className="block text-sm font-medium text-gray-700 mb-2">
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
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200 disabled:opacity-50'
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
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
        <div className="rounded-xl border border-blue-200 bg-blue-50 p-5 mb-6">
          <div className="flex items-center gap-3">
            <Spinner className="py-0" />
            <p className="text-sm text-blue-700">
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
            <div className="rounded-xl bg-gray-50 p-4">
              <p className="text-xs font-medium uppercase tracking-wider text-gray-500">Forecast Direction</p>
              <div className="mt-1 flex items-center gap-2">
                <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium ${
                  directionColor === 'green'
                    ? 'bg-green-50 text-green-700 ring-1 ring-inset ring-green-600/20'
                    : directionColor === 'red'
                      ? 'bg-red-50 text-red-700 ring-1 ring-inset ring-red-600/20'
                      : 'bg-gray-100 text-gray-600 ring-1 ring-inset ring-gray-500/20'
                }`}>
                  <span className={`h-1.5 w-1.5 rounded-full ${
                    directionColor === 'green' ? 'bg-green-500' : directionColor === 'red' ? 'bg-red-500' : 'bg-gray-400'
                  }`} />
                  {forecast.direction}
                </span>
              </div>
            </div>

            <div className="rounded-xl bg-gray-50 p-4">
              <p className="text-xs font-medium uppercase tracking-wider text-gray-500">Current Price</p>
              <p className="mt-1 text-xl font-bold text-gray-900 font-mono">
                {forecast.last_historical_price?.toFixed(4)}
              </p>
            </div>

            <div className="rounded-xl bg-gray-50 p-4">
              <p className="text-xs font-medium uppercase tracking-wider text-gray-500">Forecast Period</p>
              <p className="mt-1 text-xl font-bold text-gray-900">
                {forecast.forecast_horizon ?? horizon} days
              </p>
            </div>

            <div className="rounded-xl bg-gray-50 p-4">
              <p className="text-xs font-medium uppercase tracking-wider text-gray-500">Model Confidence</p>
              <p className="mt-1 text-xl font-bold text-gray-900">
                {forecast.confidence != null ? `${(forecast.confidence * 100).toFixed(1)}%` : '—'}
              </p>
            </div>
          </div>

          {/* Model Info */}
          <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm mb-6">
            <h2 className="text-sm font-semibold text-gray-900 mb-3">Model Information</h2>
            <div className="grid gap-3 sm:grid-cols-3 text-sm">
              <div>
                <span className="text-gray-500">Pair:</span>{' '}
                <span className="font-medium">{forecast.pair ? formatPair(forecast.pair) : '—'}</span>
              </div>
              <div>
                <span className="text-gray-500">Model:</span>{' '}
                <span className="font-medium font-mono text-xs">{forecast.model_name || 'Unknown'} ({forecast.model_version || '—'})</span>
              </div>
              {forecast.trained_at && (
                <div>
                  <span className="text-gray-500">Trained At:</span>{' '}
                  <span className="font-medium">{new Date(forecast.trained_at).toLocaleString()}</span>
                </div>
              )}
            </div>
          </div>

          {/* Chart */}
          <div className="rounded-xl border border-gray-200 bg-white shadow-sm mb-6">
            <div className="px-5 py-4 border-b border-gray-200 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
              <h2 className="text-sm font-semibold text-gray-900">
                Price Forecast Chart
              </h2>
              <div className="flex items-center gap-4 text-xs text-gray-500">
                <span className="flex items-center gap-1.5">
                  <span className="h-0.5 w-4 bg-blue-600 inline-block" />
                  Historical
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="h-0.5 w-4 bg-orange-500 inline-block" style={{ borderTop: '2px dashed #f97316', background: 'none', height: 0 }} />
                  Forecast
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="h-2 w-4 bg-orange-100 rounded inline-block border border-orange-200" />
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
            <div className="rounded-xl border border-gray-200 bg-white shadow-sm overflow-hidden">
              <div className="px-5 py-4 border-b border-gray-200">
                <h2 className="text-sm font-semibold text-gray-900">
                  Forecast Details
                  <span className="ml-2 text-xs font-normal text-gray-500">
                    {forecast.forecast.length} periods
                  </span>
                </h2>
              </div>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Date
                      </th>
                      <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Predicted Price
                      </th>
                      <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Upper Bound
                      </th>
                      <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Lower Bound
                      </th>
                      <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Direction
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200">
                    {forecast.forecast.map((point, idx) => (
                      <tr key={idx} className="hover:bg-gray-50">
                        <td className="px-4 py-3 text-sm text-gray-900 whitespace-nowrap">
                          {point.date}
                        </td>
                        <td className="px-4 py-3 text-sm text-right text-gray-900 font-mono">
                          {point.predicted_price != null ? point.predicted_price.toFixed(4) : '—'}
                        </td>
                        <td className="px-4 py-3 text-sm text-right text-green-700 font-mono">
                          {point.upper_bound != null ? point.upper_bound.toFixed(4) : '—'}
                        </td>
                        <td className="px-4 py-3 text-sm text-right text-red-700 font-mono">
                          {point.lower_bound != null ? point.lower_bound.toFixed(4) : '—'}
                        </td>
                        <td className="px-4 py-3 text-center whitespace-nowrap">
                          <span
                            className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium ${
                              point.direction === 'UP'
                                ? 'bg-green-50 text-green-700 ring-1 ring-inset ring-green-600/20'
                                : point.direction === 'DOWN'
                                  ? 'bg-red-50 text-red-700 ring-1 ring-inset ring-red-600/20'
                                  : 'bg-gray-100 text-gray-600 ring-1 ring-inset ring-gray-500/20'
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