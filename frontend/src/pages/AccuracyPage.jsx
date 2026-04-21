import { useState, useEffect, useCallback, useRef } from 'react';
import { getPairs, getAccuracySummary, getAccuracyPredictions, getAccuracyTrend } from '../api';
import Spinner from '../components/Spinner';
import ErrorMessage from '../components/ErrorMessage';

const TIMEFRAME_OPTIONS = [
  { value: '', label: 'All Timeframes' },
  { value: '5m', label: '5 Min' },
  { value: '15m', label: '15 Min' },
  { value: '1h', label: '1 Hour' },
  { value: '4h', label: '4 Hours' },
];

function formatPair(ticker) {
  return ticker.replace('=X', '');
}

function formatPct(value) {
  if (value == null) return '—';
  return `${(value * 100).toFixed(1)}%`;
}

function formatDrift(value) {
  if (value == null) return '—';
  const sign = value >= 0 ? '+' : '';
  return `${sign}${(value * 100).toFixed(3)}%`;
}

function isDarkMode() {
  return document.documentElement.classList.contains('dark');
}

function formatDate(dateStr) {
  const d = new Date(dateStr);
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

function drawComparisonChart(canvas, predictions) {
  if (!canvas || !predictions || predictions.length < 2) return;

  const dark = isDarkMode();
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

  const colors = {
    bg: dark ? '#111827' : '#ffffff',
    gridLine: dark ? '#374151' : '#f3f4f6',
    gridText: dark ? '#9ca3af' : '#9ca3af',
    predictedLine: '#f97316',
    actualLine: '#2563eb',
    dotStroke: dark ? '#111827' : '#ffffff',
  };

  // Filter to predictions that have both target and actual prices
  const evaluated = predictions.filter(
    (p) => p.target_price != null && p.actual_price != null
  );

  if (evaluated.length < 2) {
    ctx.fillStyle = dark ? '#374151' : '#f9fafb';
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = dark ? '#9ca3af' : '#6b7280';
    ctx.font = '13px Inter, system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Need at least 2 evaluated predictions to draw chart', W / 2, H / 2);
    return;
  }

  // Build data points sorted by predicted_at
  const points = evaluated
    .filter((p) => p.predicted_at)
    .sort((a, b) => new Date(a.predicted_at) - new Date(b.predicted_at))
    .map((p) => ({
      date: new Date(p.predicted_at),
      predicted: p.target_price,
      actual: p.actual_price,
    }));

  const minDate = points[0].date.getTime();
  const maxDate = points[points.length - 1].date.getTime();
  const allPrices = points.flatMap((p) => [p.predicted, p.actual]);
  const minPrice = Math.min(...allPrices) * 0.9995;
  const maxPrice = Math.max(...allPrices) * 1.0005;

  const xScale = (date) => PAD.left + ((date.getTime() - minDate) / (maxDate - minDate)) * chartW;
  const yScale = (price) => PAD.top + chartH - ((price - minPrice) / (maxPrice - minPrice)) * chartH;

  // Clear
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = colors.bg;
  ctx.fillRect(0, 0, W, H);

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

    const price = maxPrice - ((maxPrice - minPrice) / numGridLines) * i;
    ctx.fillStyle = colors.gridText;
    ctx.font = '10px Inter, system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(price.toFixed(4), W - PAD.right + 8, y + 4);
  }

  // Predicted price line (dashed orange)
  ctx.beginPath();
  ctx.strokeStyle = colors.predictedLine;
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
  ctx.strokeStyle = colors.actualLine;
  ctx.lineWidth = 2;
  ctx.lineJoin = 'round';
  points.forEach((p, i) => {
    const x = xScale(p.date);
    const y = yScale(p.actual);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Dots at last points
  const lastPoint = points[points.length - 1];

  ctx.beginPath();
  ctx.arc(xScale(lastPoint.date), yScale(lastPoint.predicted), 4, 0, Math.PI * 2);
  ctx.fillStyle = colors.predictedLine;
  ctx.fill();
  ctx.strokeStyle = colors.dotStroke;
  ctx.lineWidth = 2;
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(xScale(lastPoint.date), yScale(lastPoint.actual), 4, 0, Math.PI * 2);
  ctx.fillStyle = colors.actualLine;
  ctx.fill();
  ctx.strokeStyle = colors.dotStroke;
  ctx.lineWidth = 2;
  ctx.stroke();

  // Date labels
  ctx.fillStyle = colors.gridText;
  ctx.font = '10px Inter, system-ui, sans-serif';
  ctx.textAlign = 'center';
  const dateStep = Math.max(1, Math.floor(points.length / 8));
  for (let i = 0; i < points.length; i += dateStep) {
    const x = xScale(points[i].date);
    ctx.fillText(formatDate(points[i].date.toISOString()), x, H - PAD.bottom + 20);
  }
  ctx.fillText(formatDate(lastPoint.date.toISOString()), xScale(lastPoint.date), H - PAD.bottom + 20);

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

function drawTrendChart(canvas, trendData) {
  if (!canvas || !trendData || trendData.length < 2) return;

  const dark = isDarkMode();
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

  const points = trendData.filter((d) => d.win_rate != null);

  if (points.length < 2) {
    ctx.fillStyle = dark ? '#374151' : '#f9fafb';
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = dark ? '#9ca3af' : '#6b7280';
    ctx.font = '13px Inter, system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Not enough trend data to display', W / 2, H / 2);
    return;
  }

  const minRate = 0;
  const maxRate = 1;

  const xScale = (i) => PAD.left + (i / (points.length - 1)) * chartW;
  const yScale = (rate) => PAD.top + chartH - ((rate - minRate) / (maxRate - minRate)) * chartH;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = dark ? '#111827' : '#ffffff';
  ctx.fillRect(0, 0, W, H);

  // Grid lines
  ctx.strokeStyle = dark ? '#374151' : '#f3f4f6';
  ctx.lineWidth = 1;
  const numGridLines = 4;
  for (let i = 0; i <= numGridLines; i++) {
    const y = PAD.top + (chartH / numGridLines) * i;
    ctx.beginPath();
    ctx.moveTo(PAD.left, y);
    ctx.lineTo(W - PAD.right, y);
    ctx.stroke();

    const rate = maxRate - ((maxRate - minRate) / numGridLines) * i;
    ctx.fillStyle = dark ? '#9ca3af' : '#9ca3af';
    ctx.font = '10px Inter, system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`${(rate * 100).toFixed(0)}%`, W - PAD.right + 8, y + 4);
  }

  // 50% reference line
  const y50 = yScale(0.5);
  ctx.beginPath();
  ctx.setLineDash([4, 4]);
  ctx.strokeStyle = dark ? '#6b7280' : '#d1d5db';
  ctx.lineWidth = 1;
  ctx.moveTo(PAD.left, y50);
  ctx.lineTo(W - PAD.right, y50);
  ctx.stroke();
  ctx.setLineDash([]);

  // Area fill
  ctx.beginPath();
  points.forEach((p, i) => {
    const x = xScale(i);
    const y = yScale(p.win_rate);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.lineTo(xScale(points.length - 1), PAD.top + chartH);
  ctx.lineTo(PAD.left, PAD.top + chartH);
  ctx.closePath();
  ctx.fillStyle = dark ? 'rgba(16, 185, 129, 0.1)' : 'rgba(16, 185, 129, 0.08)';
  ctx.fill();

  // Line
  ctx.beginPath();
  ctx.strokeStyle = '#10b981';
  ctx.lineWidth = 2;
  ctx.lineJoin = 'round';
  points.forEach((p, i) => {
    const x = xScale(i);
    const y = yScale(p.win_rate);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Dots
  points.forEach((p, i) => {
    ctx.beginPath();
    ctx.arc(xScale(i), yScale(p.win_rate), 3, 0, Math.PI * 2);
    ctx.fillStyle = '#10b981';
    ctx.fill();
    ctx.strokeStyle = dark ? '#111827' : '#ffffff';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  });

  // Period labels on x-axis
  ctx.fillStyle = dark ? '#9ca3af' : '#9ca3af';
  ctx.font = '10px Inter, system-ui, sans-serif';
  ctx.textAlign = 'center';
  const labelStep = Math.max(1, Math.floor(points.length / 8));
  for (let i = 0; i < points.length; i += labelStep) {
    ctx.fillText(points[i].period, xScale(i), H - PAD.bottom + 20);
  }
  ctx.fillText(points[points.length - 1].period, xScale(points.length - 1), H - PAD.bottom + 20);

  // Y-axis label
  ctx.save();
  ctx.translate(12, PAD.top + chartH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = dark ? '#9ca3af' : '#9ca3af';
  ctx.font = '11px Inter, system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Win Rate', 0, 0);
  ctx.restore();
}

export default function AccuracyPage() {
  const [pairs, setPairs] = useState([]);
  const [selectedPair, setSelectedPair] = useState('');
  const [selectedTimeframe, setSelectedTimeframe] = useState('');
  const [loading, setLoading] = useState(false);
  const [pairsLoading, setPairsLoading] = useState(true);
  const [summary, setSummary] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [trend, setTrend] = useState(null);
  const [error, setError] = useState(null);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(0);

  const comparisonCanvasRef = useRef(null);
  const trendCanvasRef = useRef(null);

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

  const fetchData = useCallback(async (p, tf, pg) => {
    setLoading(true);
    setSummary(null);
    setPredictions(null);
    setTrend(null);
    setError(null);
    setPage(pg);

    try {
      const [summaryData, predData, trendData] = await Promise.all([
        getAccuracySummary(p || undefined, tf || undefined),
        getAccuracyPredictions(p || undefined, tf || undefined, undefined, pg, 50),
        getAccuracyTrend(p || undefined, tf || undefined, 'week'),
      ]);
      setSummary(summaryData);
      setPredictions(predData);
      setTrend(trendData);
      setTotalPages(predData.total_pages || 0);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleSearch = useCallback(() => {
    fetchData(selectedPair, selectedTimeframe, 1);
  }, [selectedPair, selectedTimeframe, fetchData]);

  // Draw comparison chart
  useEffect(() => {
    if (predictions?.predictions && comparisonCanvasRef.current) {
      drawComparisonChart(comparisonCanvasRef.current, predictions.predictions);
    }
  }, [predictions]);

  // Draw trend chart
  useEffect(() => {
    if (trend?.trend && trendCanvasRef.current) {
      drawTrendChart(trendCanvasRef.current, trend.trend);
    }
  }, [trend]);

  // Redraw charts on resize and theme change
  useEffect(() => {
    function handleResize() {
      if (predictions?.predictions && comparisonCanvasRef.current) {
        drawComparisonChart(comparisonCanvasRef.current, predictions.predictions);
      }
      if (trend?.trend && trendCanvasRef.current) {
        drawTrendChart(trendCanvasRef.current, trend.trend);
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
  }, [predictions, trend]);

  if (pairsLoading) return <Spinner className="py-20" />;

  const hasData = summary && summary.total_predictions > 0;

  return (
    <div>
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">Prediction Accuracy</h1>
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
          View how well the model's predictions match actual market prices.
        </p>
      </div>

      {/* Filter Controls */}
      <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-5 shadow-sm mb-6">
        <div className="flex flex-col sm:flex-row gap-3 items-end">
          <div className="flex-1">
            <label htmlFor="acc-pair-select" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Forex Pair
            </label>
            <select
              id="acc-pair-select"
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

          <div className="flex-1">
            <label htmlFor="acc-timeframe-select" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Timeframe
            </label>
            <select
              id="acc-timeframe-select"
              value={selectedTimeframe}
              onChange={(e) => setSelectedTimeframe(e.target.value)}
              disabled={loading}
              className="block w-full max-w-xs rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 px-4 py-2.5 text-sm focus:border-primary-500 focus:ring-2 focus:ring-primary-200 dark:focus:ring-primary-800 focus:outline-none disabled:opacity-50"
            >
              {TIMEFRAME_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>

          <button
            onClick={handleSearch}
            disabled={loading}
            className="inline-flex items-center justify-center gap-2 rounded-lg bg-primary-600 dark:bg-primary-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-primary-700 dark:hover:bg-primary-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 dark:focus:ring-offset-gray-800"
          >
            {loading ? (
              <>
                <div className="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                Loading...
              </>
            ) : (
              'View Accuracy'
            )}
          </button>
        </div>
      </div>

      {/* Loading banner */}
      {loading && (
        <div className="rounded-xl border border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-900/20 p-5 mb-6">
          <div className="flex items-center gap-3">
            <Spinner className="py-0" />
            <p className="text-sm text-blue-700 dark:text-blue-200">
              Fetching accuracy data{selectedPair ? ` for ${formatPair(selectedPair)}` : ''}...
            </p>
          </div>
        </div>
      )}

      {/* Error */}
      {error && !loading && <ErrorMessage message={error} onRetry={handleSearch} />}

      {/* Results */}
      {!loading && hasData && (
        <>
          {/* Summary Metric Cards */}
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4 mb-6">
            <MetricCard
              label="Win Rate"
              value={formatPct(summary.win_rate)}
              subtext={`${summary.correct_count} of ${summary.evaluated_predictions} evaluated`}
              color={summary.win_rate != null && summary.win_rate >= 0.5 ? 'green' : 'red'}
            />
            <MetricCard
              label="Dir. Correctness"
              value={formatPct(summary.win_rate)}
              subtext="Directional accuracy"
              color={summary.win_rate != null && summary.win_rate >= 0.5 ? 'green' : 'red'}
            />
            <MetricCard
              label="Avg. % Drift"
              value={formatDrift(summary.avg_percentage_drift)}
              subtext="Avg price deviation"
              color="blue"
            />
            <MetricCard
              label="Predictions"
              value={summary.total_predictions?.toLocaleString()}
              subtext={`${summary.pending_count} pending`}
              color="gray"
            />
          </div>

          {/* Best & Worst Predictions */}
          <div className="grid gap-4 sm:grid-cols-2 mb-6">
            {summary.best_prediction && (
              <div className="rounded-xl border border-green-200 dark:border-green-800 bg-green-50 dark:bg-green-900/20 p-4">
                <p className="text-xs font-medium uppercase tracking-wider text-green-700 dark:text-green-400 mb-2">Best Prediction</p>
                <div className="text-sm text-green-900 dark:text-green-100 space-y-1">
                  <p>
                    <span className="text-green-600 dark:text-green-400">Pair:</span>{' '}
                    {formatPair(summary.best_prediction.pair)}
                  </p>
                  <p>
                    <span className="text-green-600 dark:text-green-400">Drift:</span>{' '}
                    {formatDrift(summary.best_prediction.percentage_drift)}
                  </p>
                  <p>
                    <span className="text-green-600 dark:text-green-400">Direction:</span>{' '}
                    {summary.best_prediction.predicted_direction}
                    {' → '}
                    <span className={
                      summary.best_prediction.status === 'correct'
                        ? 'text-green-700 dark:text-green-300'
                        : 'text-red-700 dark:text-red-300'
                    }>
                      {summary.best_prediction.status}
                    </span>
                  </p>
                </div>
              </div>
            )}
            {summary.worst_prediction && (
              <div className="rounded-xl border border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-900/20 p-4">
                <p className="text-xs font-medium uppercase tracking-wider text-red-700 dark:text-red-400 mb-2">Worst Prediction</p>
                <div className="text-sm text-red-900 dark:text-red-100 space-y-1">
                  <p>
                    <span className="text-red-600 dark:text-red-400">Pair:</span>{' '}
                    {formatPair(summary.worst_prediction.pair)}
                  </p>
                  <p>
                    <span className="text-red-600 dark:text-red-400">Drift:</span>{' '}
                    {formatDrift(summary.worst_prediction.percentage_drift)}
                  </p>
                  <p>
                    <span className="text-red-600 dark:text-red-400">Direction:</span>{' '}
                    {summary.worst_prediction.predicted_direction}
                    {' → '}
                    <span className={
                      summary.worst_prediction.status === 'correct'
                        ? 'text-green-700 dark:text-green-300'
                        : 'text-red-700 dark:text-red-300'
                    }>
                      {summary.worst_prediction.status}
                    </span>
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Predicted vs Actual Price Chart */}
          <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 shadow-sm mb-6">
            <div className="px-5 py-4 border-b border-gray-200 dark:border-gray-700 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
              <h2 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
                Predicted vs Actual Prices
              </h2>
              <div className="flex items-center gap-4 text-xs text-gray-500 dark:text-gray-400">
                <span className="flex items-center gap-1.5">
                  <span className="h-0.5 w-4 bg-orange-500 inline-block" style={{ borderTop: '2px dashed #f97316', background: 'none', height: 0 }} />
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
                ref={comparisonCanvasRef}
                style={{ width: '100%', height: '400px' }}
                aria-label="Chart comparing predicted prices against actual market prices"
                role="img"
              />
            </div>
          </div>

          {/* Win Rate Trend Chart */}
          <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 shadow-sm mb-6">
            <div className="px-5 py-4 border-b border-gray-200 dark:border-gray-700">
              <h2 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
                Win Rate Trend
                <span className="ml-2 text-xs font-normal text-gray-500 dark:text-gray-400">
                  Weekly
                </span>
              </h2>
            </div>
            <div className="p-4">
              <canvas
                ref={trendCanvasRef}
                style={{ width: '100%', height: '300px' }}
                aria-label="Chart showing win rate trend over time"
                role="img"
              />
            </div>
          </div>

          {/* Predictions Table */}
          {predictions?.predictions?.length > 0 && (
            <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 shadow-sm overflow-hidden mb-6">
              <div className="px-5 py-4 border-b border-gray-200 dark:border-gray-700 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
                <h2 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
                  Prediction Records
                  <span className="ml-2 text-xs font-normal text-gray-500 dark:text-gray-400">
                    {predictions.total?.toLocaleString()} total
                  </span>
                </h2>
                {totalPages > 1 && (
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => fetchData(selectedPair, selectedTimeframe, page - 1)}
                      disabled={page <= 1}
                      className="px-3 py-1.5 rounded-lg text-xs font-medium border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      Previous
                    </button>
                    <span className="text-xs text-gray-500 dark:text-gray-400">
                      Page {page} of {totalPages}
                    </span>
                    <button
                      onClick={() => fetchData(selectedPair, selectedTimeframe, page + 1)}
                      disabled={page >= totalPages}
                      className="px-3 py-1.5 rounded-lg text-xs font-medium border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      Next
                    </button>
                  </div>
                )}
              </div>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                  <thead className="bg-gray-50 dark:bg-gray-700/50">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                        Date
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                        Pair
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                        Direction
                      </th>
                      <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                        Predicted
                      </th>
                      <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                        Actual
                      </th>
                      <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                        Drift
                      </th>
                      <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                        Status
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                    {predictions.predictions.map((pred) => (
                      <tr key={pred.id} className="hover:bg-gray-50 dark:hover:bg-gray-700/50">
                        <td className="px-4 py-3 text-sm text-gray-900 dark:text-gray-100 whitespace-nowrap">
                          {pred.predicted_at
                            ? new Date(pred.predicted_at).toLocaleString()
                            : '—'}
                        </td>
                        <td className="px-4 py-3 text-sm text-gray-900 dark:text-gray-100 whitespace-nowrap">
                          {formatPair(pred.pair)}
                        </td>
                        <td className="px-4 py-3 whitespace-nowrap">
                          <span
                            className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium ${
                              pred.predicted_direction === 'UP'
                                ? 'bg-green-50 dark:bg-green-900/30 text-green-700 dark:text-green-400 ring-1 ring-inset ring-green-600/20 dark:ring-green-500/30'
                                : pred.predicted_direction === 'DOWN'
                                  ? 'bg-red-50 dark:bg-red-900/30 text-red-700 dark:text-red-400 ring-1 ring-inset ring-red-600/20 dark:ring-red-500/30'
                                  : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 ring-1 ring-inset ring-gray-500/20 dark:ring-gray-500/30'
                            }`}
                          >
                            <span
                              className={`h-1.5 w-1.5 rounded-full ${
                                pred.predicted_direction === 'UP'
                                  ? 'bg-green-500 dark:bg-green-400'
                                  : pred.predicted_direction === 'DOWN'
                                    ? 'bg-red-500 dark:bg-red-400'
                                    : 'bg-gray-400'
                              }`}
                            />
                            {pred.predicted_direction}
                          </span>
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
        </>
      )}

      {/* Empty state */}
      {!loading && summary && !hasData && (
        <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-12 shadow-sm text-center">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="mx-auto h-12 w-12 text-gray-400 dark:text-gray-500"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={1.5}
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 0 1 3 19.875v-6.75ZM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125V8.625ZM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125V4.125Z"
            />
          </svg>
          <h3 className="mt-4 text-sm font-semibold text-gray-900 dark:text-gray-100">No Accuracy Data</h3>
          <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
            No prediction accuracy data is available for the selected filters. 
            Accuracy is tracked once predictions have been evaluated against actual market prices.
          </p>
          <button
            onClick={handleSearch}
            className="mt-4 inline-flex items-center gap-2 rounded-lg bg-primary-600 dark:bg-primary-600 px-4 py-2 text-sm font-medium text-white hover:bg-primary-700 dark:hover:bg-primary-700 transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 dark:focus:ring-offset-gray-800"
          >
            Try Different Filters
          </button>
        </div>
      )}
    </div>
  );
}

function MetricCard({ label, value, subtext, color = 'blue' }) {
  const colorClasses = {
    blue: 'bg-primary-50 dark:bg-primary-900/30 text-primary-700 dark:text-primary-400',
    green: 'bg-green-50 dark:bg-green-900/30 text-green-700 dark:text-green-400',
    red: 'bg-red-50 dark:bg-red-900/30 text-red-700 dark:text-red-400',
    gray: 'bg-gray-50 dark:bg-gray-800 text-gray-700 dark:text-gray-400',
  };

  return (
    <div className={`rounded-xl p-4 ${colorClasses[color] || colorClasses.blue}`}>
      <p className="text-xs font-medium uppercase tracking-wider opacity-90">{label}</p>
      <p className="mt-1 text-2xl font-bold">{value}</p>
      {subtext && (
        <p className="mt-0.5 text-xs opacity-75">{subtext}</p>
      )}
    </div>
  );
}
