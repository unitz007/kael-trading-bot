import { useState, useEffect, useCallback } from 'react';
import TradingViewChart from '../components/TradingViewChart';
import { getPairs, getHistory, getForecast } from '../api';
import { useTheme } from '../components/ThemeProvider';

const SUPPORTED_TIMEFRAMES = ['5m', '15m', '1h', '4h'];

export default function ChartPage() {
  const { theme } = useTheme();
  const [pairs, setPairs] = useState([]);
  const [selectedPair, setSelectedPair] = useState('');
  const [timeframe, setTimeframe] = useState('1h');
  const [historyData, setHistoryData] = useState([]);
  const [forecastData, setForecastData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Load available pairs
  useEffect(() => {
    let cancelled = false;
    async function loadPairs() {
      try {
        const data = await getPairs();
        if (!cancelled && data.pairs) {
          setPairs(data.pairs);
          if (data.pairs.length > 0 && !selectedPair) {
            setSelectedPair(data.pairs[0]);
          }
        }
      } catch {
        // Silently ignore — pairs are not critical
      }
    }
    loadPairs();
    return () => {
      cancelled = true;
    };
  }, [selectedPair]);

  // Load history + forecast when pair or timeframe changes
  const loadChartData = useCallback(async (pair, tf) => {
    if (!pair) return;
    setLoading(true);
    setError(null);
    try {
      // Fetch history data (OHLCV)
      const history = await getHistory(pair);
      if (history.data) {
        setHistoryData(history.data);
      } else {
        setHistoryData([]);
        setError(history.error || 'No history data available for this pair.');
      }

      // Fetch forecast data (only for supported timeframes)
      if (SUPPORTED_TIMEFRAMES.includes(tf)) {
        try {
          const forecast = await getForecast(pair, 30, tf);
          if (forecast.forecast) {
            setForecastData(forecast.forecast);
          } else {
            setForecastData([]);
          }
        } catch {
          // Forecast is optional — don't block on errors
          setForecastData([]);
        }
      } else {
        setForecastData([]);
      }
    } catch (err) {
      setError(err.message || 'Failed to load chart data.');
      setHistoryData([]);
      setForecastData([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadChartData(selectedPair, timeframe);
  }, [selectedPair, timeframe, loadChartData]);

  const handleTimeframeChange = (tf) => {
    setTimeframe(tf);
  };

  return (
    <div className="flex flex-col h-full relative">
      {/* Header */}
      <div className="mb-4">
        <h1 className="text-xl font-bold text-gray-900 dark:text-gray-100">
          Chart
        </h1>
        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
          TradingView-style candlestick chart with overlays
        </p>
      </div>

      {/* Pair selector */}
      <div className="mb-3 flex items-center gap-2">
        <label
          htmlFor="pair-select"
          className="text-sm font-medium text-gray-700 dark:text-gray-300"
        >
          Pair:
        </label>
        <select
          id="pair-select"
          value={selectedPair}
          onChange={(e) => setSelectedPair(e.target.value)}
          className="text-sm border border-gray-300 dark:border-gray-600 rounded-md px-3 py-1.5 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-400"
        >
          <option value="">Select a pair…</option>
          {pairs.map((p) => (
            <option key={p} value={p}>
              {p.replace('=X', '').replace('^', '')}
            </option>
          ))}
        </select>
      </div>

      {/* Chart */}
      <div className="flex-1 min-h-0">
        <TradingViewChart
          pair={selectedPair}
          theme={theme}
          historyData={historyData}
          forecast={forecastData}
          loading={loading}
          error={error}
          onTimeframeChange={handleTimeframeChange}
          activeTimeframe={timeframe}
        />
      </div>
    </div>
  );
}
