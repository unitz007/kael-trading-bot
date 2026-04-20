import { useState, useEffect, useCallback, useRef } from 'react';
import { useSearchParams } from 'react-router-dom';
import { getPairs, getTradeSetup } from '../api';
import Spinner from '../components/Spinner';
import ErrorMessage from '../components/ErrorMessage';

const TIMEFRAME_OPTIONS = [
  { value: '5m', label: '5 Minutes' },
  { value: '15m', label: '15 Minutes' },
  { value: '1h', label: '1 Hour' },
  { value: '4h', label: '4 Hours' },
];

function formatPair(ticker) {
  return ticker.replace('=X', '');
}

function formatPrice(price) {
  return Number(price).toFixed(5);
}

function ConfidenceBar({ value }) {
  const pct = Math.round(value * 100);
  const color =
    pct >= 70 ? 'bg-green-500' : pct >= 50 ? 'bg-yellow-500' : 'bg-red-500';

  return (
    <div className="mt-1.5">
      <div className="h-2 w-full rounded-full bg-gray-200 overflow-hidden dark:bg-gray-700">
        <div
          className={`h-full rounded-full transition-all duration-500 ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">{pct}% confidence</p>
    </div>
  );
}

export default function TradeSetupPage() {
  const [pairs, setPairs] = useState([]);
  const [selectedPair, setSelectedPair] = useState('');
  const [selectedTimeframe, setSelectedTimeframe] = useState('1h');
  const [loading, setLoading] = useState(false);
  const [setup, setSetup] = useState(null);
  const [error, setError] = useState(null);
  const [pairsLoading, setPairsLoading] = useState(true);

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

  const fetchSetup = useCallback(async () => {
    if (!selectedPair) return;
    setLoading(true);
    setSetup(null);
    setError(null);
    try {
      const data = await getTradeSetup(selectedPair, selectedTimeframe);
      setSetup(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [selectedPair, selectedTimeframe]);

  if (pairsLoading) return <Spinner className="py-20" />;

  return (
    <div>
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">Trade Setup</h1>
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
          Generate actionable trade setups with entry, take profit, stop loss, and model confidence.
        </p>
      </div>

      {/* Pair Selection */}
      <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm mb-6 dark:border-gray-700 dark:bg-gray-800">
        <label htmlFor="setup-pair-select" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Select Forex Pair
        </label>
        <div className="flex flex-col gap-4">
          <div className="flex flex-col sm:flex-row gap-3">
            <select
              id="setup-pair-select"
              value={selectedPair}
              onChange={(e) => {
                setSelectedPair(e.target.value);
                setSetup(null);
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

            <select
              id="setup-timeframe-select"
              value={selectedTimeframe}
              onChange={(e) => {
                setSelectedTimeframe(e.target.value);
                setSetup(null);
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

            <button
              onClick={fetchSetup}
              disabled={loading || !selectedPair}
              className="inline-flex items-center justify-center gap-2 rounded-lg bg-primary-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-primary-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <>
                  <div className="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                  Generating...
                </>
              ) : (
                'Generate Setup'
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
              Generating trade setup for {formatPair(selectedPair)}...
            </p>
          </div>
        </div>
      )}

      {/* Error */}
      {error && !loading && <ErrorMessage message={error} onRetry={fetchSetup} />}

      {/* Trade Setup Results */}
      {setup && (
        <>
          {/* Direction Banner */}
          <div
            className={`rounded-xl p-5 mb-6 ${
              setup.direction === 'buy'
                ? 'bg-green-50 border border-green-200 dark:bg-green-900/20 dark:border-green-800/50'
                : 'bg-red-50 border border-red-200 dark:bg-red-900/20 dark:border-red-800/50'
            }`}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div
                  className={`flex h-12 w-12 items-center justify-center rounded-full ${
                    setup.direction === 'buy'
                      ? 'bg-green-100 text-green-600 dark:bg-green-800/50 dark:text-green-400'
                      : 'bg-red-100 text-red-600 dark:bg-red-800/50 dark:text-red-400'
                  }`}
                >
                  {setup.direction === 'buy' ? (
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 19.5l15-15m0 0H8.25m11.25 0v11.25" />
                    </svg>
                  ) : (
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 4.5l15 15m0 0V8.25m0 11.25H8.25" />
                    </svg>
                  )}
                </div>
                <div>
                  <h2 className="text-lg font-bold text-gray-900 dark:text-gray-100">
                    {setup.direction.toUpperCase()} — {formatPair(setup.pair)}
                  </h2>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Model-generated trade setup
                  </p>
                </div>
              </div>
              <div className="text-right">
                <p className="text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">Confidence</p>
                <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                  {(setup.confidence * 100).toFixed(1)}%
                </p>
              </div>
            </div>
            <ConfidenceBar value={setup.confidence} />
          </div>

          {/* Price Levels Grid */}
          <div className="grid gap-4 sm:grid-cols-3 mb-6">
            {/* Entry */}
            <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm dark:border-gray-700 dark:bg-gray-800">
              <div className="flex items-center gap-2 mb-2">
                <div className="h-8 w-8 rounded-lg bg-blue-100 flex items-center justify-center dark:bg-blue-900/30">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-blue-600 dark:text-blue-400" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M15 10.5a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" />
                    <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 10.5c0 7.142-7.5 11.25-7.5 11.25S4.5 17.642 4.5 10.5a7.5 7.5 0 1 1 15 0Z" />
                  </svg>
                </div>
                <span className="text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">Entry Price</span>
              </div>
              <p className="text-2xl font-bold font-mono text-gray-900 dark:text-gray-100">
                {formatPrice(setup.entry_price)}
              </p>
            </div>

            {/* Take Profit */}
            <div className="rounded-xl border border-green-200 bg-green-50 p-5 shadow-sm dark:border-green-800/50 dark:bg-green-900/20">
              <div className="flex items-center gap-2 mb-2">
                <div className="h-8 w-8 rounded-lg bg-green-100 flex items-center justify-center dark:bg-green-800/50">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-green-600 dark:text-green-400" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 18 9 11.25l4.306 4.306a11.95 11.95 0 0 1 5.814-5.518l2.74-1.22m0 0-5.94-2.281m5.94 2.28-2.28 5.941" />
                  </svg>
                </div>
                <span className="text-xs font-medium uppercase tracking-wider text-green-700 dark:text-green-400">Take Profit</span>
              </div>
              <p className="text-2xl font-bold font-mono text-green-700 dark:text-green-400">
                {formatPrice(setup.take_profit)}
              </p>
              <p className="text-xs text-green-600 dark:text-green-500 mt-1">
                {setup.direction === 'buy' ? '+' : ''}
                {formatPrice(Math.abs(setup.take_profit - setup.entry_price))} from entry
              </p>
            </div>

            {/* Stop Loss */}
            <div className="rounded-xl border border-red-200 bg-red-50 p-5 shadow-sm dark:border-red-800/50 dark:bg-red-900/20">
              <div className="flex items-center gap-2 mb-2">
                <div className="h-8 w-8 rounded-lg bg-red-100 flex items-center justify-center dark:bg-red-800/50">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-red-600 dark:text-red-400" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z" />
                  </svg>
                </div>
                <span className="text-xs font-medium uppercase tracking-wider text-red-700 dark:text-red-400">Stop Loss</span>
              </div>
              <p className="text-2xl font-bold font-mono text-red-700 dark:text-red-400">
                {formatPrice(setup.stop_loss)}
              </p>
              <p className="text-xs text-red-600 dark:text-red-500 mt-1">
                {setup.direction === 'sell' ? '+' : '-'}
                {formatPrice(Math.abs(setup.stop_loss - setup.entry_price))} from entry
              </p>
            </div>
          </div>

          {/* Risk/Reward + Model Details */}
          <div className="grid gap-4 sm:grid-cols-2 mb-6">
            {/* Risk/Reward Ratio */}
            <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm dark:border-gray-700 dark:bg-gray-800">
              <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100 mb-3">Risk / Reward</h3>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-gray-500 dark:text-gray-400">Risk (SL distance):</span>
                  <p className="font-mono font-medium text-gray-900 dark:text-gray-100">
                    {formatPrice(Math.abs(setup.entry_price - setup.stop_loss))}
                  </p>
                </div>
                <div>
                  <span className="text-gray-500 dark:text-gray-400">Reward (TP distance):</span>
                  <p className="font-mono font-medium text-gray-900 dark:text-gray-100">
                    {formatPrice(Math.abs(setup.take_profit - setup.entry_price))}
                  </p>
                </div>
              </div>
              <div className="mt-3 pt-3 border-t border-gray-100 dark:border-gray-700">
                <span className="text-gray-500 dark:text-gray-400 text-sm">Ratio:</span>
                <p className="text-lg font-bold text-gray-900 dark:text-gray-100">
                  1 : {setup.rr_ratio ? setup.rr_ratio.toFixed(2) : (Math.abs(setup.take_profit - setup.entry_price) / Math.abs(setup.entry_price - setup.stop_loss)).toFixed(2)}
                </p>
                {setup.rr_backtest_info && (
                  <div className="mt-2 flex items-center gap-1.5">
                    <span className={`inline-block h-2 w-2 rounded-full ${setup.rr_backtest_info.backtested ? 'bg-green-500' : 'bg-amber-500'}`} />
                    <span className="text-xs text-gray-500 dark:text-gray-400">
                      {setup.rr_backtest_info.backtested ? 'Dynamically selected from backtest' : 'Using minimum ratio (insufficient data)'}
                    </span>
                  </div>
                )}
              </div>
              {setup.rr_backtest_info && setup.rr_backtest_info.reason && (
                <p className="mt-2 text-xs text-gray-400 dark:text-gray-500 italic">
                  {setup.rr_backtest_info.reason}
                </p>
              )}
            </div>

            {/* Model Details */}
            <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm dark:border-gray-700 dark:bg-gray-800">
              <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100 mb-3">Model Details</h3>
              <div className="space-y-2 text-sm">
                <div>
                  <span className="text-gray-500 dark:text-gray-400">Pair:</span>{' '}
                  <span className="font-medium dark:text-gray-100">{formatPair(setup.pair)}</span>
                </div>
                <div>
                  <span className="text-gray-500 dark:text-gray-400">Model:</span>{' '}
                  <span className="font-mono text-xs font-medium dark:text-gray-100">{setup.model_name}</span>
                </div>
                <div>
                  <span className="text-gray-500 dark:text-gray-400">Version:</span>{' '}
                  <span className="font-mono text-xs font-medium dark:text-gray-100">{setup.model_version}</span>
                </div>
                <div>
                  <span className="text-gray-500 dark:text-gray-400">ATR (14):</span>{' '}
                  <span className="font-mono text-xs font-medium dark:text-gray-100">{formatPrice(setup.atr)}</span>
                </div>
                <div>
                  <span className="text-gray-500 dark:text-gray-400">Generated:</span>{' '}
                  <span className="font-medium dark:text-gray-100">
                    {new Date(setup.generated_at).toLocaleString()}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Disclaimer */}
          <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 dark:border-amber-800/50 dark:bg-amber-900/20">
            <div className="flex items-start gap-3">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-5 w-5 text-amber-500 dark:text-amber-400 mt-0.5 shrink-0"
                fill="none"
                viewBox="0 0 24 24"
                strokeWidth={1.5}
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M12 9v3.75m9-.75a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9 3.75h.008v.008H12v-.008Z"
                />
              </svg>
              <p className="text-xs text-amber-800 dark:text-amber-300">
                This trade setup is generated by an ML model and is for informational purposes only.
                It does not constitute financial advice. Always perform your own analysis and risk
                assessment before placing any trades.
              </p>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
