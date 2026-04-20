import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { getPairs, getTradeSetups } from '../api';
import Spinner from '../components/Spinner';
import ErrorMessage from '../components/ErrorMessage';

const TIMEFRAME_OPTIONS = [
  { value: '5m', label: '5 Minutes' },
  { value: '15m', label: '15 Minutes' },
  { value: '1h', label: '1 Hour' },
  { value: '4h', label: '4 Hours' },
];

const DIRECTION_OPTIONS = [
  { value: '', label: 'All Directions' },
  { value: 'buy', label: 'Buy (Long)' },
  { value: 'sell', label: 'Sell (Short)' },
];

const SORT_FIELDS = [
  { value: 'generated_at', label: 'Detection Time' },
  { value: 'pair', label: 'Pair' },
  { value: 'confidence', label: 'Confidence' },
  { value: 'rr_ratio', label: 'R:R Ratio' },
  { value: 'entry_price', label: 'Entry Price' },
];

const DEFAULT_STALE_THRESHOLD_MINUTES = 60;
const DEFAULT_REFRESH_INTERVAL_SECONDS = 60;

function formatPair(ticker) {
  return ticker.replace('=X', '');
}

function formatPrice(price) {
  return Number(price).toFixed(5);
}

function formatTimestamp(isoString) {
  if (!isoString) return '—';
  const date = new Date(isoString);
  return date.toLocaleString();
}

function isStale(isoString, thresholdMinutes) {
  if (!isoString) return true;
  const ageMs = Date.now() - new Date(isoString).getTime();
  return ageMs > thresholdMinutes * 60 * 1000;
}

function timeAgo(isoString) {
  if (!isoString) return '';
  const seconds = Math.floor((Date.now() - new Date(isoString).getTime()) / 1000);
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

function EmptyState() {
  return (
    <div className="rounded-xl border border-gray-200 bg-white p-12 text-center">
      <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-gray-100">
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className="h-8 w-8 text-gray-400"
          fill="none"
          viewBox="0 0 24 24"
          strokeWidth={1.5}
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M2.25 18 9 11.25l4.306 4.306a11.95 11.95 0 0 1 5.814-5.518l2.74-1.22m0 0-5.94-2.281m5.94 2.28-2.28 5.941"
          />
        </svg>
      </div>
      <h3 className="mt-4 text-sm font-semibold text-gray-900">No trade setups found</h3>
      <p className="mt-2 text-sm text-gray-500">
        There are no recent trade setups available. Setups appear here once models are
        trained and have generated signals. Try changing the timeframe filter or refresh
        later.
      </p>
    </div>
  );
}

function SetupCard({ setup, stale }) {
  const directionBadge =
    setup.direction === 'buy'
      ? 'bg-green-100 text-green-800'
      : 'bg-red-100 text-red-800';

  return (
    <div
      className={`rounded-xl border bg-white p-5 shadow-sm transition-all ${
        stale
          ? 'border-gray-200 opacity-50'
          : 'border-gray-200 hover:shadow-md'
      }`}
    >
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-3">
          <span className="text-lg font-bold text-gray-900">
            {formatPair(setup.pair)}
          </span>
          <span
            className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold ${directionBadge}`}
          >
            {setup.direction === 'buy' ? '▲ LONG' : '▼ SHORT'}
          </span>
          <span className="rounded-md bg-gray-100 px-2 py-0.5 text-xs font-medium text-gray-600">
            {setup.timeframe || '—'}
          </span>
          {stale && (
            <span className="rounded-md bg-amber-100 px-2 py-0.5 text-xs font-medium text-amber-700">
              Stale
            </span>
          )}
        </div>
        <span className="text-xs text-gray-400 whitespace-nowrap ml-2">
          {timeAgo(setup.generated_at)}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-x-6 gap-y-3 text-sm sm:grid-cols-4">
        <div>
          <span className="text-gray-500 text-xs uppercase tracking-wider">Entry</span>
          <p className="font-mono font-semibold text-gray-900">{formatPrice(setup.entry_price)}</p>
        </div>
        <div>
          <span className="text-gray-500 text-xs uppercase tracking-wider">Stop Loss</span>
          <p className="font-mono font-semibold text-red-600">{formatPrice(setup.stop_loss)}</p>
        </div>
        <div>
          <span className="text-gray-500 text-xs uppercase tracking-wider">Take Profit</span>
          <p className="font-mono font-semibold text-green-600">{formatPrice(setup.take_profit)}</p>
        </div>
        <div>
          <span className="text-gray-500 text-xs uppercase tracking-wider">R:R Ratio</span>
          <p className="font-mono font-semibold text-gray-900">
            1:{setup.rr_ratio ? setup.rr_ratio.toFixed(2) : '—'}
          </p>
        </div>
      </div>

      <div className="mt-4 flex items-center justify-between border-t border-gray-100 pt-3">
        <div className="flex items-center gap-2">
          <div className="h-2 w-full rounded-full bg-gray-200 overflow-hidden" style={{ width: '80px' }}>
            <div
              className={`h-full rounded-full ${
                setup.confidence >= 0.7
                  ? 'bg-green-500'
                  : setup.confidence >= 0.5
                  ? 'bg-yellow-500'
                  : 'bg-red-500'
              }`}
              style={{ width: `${Math.round(setup.confidence * 100)}%` }}
            />
          </div>
          <span className="text-xs text-gray-500">
            {(setup.confidence * 100).toFixed(1)}% confidence
          </span>
        </div>
        <span className="text-xs text-gray-400" title={formatTimestamp(setup.generated_at)}>
          {formatTimestamp(setup.generated_at)}
        </span>
      </div>
    </div>
  );
}

export default function LiveSetupsPage() {
  const [setups, setSetups] = useState([]);
  const [pairs, setPairs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filterPair, setFilterPair] = useState('');
  const [filterDirection, setFilterDirection] = useState('');
  const [filterTimeframe, setFilterTimeframe] = useState('1h');
  const [sortField, setSortField] = useState('generated_at');
  const [sortDir, setSortDir] = useState('desc');
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(DEFAULT_REFRESH_INTERVAL_SECONDS);
  const [staleThreshold, setStaleThreshold] = useState(DEFAULT_STALE_THRESHOLD_MINUTES);
  const [hideStale, setHideStale] = useState(false);
  const [lastRefreshed, setLastRefreshed] = useState(null);
  const intervalRef = useRef(null);
  const [pairsLoading, setPairsLoading] = useState(true);

  // Fetch available pairs on mount
  useEffect(() => {
    let cancelled = false;
    async function fetchPairs() {
      try {
        const data = await getPairs();
        if (!cancelled) {
          setPairs(data.pairs || []);
        }
      } catch {
        // Non-critical — filters will just be empty
      } finally {
        if (!cancelled) setPairsLoading(false);
      }
    }
    fetchPairs();
    return () => { cancelled = true; };
  }, []);


  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const data = await getTradeSetups(filterTimeframe);
        if (!cancelled) {
          setSetups(data.setups || []);
          setLastRefreshed(new Date().toLocaleTimeString());
        }
      } catch (err) {
        if (!cancelled) setError(err.message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => { cancelled = true; };
  }, [filterTimeframe]);

  // Auto-refresh timer
  useEffect(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (!autoRefresh || refreshInterval <= 0) return;

    intervalRef.current = setInterval(async () => {
      try {
        const data = await getTradeSetups(filterTimeframe);
        setSetups(data.setups || []);
        setLastRefreshed(new Date().toLocaleTimeString());
        setError(null);
      } catch (err) {
        // Don't overwrite main error on background refresh
      }
    }, refreshInterval * 1000);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [autoRefresh, refreshInterval, filterTimeframe]);


  const displayedSetups = useMemo(() => {
    let result = [...setups];

    // Filter by pair
    if (filterPair) {
      result = result.filter((s) => s.pair === filterPair);
    }

    // Filter by direction
    if (filterDirection) {
      result = result.filter((s) => s.direction === filterDirection);
    }

    // Hide stale
    if (hideStale) {
      result = result.filter((s) => !isStale(s.generated_at, staleThreshold));
    }

    // Sort
    result.sort((a, b) => {
      let valA = a[sortField];
      let valB = b[sortField];

      if (sortField === 'generated_at') {
        valA = valA ? new Date(valA).getTime() : 0;
        valB = valB ? new Date(valB).getTime() : 0;
      }

      if (valA < valB) return sortDir === 'asc' ? -1 : 1;
      if (valA > valB) return sortDir === 'asc' ? 1 : -1;
      return 0;
    });

    return result;
  }, [setups, filterPair, filterDirection, hideStale, staleThreshold, sortField, sortDir]);

  // Manual refresh
  const handleRefresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getTradeSetups(filterTimeframe);
      setSetups(data.setups || []);
      setLastRefreshed(new Date().toLocaleTimeString());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [filterTimeframe]);

  return (
    <div>
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Live Setups</h1>
          <p className="mt-1 text-sm text-gray-500">
            Auto-detected trade setups from the latest model scans across all pairs.
          </p>
        </div>
        <button
          onClick={handleRefresh}
          disabled={loading}
          className="inline-flex items-center justify-center gap-2 rounded-lg bg-primary-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-primary-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed shrink-0"
        >
          {loading ? (
            <>
              <div className="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
              Refreshing…
            </>
          ) : (
            <>
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-4 w-4"
                fill="none"
                viewBox="0 0 24 24"
                strokeWidth={2}
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0 3.181 3.183a8.25 8.25 0 0 0 13.803-3.7M4.031 9.865a8.25 8.25 0 0 1 13.803-3.7l3.181 3.182"
                />
              </svg>
              Refresh
            </>
          )}
        </button>
      </div>

      {/* Filters & Controls */}
      <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm mb-6">
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {/* Pair filter */}
          <div>
            <label htmlFor="filter-pair" className="block text-xs font-medium text-gray-500 uppercase tracking-wider mb-1.5">
              Pair
            </label>
            <select
              id="filter-pair"
              value={filterPair}
              onChange={(e) => setFilterPair(e.target.value)}
              className="block w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm text-gray-900 focus:border-primary-500 focus:ring-2 focus:ring-primary-200 focus:outline-none"
            >
              <option value="">All Pairs</option>
              {pairs.map((pair) => (
                <option key={pair} value={pair}>
                  {formatPair(pair)}
                </option>
              ))}
            </select>
          </div>

          {/* Direction filter */}
          <div>
            <label htmlFor="filter-direction" className="block text-xs font-medium text-gray-500 uppercase tracking-wider mb-1.5">
              Direction
            </label>
            <select
              id="filter-direction"
              value={filterDirection}
              onChange={(e) => setFilterDirection(e.target.value)}
              className="block w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm text-gray-900 focus:border-primary-500 focus:ring-2 focus:ring-primary-200 focus:outline-none"
            >
              {DIRECTION_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>

          {/* Timeframe filter */}
          <div>
            <label htmlFor="filter-timeframe" className="block text-xs font-medium text-gray-500 uppercase tracking-wider mb-1.5">
              Timeframe
            </label>
            <select
              id="filter-timeframe"
              value={filterTimeframe}
              onChange={(e) => setFilterTimeframe(e.target.value)}
              className="block w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm text-gray-900 focus:border-primary-500 focus:ring-2 focus:ring-primary-200 focus:outline-none"
            >
              {TIMEFRAME_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>

          {/* Sort */}
          <div>
            <label htmlFor="sort-field" className="block text-xs font-medium text-gray-500 uppercase tracking-wider mb-1.5">
              Sort By
            </label>
            <div className="flex gap-2">
              <select
                id="sort-field"
                value={sortField}
                onChange={(e) => {
                  setSortField(e.target.value);
                  if (e.target.value !== sortField) setSortDir('desc');
                }}
                className="block flex-1 rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm text-gray-900 focus:border-primary-500 focus:ring-2 focus:ring-primary-200 focus:outline-none"
              >
                {SORT_FIELDS.map((f) => (
                  <option key={f.value} value={f.value}>
                    {f.label}
                  </option>
                ))}
              </select>
              <button
                onClick={() => setSortDir((d) => (d === 'desc' ? 'asc' : 'desc'))}
                className="inline-flex items-center justify-center rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm text-gray-600 hover:bg-gray-50 transition-colors"
                title={sortDir === 'desc' ? 'Sort ascending' : 'Sort descending'}
              >
                {sortDir === 'desc' ? (
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 13.5 12 21m0 0-7.5-7.5M12 21V3" />
                  </svg>
                ) : (
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 10.5 12 3m0 0 7.5 7.5M12 3v18" />
                  </svg>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Auto-refresh & stale controls */}
        <div className="mt-4 pt-4 border-t border-gray-100 flex flex-wrap items-center gap-x-6 gap-y-3">
          <label className="inline-flex items-center gap-2 text-sm text-gray-600 cursor-pointer">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="h-4 w-4 rounded border-gray-300 text-primary-600 focus:ring-primary-500"
            />
            Auto-refresh
          </label>

          {autoRefresh && (
            <div className="flex items-center gap-2 text-sm text-gray-600">
              <span>every</span>
              <input
                type="number"
                min={10}
                max={600}
                value={refreshInterval}
                onChange={(e) => setRefreshInterval(Math.max(10, Number(e.target.value) || 60))}
                className="w-16 rounded-lg border border-gray-300 px-2 py-1 text-sm text-center focus:border-primary-500 focus:ring-1 focus:ring-primary-200 focus:outline-none"
              />
              <span>seconds</span>
            </div>
          )}

          <div className="flex items-center gap-2 text-sm text-gray-600">
            <span>Stale after</span>
            <input
              type="number"
              min={5}
              max={1440}
              value={staleThreshold}
              onChange={(e) => setStaleThreshold(Math.max(5, Number(e.target.value) || 60))}
              className="w-16 rounded-lg border border-gray-300 px-2 py-1 text-sm text-center focus:border-primary-500 focus:ring-1 focus:ring-primary-200 focus:outline-none"
            />
            <span>min</span>
          </div>

          <label className="inline-flex items-center gap-2 text-sm text-gray-600 cursor-pointer">
            <input
              type="checkbox"
              checked={hideStale}
              onChange={(e) => setHideStale(e.target.checked)}
              className="h-4 w-4 rounded border-gray-300 text-primary-600 focus:ring-primary-500"
            />
            Hide stale setups
          </label>
        </div>
      </div>

      {/* Status bar */}
      {(lastRefreshed || displayedSetups.length > 0) && (
        <div className="flex items-center justify-between text-xs text-gray-400 mb-4 px-1">
          <span>
            {displayedSetups.length} setup{displayedSetups.length !== 1 ? 's' : ''} shown
            {setups.length !== displayedSetups.length && (
              <> ({setups.length} total)</>
            )}
          </span>
          {lastRefreshed && (
            <span>
              Last refreshed: {lastRefreshed}
              {autoRefresh && (
                <span className="ml-1 inline-flex items-center gap-1">
                  <span className="h-1.5 w-1.5 rounded-full bg-green-500 animate-pulse" />
                  Live
                </span>
              )}
            </span>
          )}
        </div>
      )}

      {/* Error */}
      {error && !loading && <ErrorMessage message={error} onRetry={handleRefresh} />}

      {/* Loading */}
      {loading && <Spinner className="py-12" />}

      {/* Content */}
      {!loading && !error && (
        <>
          {displayedSetups.length === 0 ? (
            <EmptyState />
          ) : (
            <div className="grid gap-4">
              {displayedSetups.map((setup) => (
                <SetupCard
                  key={`${setup.pair}-${setup.timeframe}-${setup.generated_at}`}
                  setup={setup}
                  stale={isStale(setup.generated_at, staleThreshold)}
                />
              ))}
            </div>
          )}
        </>
      )}

      {/* Disclaimer */}
      <div className="mt-8 rounded-xl border border-amber-200 bg-amber-50 p-4">
        <div className="flex items-start gap-3">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-5 w-5 text-amber-500 mt-0.5 shrink-0"
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
          <p className="text-xs text-amber-800">
            Trade setups shown here are generated by ML models and are for informational purposes only.
            They do not constitute financial advice. Always perform your own analysis and risk assessment
            before placing any trades.
          </p>
        </div>
      </div>
    </div>
  );
}