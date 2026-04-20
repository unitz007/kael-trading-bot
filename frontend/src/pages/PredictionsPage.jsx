import { useState, useEffect, useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import { getPairs, getPredictions } from '../api';
import Spinner from '../components/Spinner';
import ErrorMessage from '../components/ErrorMessage';

function formatPair(ticker) {
  return ticker.replace('=X', '');
}

export default function PredictionsPage() {
  const [searchParams] = useSearchParams();
  const preselectedPair = searchParams.get('pair') || '';

  const [pairs, setPairs] = useState([]);
  const [selectedPair, setSelectedPair] = useState(preselectedPair);
  const [loading, setLoading] = useState(false);
  const [predictions, setPredictions] = useState(null);
  const [error, setError] = useState(null);
  const [pairsLoading, setPairsLoading] = useState(true);

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

  const fetchPredictions = useCallback(async () => {
    if (!selectedPair) return;
    setLoading(true);
    setPredictions(null);
    setError(null);
    try {
      const data = await getPredictions(selectedPair);
      setPredictions(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [selectedPair]);

  if (pairsLoading) return <Spinner className="py-20" />;

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">Predictions</h1>
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
          View prediction results from the latest trained model for a forex pair.
        </p>
      </div>

      {/* Pair Selection */}
      <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-5 shadow-sm mb-6">
        <label htmlFor="pred-pair-select" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Select Forex Pair
        </label>
        <div className="flex flex-col sm:flex-row gap-3">
          <select
            id="pred-pair-select"
            value={selectedPair}
            onChange={(e) => {
              setSelectedPair(e.target.value);
              setPredictions(null);
              setError(null);
            }}
            disabled={loading}
            className="block w-full max-w-xs rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 px-4 py-2.5 text-sm focus:border-primary-500 focus:ring-2 focus:ring-primary-200 dark:focus:ring-primary-800 focus:outline-none disabled:opacity-50"
          >
            {pairs.map((pair) => (
              <option key={pair} value={pair}>
                {formatPair(pair)}
              </option>
            ))}
          </select>

          <button
            onClick={fetchPredictions}
            disabled={loading || !selectedPair}
            className="inline-flex items-center justify-center gap-2 rounded-lg bg-primary-600 dark:bg-primary-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-primary-700 dark:hover:bg-primary-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 dark:focus:ring-offset-gray-800"
          >
            {loading ? (
              <>
                <div className="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                Loading...
              </>
            ) : (
              'Get Predictions'
            )}
          </button>
        </div>
      </div>

      {/* Loading */}
      {loading && (
        <div className="rounded-xl border border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-900/20 p-5 mb-6">
          <div className="flex items-center gap-3">
            <Spinner className="py-0" />
            <p className="text-sm text-blue-700 dark:text-blue-200">
              Fetching predictions for {formatPair(selectedPair)}...
            </p>
          </div>
        </div>
      )}

      {/* Error */}
      {error && !loading && <ErrorMessage message={error} onRetry={fetchPredictions} />}

      {/* Predictions Results */}
      {predictions && (
        <>
          {/* Summary Cards */}
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4 mb-6">
            <SummaryCard
              label="Total Predictions"
              value={predictions.total_predictions?.toLocaleString()}
            />
            <SummaryCard
              label="Predicted UP"
              value={predictions.up_count?.toLocaleString()}
              color="green"
            />
            <SummaryCard
              label="Predicted DOWN"
              value={predictions.down_count?.toLocaleString()}
              color="red"
            />
            <SummaryCard
              label="Model Version"
              value={predictions.model_version}
            />
          </div>

          {/* Model Info */}
          <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-5 shadow-sm mb-6">
            <h2 className="text-sm font-semibold text-gray-900 dark:text-gray-100 mb-3">Model Information</h2>
            <div className="grid gap-3 sm:grid-cols-3 text-sm">
              <div>
                <span className="text-gray-500 dark:text-gray-400">Pair:</span>{' '}
                <span className="font-medium text-gray-900 dark:text-gray-100">{formatPair(predictions.pair)}</span>
              </div>
              <div>
                <span className="text-gray-500 dark:text-gray-400">Model Name:</span>{' '}
                <span className="font-medium font-mono text-xs text-gray-900 dark:text-gray-100">{predictions.model_name}</span>
              </div>
              {predictions.trained_at && (
                <div>
                  <span className="text-gray-500 dark:text-gray-400">Trained At:</span>{' '}
                  <span className="font-medium text-gray-900 dark:text-gray-100">
                    {new Date(predictions.trained_at).toLocaleString()}
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* Predictions Table */}
          {predictions.predictions?.length > 0 && (
            <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 shadow-sm overflow-hidden">
              <div className="px-5 py-4 border-b border-gray-200 dark:border-gray-700">
                <h2 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
                  Prediction Results
                  <span className="ml-2 text-xs font-normal text-gray-500 dark:text-gray-400">
                    Showing most recent predictions
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
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                        Prediction
                      </th>
                      <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                        Probability (UP)
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                    {[...predictions.predictions].reverse().map((pred, idx) => (
                      <tr key={idx} className="hover:bg-gray-50 dark:hover:bg-gray-700/50">
                        <td className="px-4 py-3 text-sm text-gray-900 dark:text-gray-100 whitespace-nowrap">
                          {pred.date}
                        </td>
                        <td className="px-4 py-3 whitespace-nowrap">
                          <span
                            className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium ${
                              pred.prediction === 'UP'
                                ? 'bg-green-50 dark:bg-green-900/30 text-green-700 dark:text-green-400 ring-1 ring-inset ring-green-600/20 dark:ring-green-500/30'
                                : 'bg-red-50 dark:bg-red-900/30 text-red-700 dark:text-red-400 ring-1 ring-inset ring-red-600/20 dark:ring-red-500/30'
                            }`}
                          >
                            <span
                              className={`h-1.5 w-1.5 rounded-full ${
                                pred.prediction === 'UP' ? 'bg-green-500 dark:bg-green-400' : 'bg-red-500 dark:bg-red-400'
                              }`}
                            />
                            {pred.prediction}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-sm text-right text-gray-900 dark:text-gray-100 font-mono">
                          {pred.probability_up != null
                            ? `${(pred.probability_up * 100).toFixed(1)}%`
                            : '—'}
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

function SummaryCard({ label, value, color = 'blue' }) {
  const colorClasses = {
    blue: 'bg-primary-50 dark:bg-primary-900/30 text-primary-700 dark:text-primary-400',
    green: 'bg-green-50 dark:bg-green-900/30 text-green-700 dark:text-green-400',
    red: 'bg-red-50 dark:bg-red-900/30 text-red-700 dark:text-red-400',
  };

  return (
    <div className={`rounded-xl p-4 ${colorClasses[color] || colorClasses.blue}`}>
      <p className="text-xs font-medium uppercase tracking-wider opacity-90">{label}</p>
      <p className="mt-1 text-2xl font-bold">{value}</p>
    </div>
  );
}