import { useState, useEffect, useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import { getPairs, trainModel } from '../api';
import Spinner from '../components/Spinner';
import ErrorMessage from '../components/ErrorMessage';

function formatPair(ticker) {
  return ticker.replace('=X', '');
}

export default function TrainingPage() {
  const [searchParams] = useSearchParams();
  const preselectedPair = searchParams.get('pair') || '';

  const [pairs, setPairs] = useState([]);
  const [selectedPair, setSelectedPair] = useState(preselectedPair);
  const [training, setTraining] = useState(false);
  const [result, setResult] = useState(null);
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

  const handleTrain = useCallback(async () => {
    if (!selectedPair || training) return;
    setTraining(true);
    setResult(null);
    setError(null);
    try {
      const data = await trainModel(selectedPair);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setTraining(false);
    }
  }, [selectedPair, training]);

  if (pairsLoading) return <Spinner className="py-20" />;

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Model Training</h1>
        <p className="mt-1 text-sm text-gray-500">
          Train an XGBoost model for a forex pair using historical data.
        </p>
      </div>

      {/* Pair Selection */}
      <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm mb-6">
        <label htmlFor="pair-select" className="block text-sm font-medium text-gray-700 mb-2">
          Select Forex Pair
        </label>
        <select
          id="pair-select"
          value={selectedPair}
          onChange={(e) => {
            setSelectedPair(e.target.value);
            setResult(null);
            setError(null);
          }}
          disabled={training}
          className="block w-full max-w-xs rounded-lg border border-gray-300 bg-white px-4 py-2.5 text-sm text-gray-900 focus:border-primary-500 focus:ring-2 focus:ring-primary-200 focus:outline-none disabled:opacity-50"
        >
          {pairs.map((pair) => (
            <option key={pair} value={pair}>
              {formatPair(pair)}
            </option>
          ))}
        </select>

        <button
          onClick={handleTrain}
          disabled={training || !selectedPair}
          className="mt-4 inline-flex items-center gap-2 rounded-lg bg-primary-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-primary-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {training && (
            <div className="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
          )}
          {training ? 'Training Model...' : 'Train Model'}
        </button>
      </div>

      {/* Status during training */}
      {training && (
        <div className="rounded-xl border border-blue-200 bg-blue-50 p-5 mb-6">
          <div className="flex items-center gap-3">
            <Spinner className="py-0" />
            <div>
              <p className="text-sm font-medium text-blue-800">Training in progress</p>
              <p className="text-sm text-blue-700">
                Training model for {formatPair(selectedPair)}. This may take a moment...
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Error */}
      {error && !training && <ErrorMessage message={error} onRetry={handleTrain} />}

      {/* Results */}
      {result && (
        <div className="rounded-xl border border-green-200 bg-green-50 p-5">
          <div className="flex items-center gap-2 mb-4">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-5 w-5 text-green-600"
              fill="none"
              viewBox="0 0 24 24"
              strokeWidth={1.5}
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M9 12.75 11.25 15 15 9.75M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z"
              />
            </svg>
            <h2 className="text-lg font-semibold text-green-800">Training Complete</h2>
          </div>

          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            <ResultCard label="Forex Pair" value={formatPair(result.pair)} />
            <ResultCard label="Model Name" value={result.model_name} />
            <ResultCard label="Model Version" value={result.model_version} />
            <ResultCard label="Model Type" value={result.model_type} />
            <ResultCard label="Duration" value={`${result.duration_seconds}s`} />
            <ResultCard label="Samples Trained" value={result.samples_trained?.toLocaleString()} />
            <ResultCard label="Features Used" value={result.num_features} />
            {result.saved_path && (
              <ResultCard label="Saved Path" value={result.saved_path} />
            )}
          </div>

          {result.test_metrics && (
            <div className="mt-6">
              <h3 className="text-sm font-semibold text-green-800 mb-3">Test Metrics</h3>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-green-200">
                  <thead>
                    <tr>
                      <th className="px-4 py-2 text-left text-xs font-medium text-green-700 uppercase tracking-wider">Metric</th>
                      <th className="px-4 py-2 text-right text-xs font-medium text-green-700 uppercase tracking-wider">Value</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-green-200">
                    {Object.entries(result.test_metrics).map(([key, value]) => (
                      <tr key={key}>
                        <td className="px-4 py-2 text-sm text-green-900 capitalize">
                          {key.replace(/_/g, ' ')}
                        </td>
                        <td className="px-4 py-2 text-sm text-green-900 text-right font-mono">
                          {typeof value === 'number' ? value.toFixed(4) : value}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function ResultCard({ label, value }) {
  return (
    <div className="rounded-lg bg-white/60 px-4 py-3">
      <p className="text-xs font-medium text-green-600 uppercase tracking-wider">{label}</p>
      <p className="mt-1 text-sm font-semibold text-green-900 truncate" title={value}>
        {value}
      </p>
    </div>
  );
}
