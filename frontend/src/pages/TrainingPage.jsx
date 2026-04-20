import { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { getPairs, trainModel } from '../api';
import Spinner from '../components/Spinner';
import ErrorMessage from '../components/ErrorMessage';

const TIMEFRAME_OPTIONS = [
  { value: '5m', label: '5 Minutes' },
  { value: '15m', label: '15 Minutes' },
  { value: '1h', label: '1 Hour' },
  { value: '4h', label: '4 Hours' },
];

export default function TrainingPage() {
  const [searchParams] = useSearchParams();
  const pairParam = searchParams.get('pair') || '';

  const [pairs, setPairs] = useState([]);
  const [selectedPair, setSelectedPair] = useState(pairParam);
  const [selectedTimeframe, setSelectedTimeframe] = useState('1h');
  const [loading, setLoading] = useState(false);
  const [training, setTraining] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  useEffect(() => {
    const fetchPairs = async () => {
      setLoading(true);
      try {
        const data = await getPairs();
        setPairs(Array.isArray(data.pairs) ? data.pairs : []);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    fetchPairs();
  }, []);

  useEffect(() => {
    if (pairParam) {
      setSelectedPair(pairParam);
    }
  }, [pairParam]);

  const handleTrain = async () => {
    if (!selectedPair) return;
    setTraining(true);
    setError(null);
    setResult(null);
    try {
      const data = await trainModel(selectedPair, selectedTimeframe);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setTraining(false);
    }
  };

  if (loading) return <Spinner className="py-20" />;
  if (error && !training) return <ErrorMessage message={error} />;

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Model Training</h1>
        <p className="mt-1 text-sm text-gray-500">
          Select a forex pair and start a new model training session.
        </p>
      </div>

      <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
        <div className="flex flex-col sm:flex-row gap-4 items-end">
          <div className="flex-1 w-full">
            <label
              htmlFor="pair-select"
              className="block text-sm font-medium text-gray-700 mb-1"
            >
              Forex Pair
            </label>
            <select
              id="pair-select"
              value={selectedPair}
              onChange={(e) => setSelectedPair(e.target.value)}
              className="block w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm text-gray-900 shadow-sm focus:border-primary-500 focus:ring-1 focus:ring-primary-500"
            >
              <option value="">Select a pair...</option>
              {pairs.map((pair) => (
                <option key={pair} value={pair}>
                  {pair}
                </option>
              ))}
            </select>
          </div>

          <div className="flex-1 w-full">
            <label
              htmlFor="timeframe-select"
              className="block text-sm font-medium text-gray-700 mb-1"
            >
              Time Frame
            </label>
            <select
              id="timeframe-select"
              value={selectedTimeframe}
              onChange={(e) => setSelectedTimeframe(e.target.value)}
              className="block w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm text-gray-900 shadow-sm focus:border-primary-500 focus:ring-1 focus:ring-primary-500"
            >
              {TIMEFRAME_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>

          <button
            onClick={handleTrain}
            disabled={!selectedPair || training}
            className="rounded-lg bg-primary-600 px-4 py-2 text-sm font-medium text-white hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors whitespace-nowrap"
          >
            {training ? <Spinner className="h-5 w-5 inline" /> : 'Train Model'}
          </button>
        </div>
      </div>

      {training && (
        <div className="mt-6">
          <Spinner className="py-10" />
          <p className="text-center text-sm text-gray-500 mt-2">
            Training model for {selectedPair}...
          </p>
        </div>
      )}

      {result && (
        <div className="mt-6 rounded-xl border border-gray-200 bg-white shadow-sm overflow-hidden">
          <div className="px-6 py-4 border-b border-gray-200 bg-green-50">
            <h2 className="text-lg font-semibold text-green-800">
              Training Complete
            </h2>
          </div>

          <div className="px-6 py-4">
            <table className="w-full text-sm">
              <tbody className="divide-y divide-gray-200">
                {result.model_name && (
                  <tr>
                    <td className="py-2 pr-4 font-medium text-gray-700 whitespace-nowrap">
                      Model
                    </td>
                    <td className="py-2 text-gray-900">{result.model_name}</td>
                  </tr>
                )}
                {result.version != null && (
                  <tr>
                    <td className="py-2 pr-4 font-medium text-gray-700 whitespace-nowrap">
                      Version
                    </td>
                    <td className="py-2 text-gray-900">{result.version}</td>
                  </tr>
                )}
                {result.model_type && (
                  <tr>
                    <td className="py-2 pr-4 font-medium text-gray-700 whitespace-nowrap">
                      Type
                    </td>
                    <td className="py-2 text-gray-900">{result.model_type}</td>
                  </tr>
                )}
                {result.trained_at && (
                  <tr>
                    <td className="py-2 pr-4 font-medium text-gray-700 whitespace-nowrap">
                      Trained At
                    </td>
                    <td className="py-2 text-gray-900">
                      {new Date(result.trained_at).toLocaleString()}
                    </td>
                  </tr>
                )}
                {result.training_duration != null && (
                  <tr>
                    <td className="py-2 pr-4 font-medium text-gray-700 whitespace-nowrap">
                      Duration
                    </td>
                    <td className="py-2 text-gray-900">
                      {result.training_duration}s
                    </td>
                  </tr>
                )}
                {result.status && (
                  <tr>
                    <td className="py-2 pr-4 font-medium text-gray-700 whitespace-nowrap">
                      Status
                    </td>
                    <td className="py-2 text-gray-900">
                      <span className="inline-flex items-center rounded-full bg-green-50 px-2.5 py-0.5 text-xs font-medium text-green-700 ring-1 ring-inset ring-green-600/20">
                        {result.status}
                      </span>
                    </td>
                  </tr>
                )}
              </tbody>
            </table>

            {result.test_metrics &&
              typeof result.test_metrics === 'object' &&
              !Array.isArray(result.test_metrics) &&
              Object.keys(result.test_metrics).length > 0 && (
                <div className="mt-6">
                  <h3 className="text-sm font-semibold text-gray-700 mb-3">
                    Test Metrics
                  </h3>
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-gray-200">
                        <th className="text-left py-2 pr-4 font-medium text-gray-700">
                          Metric
                        </th>
                        <th className="text-left py-2 font-medium text-gray-700">
                          Value
                        </th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-200">
                      {Object.entries(result.test_metrics).map(
                        ([key, value]) => (
                          <tr key={key}>
                            <td className="py-2 pr-4 text-gray-700">{key}</td>
                            <td className="py-2 text-gray-900">
                              {typeof value === 'number'
                                ? value.toFixed(4)
                                : String(value ?? '')}
                            </td>
                          </tr>
                        )
                      )}
                    </tbody>
                  </table>
                </div>
              )}
          </div>
        </div>
      )}

      {!loading && !training && !result && !error && (
        <div className="mt-8 text-center text-gray-400">
          <p className="text-sm">No training results yet. Select a pair and click Train Model to get started.</p>
        </div>
      )}
    </div>
  );
}