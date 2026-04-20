import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { getPairs, getModels } from '../api';
import Spinner from '../components/Spinner';
import ErrorMessage from '../components/ErrorMessage';

function formatPair(ticker) {
  return ticker.replace('=X', '');
}

export default function ForexPairsPage() {
  const [pairsData, setPairsData] = useState(null);
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [pairsResult, modelsResult] = await Promise.all([
        getPairs(),
        getModels().catch(() => ({ models: [] })),
      ]);
      setPairsData(pairsResult);
      setModels(modelsResult.models || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const getModelInfo = (ticker) => {
    return models.filter((m) => m.model_name.includes(ticker.replace('=X', '').toLowerCase()));
  };

  if (loading) return <Spinner className="py-20" />;
  if (error) return <ErrorMessage message={error} onRetry={fetchData} />;

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Forex Pairs</h1>
        <p className="mt-1 text-sm text-gray-500">
          Browse available forex pairs and their trained models.
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
        {pairsData?.pairs?.map((pair) => {
          const pairModels = getModelInfo(pair);
          const latestModel = pairModels.length > 0
            ? pairModels.sort((a, b) => (b.version > a.version ? 1 : -1))[0]
            : null;

          return (
            <div
              key={pair}
              className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm hover:shadow-md transition-shadow"
            >
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold text-gray-900">
                  {formatPair(pair)}
                </h2>
                {latestModel ? (
                  <span className="inline-flex items-center rounded-full bg-green-50 px-2.5 py-0.5 text-xs font-medium text-green-700 ring-1 ring-inset ring-green-600/20">
                    Model Ready
                  </span>
                ) : (
                  <span className="inline-flex items-center rounded-full bg-gray-50 px-2.5 py-0.5 text-xs font-medium text-gray-600 ring-1 ring-inset ring-gray-500/10">
                    No Model
                  </span>
                )}
              </div>

              <div className="mt-3 space-y-1.5">
                <div className="flex items-center text-sm text-gray-500">
                  <span className="font-medium text-gray-700 w-24">Ticker:</span>
                  <code className="text-xs bg-gray-100 px-1.5 py-0.5 rounded">{pair}</code>
                </div>
                {latestModel && (
                  <>
                    <div className="flex items-center text-sm text-gray-500">
                      <span className="font-medium text-gray-700 w-24">Version:</span>
                      {latestModel.version}
                    </div>
                    <div className="flex items-center text-sm text-gray-500">
                      <span className="font-medium text-gray-700 w-24">Type:</span>
                      {latestModel.model_type}
                    </div>
                    {latestModel.trained_at && (
                      <div className="flex items-center text-sm text-gray-500">
                        <span className="font-medium text-gray-700 w-24">Trained:</span>
                        {new Date(latestModel.trained_at).toLocaleDateString()}
                      </div>
                    )}
                  </>
                )}
              </div>

              <div className="mt-4 flex gap-2">
                <Link
                  to={`/training?pair=${encodeURIComponent(pair)}`}
                  className="flex-1 rounded-lg bg-primary-600 px-3 py-2 text-center text-sm font-medium text-white hover:bg-primary-700 transition-colors"
                >
                  Train
                </Link>
                <Link
                  to={`/predictions?pair=${encodeURIComponent(pair)}`}
                  className="flex-1 rounded-lg border border-gray-300 px-3 py-2 text-center text-sm font-medium text-gray-700 hover:bg-gray-50 transition-colors"
                >
                  Predict
                </Link>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
