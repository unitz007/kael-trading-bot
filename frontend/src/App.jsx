import { useState } from 'react';
import { Routes, Route } from 'react-router-dom';
import Navbar, { TopBar } from './components/Navbar';
import ThemeToggle from './components/ThemeToggle';
import ForexPairsPage from './pages/ForexPairsPage';
import TrainingPage from './pages/TrainingPage';
import PredictionsPage from './pages/PredictionsPage';
import ForecastPage from './pages/ForecastPage';
import TradeSetupPage from './pages/TradeSetupPage';
import LiveSetupsPage from './pages/LiveSetupsPage';
import AccuracyPage from './pages/AccuracyPage';
import ChartPage from './pages/ChartPage';

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="min-h-screen flex bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-100">
      <Navbar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />

      <div className="flex-1 flex flex-col min-w-0">
        <TopBar onMenuToggle={() => setSidebarOpen(true)} />

        <main className="flex-1 px-4 py-6 sm:px-6 lg:px-8 max-w-7xl w-full mx-auto">
          <Routes>
            <Route path="/" element={<ForexPairsPage />} />
            <Route path="/training" element={<TrainingPage />} />
            <Route path="/predictions" element={<PredictionsPage />} />
            <Route path="/forecast" element={<ForecastPage />} />
            <Route path="/trade-setup" element={<TradeSetupPage />} />
            <Route path="/live-setups" element={<LiveSetupsPage />} />
            <Route path="/accuracy" element={<AccuracyPage />} />
            <Route path="/chart" element={<ChartPage />} />
          </Routes>
        </main>
        {/* Fixed theme toggle at bottom-left of viewport */}
        <div className="fixed bottom-4 left-4 z-50">
          <ThemeToggle />
        </div>
      </div>
    </div>
  );
}