import { useEffect, useRef, useCallback, useState } from 'react';
import { createChart, CrosshairMode, ColorType } from 'lightweight-charts';
import { useTheme } from './ThemeProvider';

/**
 * Timeframe options matching acceptance criteria:
 * M1, M5, M15, M30, H1, H4, D1
 */
const TIMEFRAMES = [
  { label: 'M1', apiValue: '1m' },
  { label: 'M5', apiValue: '5m' },
  { label: 'M15', apiValue: '15m' },
  { label: 'M30', apiValue: '30m' },
  { label: 'H1', apiValue: '1h' },
  { label: 'H4', apiValue: '4h' },
  { label: 'D1', apiValue: '1d' },
];

/**
 * TradingViewChart – renders a professional candlestick chart using
 * lightweight-charts (TradingView's open-source charting engine).
 *
 * Props
 * -----
 * pair          – forex pair ticker string (e.g. "EURUSD=X")
 * historyData   – array of OHLCV records from the history API
 * predictions   – optional array of { date, prediction, probability_up }
 * forecast      – optional array of { date, predicted_price, upper_bound, lower_bound }
 * loading       – boolean
 * error         – string | null
 * onTimeframeChange – callback(tf: string) when user picks a timeframe
 * activeTimeframe   – currently selected timeframe string
 */
export default function TradingViewChart({
  pair,
  historyData,
  predictions = [],
  forecast = [],
  loading = false,
  error = null,
  onTimeframeChange,
  activeTimeframe = '1h',
}) {
  const { theme } = useTheme();
  const chartContainerRef = useRef(null);
  const chartAreaRef = useRef(null);
  const chartRef = useRef(null);
  const candleSeriesRef = useRef(null);
  const volumeSeriesRef = useRef(null);
  const [crosshairData, setCrosshairData] = useState(null);
  const [chartVersion, setChartVersion] = useState(0);

  const isDark = theme === 'dark';
  
  // Early exit if essential props are missing
  if (!pair) {
    return (
      <div className="flex flex-col h-full min-h-[400px]">
        {/* ---- Toolbar ---- */}
        <div className="flex flex-wrap items-center gap-2 mb-2 px-1">
          <span className="text-sm font-semibold text-gray-700 dark:text-gray-200 bg-gray-100 dark:bg-gray-800 px-3 py-1 rounded">
            Select a pair
          </span>
        </div>
        <div className="flex-1 flex items-center justify-center bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
          <p className="text-gray-500 dark:text-gray-400">Please select a trading pair to view the chart</p>
        </div>
      </div>
    );
  }

// Add helper function for validating refs
  const isValidRef = (ref) => ref && ref.current;

  // Theme-based colour palette — mimics TradingView dark & light
  const colors = isDark
    ? {
        background: '#131722',
        gridColor: '#1e222d',
        textColor: '#787b86',
        borderColor: '#2a2e39',
        candleUp: '#26a69a',
        candleDown: '#ef5350',
        volumeUp: '#26a69a80',
        volumeDown: '#ef535080',
        crosshairColor: '#758696',
        predictionLine: '#2196f3',
        forecastLine: '#ff9800',
        forecastBand: '#ff980020',
      }
    : {
        background: '#ffffff',
        gridColor: '#f0f3fa',
        textColor: '#787b86',
        borderColor: '#e0e3eb',
        candleUp: '#26a69a',
        candleDown: '#ef5350',
        volumeUp: '#26a69a60',
        volumeDown: '#ef535060',
        crosshairColor: '#758696',
        predictionLine: '#1976d2',
        forecastLine: '#e65100',
        forecastBand: '#ff980025',
      };

  // -------------------------------------------------------------------
  // Data preparation helpers
  // -------------------------------------------------------------------

  /** Build a price-proxy line from prediction directions. */
  const buildPredictionOverlay = useCallback(() => {
    if (!predictions.length || !historyData?.length) return [];

    const priceMap = {};
    historyData.forEach((d) => {
      priceMap[String(d.Date || d.date)] = parseFloat(d.Close || d.close);
    });

    return predictions
      .filter((p) => priceMap[p.date] !== undefined)
      .map((p) => {
        const base = priceMap[p.date];
        const dir = p.prediction === 'UP' ? 1 : p.prediction === 'DOWN' ? -1 : 0;
        const offset = dir * base * 0.003;
        return { time: p.date, value: base + offset, direction: p.prediction };
      });
  }, [predictions, historyData]);

  /** Build forecast line + upper / lower bands. */
  const buildForecastOverlay = useCallback(() => {
    if (!forecast.length) return { line: [], upper: [], lower: [] };
    return {
      line: forecast.map((f) => ({ time: f.date, value: f.predicted_price })),
      upper: forecast.map((f) => ({ time: f.date, value: f.upper_bound })),
      lower: forecast.map((f) => ({ time: f.date, value: f.lower_bound })),
    };
  }, [forecast]);

  // -------------------------------------------------------------------
  // Chart lifecycle — create / destroy
  // -------------------------------------------------------------------

  useEffect(() => {
    const container = chartContainerRef.current;
    const chartArea = chartAreaRef.current;
    
    // Validate that we have valid DOM elements
    if (!container || !(container instanceof HTMLElement)) {
      console.warn('Invalid chart container element');
      return () => {}; // Return cleanup function
    }
    
    if (!chartArea || !(chartArea instanceof HTMLElement)) {
      console.warn('Invalid chart area element');
      return () => {}; // Return cleanup function
    }
    
    // Ensure container is attached to DOM
    if (!document.contains(container)) {
      console.warn('Chart container not attached to DOM');
      return () => {}; // Return cleanup function
    }

    const initialWidth = Math.max(chartArea.clientWidth || 0, 100);
    const initialHeight = Math.max(chartArea.clientHeight || 0, 100);
    
    // Validate dimensions
    if (initialWidth <= 0 || initialHeight <= 0) {
      console.warn('Invalid chart dimensions:', { width: initialWidth, height: initialHeight });
      return () => {}; // Return cleanup function
    }

    // Validate that createChart function is available
    if (typeof createChart !== 'function') {
      console.error('TradingView lightweight-charts library not loaded properly');
      return () => {}; // Return cleanup function
    }

    let chart;
    try {
      chart = createChart(container, {
        width: initialWidth,
        height: initialHeight,
        layout: {
          background: { type: ColorType.Solid, color: colors.background },
          textColor: colors.textColor,
        },
        grid: {
          vertLines: { color: colors.gridColor },
          horzLines: { color: colors.gridColor },
        },
        crosshair: {
          mode: CrosshairMode.Normal,
          vertLine: {
            color: colors.crosshairColor,
            width: 1,
            style: 2,
            labelBackgroundColor: colors.borderColor,
          },
          horzLine: {
            color: colors.crosshairColor,
            width: 1,
            style: 2,
            labelBackgroundColor: colors.borderColor,
          },
        },
        rightPriceScale: {
          borderColor: colors.borderColor,
          scaleMargins: { top: 0.1, bottom: 0.25 },
        },
        timeScale: {
          borderColor: colors.borderColor,
          timeVisible: true,
          secondsVisible: false,
        },
        handleScroll: true,
        handleScale: true,
      });
    } catch (error) {
      console.error('Failed to create chart:', error);
      return () => {}; // Return cleanup function
    }

    // Validate chart object before creating series
    if (!chart || typeof chart.addCandlestickSeries !== 'function') {
      console.error('Invalid chart object or missing addCandlestickSeries method');
      try {
        chart?.remove();
      } catch (removeError) {
        console.error('Failed to remove chart:', removeError);
      }
      return () => {}; // Return cleanup function
    }

    // Candlestick series (main price chart)
    let candleSeries;
    try {
      candleSeries = chart.addCandlestickSeries({
        upColor: colors.candleUp,
        downColor: colors.candleDown,
        borderDownColor: colors.candleDown,
        borderUpColor: colors.candleUp,
        wickDownColor: colors.candleDown,
        wickUpColor: colors.candleUp,
      });
    } catch (error) {
      console.error('Failed to create candlestick series:', error);
      try {
        chart.remove();
      } catch (removeError) {
        console.error('Failed to remove chart:', removeError);
      }
      return () => {}; // Return cleanup function
    }

    // Validate histogram series support
    let volumeSeries;
    if (typeof chart.addHistogramSeries !== 'function') {
      console.warn('addHistogramSeries not available, skipping volume series');
      volumeSeries = null;
    } else {
      // Volume histogram (bottom 20 %)
      try {
        volumeSeries = chart.addHistogramSeries({
          priceFormat: { type: 'volume' },
          priceScaleId: 'volume',
        });
        const volumeScale = chart.priceScale('volume');
        if (volumeScale) {
          volumeScale.applyOptions({
            scaleMargins: { top: 0.8, bottom: 0 },
          });
        }
      } catch (error) {
        console.error('Failed to create volume series:', error);
        // Continue without volume series rather than crashing
        volumeSeries = null;
      }
    }

    // Crosshair tooltip
    const crosshairHandler = (param) => {
      try {
        if (!param?.time || !param.seriesData) {
          setCrosshairData(null);
          return;
        }
        const cd = param.seriesData.get(candleSeries);
        const vd = param.seriesData.get(volumeSeries);
        if (cd) {
          setCrosshairData({
            time: param.time,
            open: cd.open,
            high: cd.high,
            low: cd.low,
            close: cd.close,
            volume: vd ? vd.value : null,
          });
        } else {
          setCrosshairData(null);
        }
      } catch (error) {
        console.error('Error in crosshair handler:', error);
        setCrosshairData(null);
      }
    };
    
    if (chart) {
      try {
        chart.subscribeCrosshairMove(crosshairHandler);
      } catch (error) {
        console.error('Error subscribing to crosshair move:', error);
      }
    }

    // Store refs for data-feeding effects
    chartRef.current = chart;
    candleSeriesRef.current = candleSeries;
    // Only set volumeSeriesRef if successfully created
    if (volumeSeries) {
      volumeSeriesRef.current = volumeSeries;
    }
    setChartVersion((v) => v + 1);

    // Responsive resize — observe the chart-area wrapper
    let resizeObserver;
    try {
      resizeObserver = new ResizeObserver((entries) => {
        if (!chart) return;
        for (const entry of entries) {
          const { width, height } = entry.contentRect;
          if (width > 0 && height > 0) {
            try {
              chart.applyOptions({ width, height });
            } catch (e) {
              // Ignore errors when resizing
            }
          }
        }
      });
      if (chartArea) {
        resizeObserver.observe(chartArea);
      }
    } catch (error) {
      console.error('Error setting up resize observer:', error);
      resizeObserver = null;
    }

    let crosshairUnsubscribed = false;
    
    return () => {
      if (resizeObserver) {
        try {
          resizeObserver.disconnect();
        } catch (e) {
          console.warn('Error disconnecting resize observer:', e);
        }
      }
      if (chart && !crosshairUnsubscribed) {
        try {
          chart.unsubscribeCrosshairMove(crosshairHandler);
          crosshairUnsubscribed = true;
        } catch (e) {
          console.warn('Error unsubscribing from crosshair move:', e);
        }
      }
      if (chart) {
        try {
          chart.remove();
        } catch (e) {
          console.warn('Error removing chart:', e);
        }
      }
      chartRef.current = null;
      candleSeriesRef.current = null;
      volumeSeriesRef.current = null;
    };
    // Recreate chart when theme changes so all colours refresh.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isDark]);

  // -------------------------------------------------------------------
  // Feed data into the chart
  // -------------------------------------------------------------------

  // Candlestick OHLC
  useEffect(() => {
    if (!candleSeriesRef.current || !historyData?.length) return;
    try {
      candleSeriesRef.current.setData(
        historyData.map((d) => ({
          time: String(d.Date || d.date),
          open: parseFloat(d.Open || d.open),
          high: parseFloat(d.High || d.high),
          low: parseFloat(d.Low || d.low),
          close: parseFloat(d.Close || d.close),
        })),
      );
    } catch (error) {
      console.error('Error setting candlestick data:', error);
    }
  }, [historyData, chartVersion]);

  // Volume bars
  useEffect(() => {
    if (!volumeSeriesRef.current || !historyData?.length) return;
    try {
      volumeSeriesRef.current.setData(
        historyData.map((d) => {
          const close = parseFloat(d.Close || d.close);
          const open = parseFloat(d.Open || d.open);
          return {
            time: String(d.Date || d.date),
            value: parseFloat(d.Volume || d.volume || 0),
            color: close >= open ? colors.volumeUp : colors.volumeDown,
          };
        }),
      );
    } catch (error) {
      console.error('Error setting volume data:', error);
    }
  }, [historyData, colors.volumeUp, colors.volumeDown, chartVersion]);

  // Prediction overlay (dashed line)
  useEffect(() => {
    if (!chartRef.current) return;
    const data = buildPredictionOverlay();
    if (!data.length) return;

    try {
      const series = chartRef.current.addLineSeries({
        color: colors.predictionLine,
        lineWidth: 1,
        lineStyle: 2,
        priceScaleId: 'right',
        lastValueVisible: false,
        priceLineVisible: false,
      });
      series.setData(data);

      return () => {
        try { 
          if (chartRef.current && series) {
            chartRef.current.removeSeries(series);
          }
        } catch { /* already removed */ }
      };
    } catch (error) {
      console.error('Error creating prediction overlay:', error);
      return () => {};
    }
  }, [buildPredictionOverlay, colors.predictionLine, chartVersion]);

  // Forecast overlay (line + upper/lower bands)
  useEffect(() => {
    if (!chartRef.current) return;
    const { line, upper, lower } = buildForecastOverlay();
    if (!line.length) return;

    try {
      const fLine = chartRef.current.addLineSeries({
        color: colors.forecastLine,
        lineWidth: 2,
        priceScaleId: 'right',
        lastValueVisible: true,
        priceLineVisible: true,
        title: 'Forecast',
      });
      fLine.setData(line);

      const uLine = chartRef.current.addLineSeries({
        color: colors.forecastBand,
        lineWidth: 1,
        lineStyle: 1,
        priceScaleId: 'right',
        lastValueVisible: false,
        priceLineVisible: false,
      });
      uLine.setData(upper);

      const lLine = chartRef.current.addLineSeries({
        color: colors.forecastBand,
        lineWidth: 1,
        lineStyle: 1,
        priceScaleId: 'right',
        lastValueVisible: false,
        priceLineVisible: false,
      });
      lLine.setData(lower);

      return () => {
        try {
          if (chartRef.current) {
            if (fLine) chartRef.current.removeSeries(fLine);
            if (uLine) chartRef.current.removeSeries(uLine);
            if (lLine) chartRef.current.removeSeries(lLine);
          }
        } catch { /* already removed */ }
      };
    } catch (error) {
      console.error('Error creating forecast overlay:', error);
      return () => {};
    }
  }, [buildForecastOverlay, colors.forecastLine, colors.forecastBand, chartVersion]);

  // -------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------

  return (
    <div className="flex flex-col h-full min-h-[400px]">
      {/* ---- Toolbar ---- */}
      <div className="flex flex-wrap items-center gap-2 mb-2 px-1">
        {/* Pair label */}
        <span className="text-sm font-semibold text-gray-700 dark:text-gray-200 bg-gray-100 dark:bg-gray-800 px-3 py-1 rounded">
          {pair || 'Select a pair'}
        </span>

        {/* Timeframe selector */}
        <div className="flex gap-0.5 bg-gray-100 dark:bg-gray-800 rounded overflow-hidden">
          {TIMEFRAMES.map((tf) => (
            <button
              key={tf.apiValue}
              onClick={() => onTimeframeChange?.(tf.apiValue)}
              className={`px-2.5 py-1 text-xs font-medium transition-colors ${
                activeTimeframe === tf.apiValue
                  ? isDark
                    ? 'bg-gray-600 text-white'
                    : 'bg-gray-300 text-gray-900'
                  : isDark
                    ? 'text-gray-400 hover:text-gray-200 hover:bg-gray-700'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-200'
              }`}
            >
              {tf.label}
            </button>
          ))}
        </div>

        {/* Zoom / pan reset */}
        <button
          onClick={() => chartRef.current?.timeScale().fitContent()}
          className="px-2.5 py-1 text-xs font-medium text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200 bg-gray-100 dark:bg-gray-800 rounded transition-colors"
          title="Reset zoom"
        >
          ⟲ Fit
        </button>

        {/* OHLCV readout on crosshair */}
        {crosshairData && (
          <div className="hidden sm:flex items-center gap-3 ml-auto text-xs font-mono text-gray-500 dark:text-gray-400">
            <span>
              O: <span className="text-gray-700 dark:text-gray-300">{crosshairData.open?.toFixed(5)}</span>
            </span>
            <span>
              H: <span className="text-gray-700 dark:text-gray-300">{crosshairData.high?.toFixed(5)}</span>
            </span>
            <span>
              L: <span className="text-gray-700 dark:text-gray-300">{crosshairData.low?.toFixed(5)}</span>
            </span>
            <span>
              C:{' '}
              <span
                className={`font-semibold ${
                  crosshairData.close >= (crosshairData.open ?? 0) ? 'text-green-500' : 'text-red-500'
                }`}
              >
                {crosshairData.close?.toFixed(5)}
              </span>
            </span>
            {crosshairData.volume != null && (
              <span>
                Vol: <span className="text-gray-700 dark:text-gray-300">{(crosshairData.volume / 1e6).toFixed(2)}M</span>
              </span>
            )}
          </div>
        )}
      </div>

      {/* ---- Chart area ---- */}
      <div ref={chartAreaRef} className="relative flex-1 min-h-[350px]">
        <div
          ref={chartContainerRef}
          className="absolute inset-0 rounded-lg overflow-hidden border"
          style={{ borderColor: colors.borderColor }}
        />

        {/* Loading overlay — scoped to chart area */}
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-white/60 dark:bg-gray-900/60 rounded-lg z-10">
            <div className="flex items-center gap-2 text-gray-600 dark:text-gray-300">
              <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              <span className="text-sm">Loading chart data…</span>
            </div>
          </div>
        )}
      </div>

      {/* Error state */}
      {error && (
        <div className="mt-2 p-3 bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 rounded-lg text-sm text-red-600 dark:text-red-400">
          {error}
        </div>
      )}
    </div>
  );
}