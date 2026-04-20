/**
 * WCAG AA Contrast Compliance Checker
 *
 * This script audits all Tailwind CSS color combinations used across the app
 * for WCAG AA compliance. It checks both light and dark mode color pairs.
 *
 * Usage: node scripts/wcag-contrast-check.js
 *
 * Exit codes:
 *   0 - All color pairs pass WCAG AA
 *   1 - One or more color pairs fail WCAG AA
 */

const WCAG_AA_NORMAL_TEXT = 4.5;
const WCAG_AA_LARGE_TEXT = 3.0;
const WCAG_AA_UI_COMPONENTS = 3.0;

// Standard Tailwind colors (hex values)
const colors = {
  // Gray scale
  'gray-50':  '#f9fafb',
  'gray-100': '#f3f4f6',
  'gray-200': '#e5e7eb',
  'gray-300': '#d1d5db',
  'gray-400': '#9ca3af',
  'gray-500': '#6b7280',
  'gray-600': '#4b5563',
  'gray-700': '#374151',
  'gray-800': '#1f2937',
  'gray-900': '#111827',

  // Primary (blue)
  'primary-50':  '#eff6ff',
  'primary-100': '#dbeafe',
  'primary-200': '#bfdbfe',
  'primary-300': '#93c5fd',
  'primary-400': '#60a5fa',
  'primary-500': '#3b82f6',
  'primary-600': '#2563eb',
  'primary-700': '#1d4ed8',
  'primary-800': '#1e40af',
  'primary-900': '#1e3a8a',

  // Green
  'green-50':  '#f0fdf4',
  'green-100': '#dcfce7',
  'green-200': '#bbf7d0',
  'green-300': '#86efac',
  'green-400': '#4ade80',
  'green-500': '#22c55e',
  'green-600': '#16a34a',
  'green-700': '#15803d',
  'green-800': '#166534',
  'green-900': '#14532d',

  // Red
  'red-50':  '#fef2f2',
  'red-100': '#fee2e2',
  'red-200': '#fecaca',
  'red-300': '#fca5a5',
  'red-400': '#f87171',
  'red-500': '#ef4444',
  'red-600': '#dc2626',
  'red-700': '#b91c1c',
  'red-800': '#991b1b',
  'red-900': '#7f1d1d',

  // Blue (info states)
  'blue-50':  '#eff6ff',
  'blue-100': '#dbeafe',
  'blue-200': '#bfdbfe',
  'blue-300': '#93c5fd',
  'blue-400': '#60a5fa',
  'blue-500': '#3b82f6',
  'blue-700': '#1d4ed8',
  'blue-800': '#1e40af',
  'blue-900': '#1e3a8a',

  // Amber (warnings)
  'amber-50':  '#fffbeb',
  'amber-100': '#fef3c7',
  'amber-200': '#fde68a',
  'amber-300': '#fcd34d',
  'amber-400': '#fbbf24',
  'amber-500': '#f59e0b',
  'amber-600': '#d97706',
  'amber-700': '#b45309',
  'amber-800': '#92400e',
  'amber-900': '#78350f',

  // Orange (forecast)
  'orange-100': '#ffedd5',
  'orange-200': '#fed7aa',
  'orange-500': '#f97316',
  'orange-700': '#c2410c',
  'orange-800': '#9a3412',
  'orange-900': '#7c2d12',

  // Yellow (confidence bars)
  'yellow-500': '#eab308',

  // White
  'white': '#ffffff',
};

// Helper: parse hex to RGB
function hexToRgb(hex) {
  const cleaned = hex.replace('#', '');
  return {
    r: parseInt(cleaned.substring(0, 2), 16),
    g: parseInt(cleaned.substring(2, 4), 16),
    b: parseInt(cleaned.substring(4, 6), 16),
  };
}

// Helper: calculate relative luminance (WCAG 2.1 formula)
function relativeLuminance(hex) {
  const { r, g, b } = hexToRgb(hex);
  const [rs, gs, bs] = [r, g, b].map((c) => {
    const s = c / 255;
    return s <= 0.03928 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
  });
  return 0.2126 * rs + 0.7152 * gs + 0.0722 * bs;
}

// Helper: calculate contrast ratio between two hex colors
function contrastRatio(hex1, hex2) {
  const l1 = relativeLuminance(hex1);
  const l2 = relativeLuminance(hex2);
  const lighter = Math.max(l1, l2);
  const darker = Math.min(l1, l2);
  return (lighter + 0.05) / (darker + 0.05);
}

// Define all color pairs used in the application
function getLightModePairs() {
  return [
    // === Main body text ===
    { desc: 'Body text (gray-900 on white)', fg: 'gray-900', bg: 'white', context: 'normal-text' },
    { desc: 'Headings (gray-900 on white)', fg: 'gray-900', bg: 'white', context: 'large-text' },
    { desc: 'Secondary text (gray-500 on white)', fg: 'gray-500', bg: 'white', context: 'normal-text' },

    // === Card backgrounds ===
    { desc: 'Card text (gray-900 on gray-50)', fg: 'gray-900', bg: 'gray-50', context: 'normal-text' },
    { desc: 'Card secondary text (gray-500 on gray-50)', fg: 'gray-500', bg: 'gray-50', context: 'normal-text' },
    { desc: 'Card secondary text (gray-700 on gray-50)', fg: 'gray-700', bg: 'gray-50', context: 'normal-text' },

    // === Navigation ===
    { desc: 'Active nav link (primary-700 on primary-50)', fg: 'primary-700', bg: 'primary-50', context: 'normal-text' },
    { desc: 'Inactive nav link (gray-600 on white)', fg: 'gray-600', bg: 'white', context: 'normal-text' },
    { desc: 'Brand name (gray-900 on white)', fg: 'gray-900', bg: 'white', context: 'large-text' },
    { desc: 'Sidebar theme label (gray-500 on white)', fg: 'gray-500', bg: 'white', context: 'normal-text' },

    // === Theme toggle ===
    { desc: 'Theme toggle icon (gray-600 on white)', fg: 'gray-600', bg: 'white', context: 'ui-component' },
    { desc: 'Theme toggle border (gray-200 on white)', fg: 'gray-200', bg: 'white', context: 'ui-component' },

    // === Green status badges ===
    { desc: 'Green badge text (green-700 on green-50)', fg: 'green-700', bg: 'green-50', context: 'normal-text' },
    { desc: 'Green dot (green-500 on green-50)', fg: 'green-500', bg: 'green-50', context: 'ui-component' },

    // === Red status badges ===
    { desc: 'Red badge text (red-700 on red-50)', fg: 'red-700', bg: 'red-50', context: 'normal-text' },
    { desc: 'Red dot (red-500 on red-50)', fg: 'red-500', bg: 'red-50', context: 'ui-component' },

    // === No model badge ===
    { desc: 'No model badge (gray-600 on gray-50)', fg: 'gray-600', bg: 'gray-50', context: 'normal-text' },

    // === Error messages ===
    { desc: 'Error title (red-800 on red-50)', fg: 'red-800', bg: 'red-50', context: 'normal-text' },
    { desc: 'Error message (red-700 on red-50)', fg: 'red-700', bg: 'red-50', context: 'normal-text' },
    { desc: 'Error icon (red-500 on red-50)', fg: 'red-500', bg: 'red-50', context: 'ui-component' },
    { desc: 'Error retry link (red-600 on red-50)', fg: 'red-600', bg: 'red-50', context: 'normal-text' },

    // === Info loading banners ===
    { desc: 'Info banner text (blue-700 on blue-50)', fg: 'blue-700', bg: 'blue-50', context: 'normal-text' },

    // === Training complete ===
    { desc: 'Training complete header (green-800 on green-50)', fg: 'green-800', bg: 'green-50', context: 'large-text' },
    { desc: 'Status badge (green-700 on green-50)', fg: 'green-700', bg: 'green-50', context: 'normal-text' },

    // === Table headers ===
    { desc: 'Table header text (gray-500 on gray-50)', fg: 'gray-500', bg: 'gray-50', context: 'normal-text' },
    { desc: 'Table cell text (gray-900 on white)', fg: 'gray-900', bg: 'white', context: 'normal-text' },
    { desc: 'Table cell mono (gray-900 on white)', fg: 'gray-900', bg: 'white', context: 'normal-text' },

    // === Prediction badges ===
    { desc: 'UP badge (green-700 on green-50)', fg: 'green-700', bg: 'green-50', context: 'normal-text' },
    { desc: 'DOWN badge (red-700 on red-50)', fg: 'red-700', bg: 'red-50', context: 'normal-text' },
    { desc: 'UP dot (green-500 on green-50)', fg: 'green-500', bg: 'green-50', context: 'ui-component' },
    { desc: 'DOWN dot (red-500 on red-50)', fg: 'red-500', bg: 'red-50', context: 'ui-component' },

    // === Summary cards ===
    { desc: 'Summary card label (primary-700 on primary-50)', fg: 'primary-700', bg: 'primary-50', context: 'normal-text' },
    { desc: 'Summary card value (primary-700 on primary-50)', fg: 'primary-700', bg: 'primary-50', context: 'large-text' },
    { desc: 'Summary card label (green-700 on green-50)', fg: 'green-700', bg: 'green-50', context: 'normal-text' },
    { desc: 'Summary card value (green-700 on green-50)', fg: 'green-700', bg: 'green-50', context: 'large-text' },
    { desc: 'Summary card label (red-700 on red-50)', fg: 'red-700', bg: 'red-50', context: 'normal-text' },
    { desc: 'Summary card value (red-700 on red-50)', fg: 'red-700', bg: 'red-50', context: 'large-text' },

    // === Trade setup direction banner ===
    { desc: 'Direction title (gray-900 on green-50)', fg: 'gray-900', bg: 'green-50', context: 'large-text' },
    { desc: 'Direction subtitle (gray-500 on green-50)', fg: 'gray-500', bg: 'green-50', context: 'normal-text' },
    { desc: 'Confidence label (gray-500 on green-50)', fg: 'gray-500', bg: 'green-50', context: 'normal-text' },
    { desc: 'Confidence value (gray-900 on green-50)', fg: 'gray-900', bg: 'green-50', context: 'large-text' },
    { desc: 'Direction icon (green-600 on green-100)', fg: 'green-600', bg: 'green-100', context: 'ui-component' },

    // === Take Profit card ===
    { desc: 'TP label (green-700 on green-50)', fg: 'green-700', bg: 'green-50', context: 'normal-text' },
    { desc: 'TP value (green-700 on green-50)', fg: 'green-700', bg: 'green-50', context: 'large-text' },
    { desc: 'TP distance (green-600 on green-50)', fg: 'green-600', bg: 'green-50', context: 'normal-text' },
    { desc: 'TP icon (green-600 on green-100)', fg: 'green-600', bg: 'green-100', context: 'ui-component' },

    // === Stop Loss card ===
    { desc: 'SL label (red-700 on red-50)', fg: 'red-700', bg: 'red-50', context: 'normal-text' },
    { desc: 'SL value (red-700 on red-50)', fg: 'red-700', bg: 'red-50', context: 'large-text' },
    { desc: 'SL distance (red-600 on red-50)', fg: 'red-600', bg: 'red-50', context: 'normal-text' },
    { desc: 'SL icon (red-600 on red-100)', fg: 'red-600', bg: 'red-100', context: 'ui-component' },

    // === Risk/Reward & Model details ===
    { desc: 'RR label (gray-500 on white)', fg: 'gray-500', bg: 'white', context: 'normal-text' },
    { desc: 'RR value (gray-900 on white)', fg: 'gray-900', bg: 'white', context: 'normal-text' },
    { desc: 'Model detail label (gray-500 on white)', fg: 'gray-500', bg: 'white', context: 'normal-text' },
    { desc: 'Model detail value (gray-900 on white)', fg: 'gray-900', bg: 'white', context: 'normal-text' },

    // === Disclaimer/Warning ===
    { desc: 'Disclaimer text (amber-800 on amber-50)', fg: 'amber-800', bg: 'amber-50', context: 'normal-text' },
    { desc: 'Warning icon (amber-600 on amber-50)', fg: 'amber-600', bg: 'amber-50', context: 'ui-component' },

    // === Code blocks ===
    { desc: 'Code text (gray-800 on gray-100)', fg: 'gray-800', bg: 'gray-100', context: 'normal-text' },

    // === Border contrast ===
    { desc: 'Card border (gray-200 on white)', fg: 'gray-200', bg: 'white', context: 'ui-component' },
    { desc: 'Input border (gray-300 on white)', fg: 'gray-300', bg: 'white', context: 'ui-component' },
    { desc: 'Confidence bar track (gray-300 on white)', fg: 'gray-300', bg: 'white', context: 'ui-component' },

    // === Empty state ===
    { desc: 'Empty state icon (gray-500 on white)', fg: 'gray-500', bg: 'white', context: 'ui-component' },
    { desc: 'Empty state heading (gray-900 on white)', fg: 'gray-900', bg: 'white', context: 'large-text' },
    { desc: 'Empty state text (gray-500 on white)', fg: 'gray-500', bg: 'white', context: 'normal-text' },

    // === Select inputs ===
    { desc: 'Select text (gray-900 on white)', fg: 'gray-900', bg: 'white', context: 'normal-text' },

    // === Status bar ===
    { desc: 'Status bar text (gray-500 on gray-50)', fg: 'gray-500', bg: 'gray-50', context: 'normal-text' },

    // === Filter labels ===
    { desc: 'Filter label (gray-500 on white)', fg: 'gray-500', bg: 'white', context: 'normal-text' },

    // === Confidence bar text ===
    { desc: 'Confidence text (gray-500 on white)', fg: 'gray-500', bg: 'white', context: 'normal-text' },

    // === Forex pair cards ===
    { desc: 'Pair name (gray-900 on white)', fg: 'gray-900', bg: 'white', context: 'large-text' },
    { desc: 'Ticker label (gray-700 on white)', fg: 'gray-700', bg: 'white', context: 'normal-text' },
    { desc: 'Ticker value (gray-500 on white)', fg: 'gray-500', bg: 'white', context: 'normal-text' },

    // === Primary buttons ===
    { desc: 'Button text (white on primary-600)', fg: 'white', bg: 'primary-600', context: 'normal-text' },

    // === Secondary links ===
    { desc: 'Predict link (gray-700 on white)', fg: 'gray-700', bg: 'white', context: 'normal-text' },
    { desc: 'Predict link border (gray-300 on white)', fg: 'gray-300', bg: 'white', context: 'ui-component' },

    // === Stale badge ===
    { desc: 'Stale badge (amber-700 on amber-100)', fg: 'amber-700', bg: 'amber-100', context: 'normal-text' },

    // === Timeframe badge ===
    { desc: 'Timeframe badge (gray-600 on gray-100)', fg: 'gray-600', bg: 'gray-100', context: 'normal-text' },

    // === R:R info text ===
    { desc: 'RR info text (gray-500 on white)', fg: 'gray-500', bg: 'white', context: 'normal-text' },
    { desc: 'RR reason (gray-500 on white)', fg: 'gray-500', bg: 'white', context: 'normal-text' },

    // === Chart legend ===
    { desc: 'Chart legend text (gray-500 on white)', fg: 'gray-500', bg: 'white', context: 'normal-text' },
    { desc: 'Chart historical line (blue-600 on white)', fg: 'blue-500', bg: 'white', context: 'ui-component' },
    { desc: 'Chart forecast line (orange-500 on white)', fg: 'orange-500', bg: 'white', context: 'ui-component' },
  ];
}

function getDarkModePairs() {
  return [
    // === Main body text ===
    { desc: 'Body text (gray-100 on gray-900)', fg: 'gray-100', bg: 'gray-900', context: 'normal-text' },
    { desc: 'Headings (gray-100 on gray-900)', fg: 'gray-100', bg: 'gray-900', context: 'large-text' },
    { desc: 'Secondary text (gray-400 on gray-900)', fg: 'gray-400', bg: 'gray-900', context: 'normal-text' },

    // === Card backgrounds ===
    { desc: 'Card text (gray-100 on gray-800)', fg: 'gray-100', bg: 'gray-800', context: 'normal-text' },
    { desc: 'Card secondary text (gray-400 on gray-800)', fg: 'gray-400', bg: 'gray-800', context: 'normal-text' },
    { desc: 'Card label text (gray-300 on gray-800)', fg: 'gray-300', bg: 'gray-800', context: 'normal-text' },
    { desc: 'Card subtle text (gray-400 on gray-800)', fg: 'gray-400', bg: 'gray-800', context: 'normal-text' },

    // === Navigation ===
    { desc: 'Active nav link (primary-400 on gray-800)', fg: 'primary-400', bg: 'gray-800', context: 'normal-text' },
    { desc: 'Inactive nav link (gray-300 on gray-800)', fg: 'gray-300', bg: 'gray-800', context: 'normal-text' },
    { desc: 'Brand name (gray-100 on gray-800)', fg: 'gray-100', bg: 'gray-800', context: 'large-text' },
    { desc: 'Sidebar theme label (gray-400 on gray-800)', fg: 'gray-400', bg: 'gray-800', context: 'normal-text' },

    // === Theme toggle ===
    { desc: 'Theme toggle icon (gray-300 on gray-800)', fg: 'gray-300', bg: 'gray-800', context: 'ui-component' },
    { desc: 'Theme toggle border (gray-700 on gray-800)', fg: 'gray-700', bg: 'gray-800', context: 'ui-component' },

    // === Green status badges (dark) ===
    { desc: 'Green badge text (green-400 on gray-800)', fg: 'green-400', bg: 'gray-800', context: 'normal-text' },
    { desc: 'Green dot (green-400 on gray-800)', fg: 'green-400', bg: 'gray-800', context: 'ui-component' },

    // === Red status badges (dark) ===
    { desc: 'Red badge text (red-400 on gray-800)', fg: 'red-400', bg: 'gray-800', context: 'normal-text' },
    { desc: 'Red dot (red-400 on gray-800)', fg: 'red-400', bg: 'gray-800', context: 'ui-component' },

    // === No model badge (dark) ===
    { desc: 'No model badge (gray-300 on gray-700)', fg: 'gray-300', bg: 'gray-700', context: 'normal-text' },

    // === Error messages (dark) ===
    { desc: 'Error title (red-300 on gray-800)', fg: 'red-300', bg: 'gray-800', context: 'normal-text' },
    { desc: 'Error message (red-400 on gray-800)', fg: 'red-400', bg: 'gray-800', context: 'normal-text' },
    { desc: 'Error icon (red-400 on gray-800)', fg: 'red-400', bg: 'gray-800', context: 'ui-component' },
    { desc: 'Error retry link (red-400 on gray-800)', fg: 'red-400', bg: 'gray-800', context: 'normal-text' },

    // === Info loading banners (dark) ===
    { desc: 'Info banner text (blue-200 on gray-800)', fg: 'blue-200', bg: 'gray-800', context: 'normal-text' },

    // === Training complete (dark) ===
    { desc: 'Training complete header (green-400 on gray-800)', fg: 'green-400', bg: 'gray-800', context: 'large-text' },

    // === Table headers (dark) ===
    { desc: 'Table header text (gray-400 on gray-800)', fg: 'gray-400', bg: 'gray-800', context: 'normal-text' },
    { desc: 'Table cell text (gray-100 on gray-800)', fg: 'gray-100', bg: 'gray-800', context: 'normal-text' },

    // === Direction banner (dark) ===
    { desc: 'Direction title (gray-100 on gray-800)', fg: 'gray-100', bg: 'gray-800', context: 'large-text' },
    { desc: 'Direction subtitle (gray-400 on gray-800)', fg: 'gray-400', bg: 'gray-800', context: 'normal-text' },
    { desc: 'Direction icon (green-400 on green-900/40 → gray-800)', fg: 'green-400', bg: 'gray-800', context: 'ui-component' },

    // === TP card (dark) ===
    { desc: 'TP label (green-400 on gray-800)', fg: 'green-400', bg: 'gray-800', context: 'normal-text' },
    { desc: 'TP value (green-400 on gray-800)', fg: 'green-400', bg: 'gray-800', context: 'large-text' },
    { desc: 'TP distance (green-500 on gray-800)', fg: 'green-500', bg: 'gray-800', context: 'normal-text' },
    { desc: 'TP icon (green-400 on green-900/40 → gray-800)', fg: 'green-400', bg: 'gray-800', context: 'ui-component' },

    // === SL card (dark) ===
    { desc: 'SL label (red-400 on gray-800)', fg: 'red-400', bg: 'gray-800', context: 'normal-text' },
    { desc: 'SL value (red-400 on gray-800)', fg: 'red-400', bg: 'gray-800', context: 'large-text' },
    { desc: 'SL distance (red-500 on gray-800)', fg: 'red-500', bg: 'gray-800', context: 'normal-text' },
    { desc: 'SL icon (red-400 on red-900/40 → gray-800)', fg: 'red-400', bg: 'gray-800', context: 'ui-component' },

    // === SL on plain cards (dark) - LiveSetups ===
    { desc: 'SL mono text (red-300 on gray-800)', fg: 'red-300', bg: 'gray-800', context: 'normal-text' },

    // === Forecast table (dark) ===
    { desc: 'Forecast lower bound (red-300 on gray-800)', fg: 'red-300', bg: 'gray-800', context: 'normal-text' },
    { desc: 'Forecast upper bound (green-400 on gray-800)', fg: 'green-400', bg: 'gray-800', context: 'normal-text' },

    // === Risk/Reward (dark) ===
    { desc: 'RR label (gray-400 on gray-800)', fg: 'gray-400', bg: 'gray-800', context: 'normal-text' },
    { desc: 'RR value (gray-100 on gray-800)', fg: 'gray-100', bg: 'gray-800', context: 'normal-text' },
    { desc: 'Model detail label (gray-400 on gray-800)', fg: 'gray-400', bg: 'gray-800', context: 'normal-text' },
    { desc: 'Model detail value (gray-100 on gray-800)', fg: 'gray-100', bg: 'gray-800', context: 'normal-text' },

    // === Disclaimer (dark) ===
    { desc: 'Disclaimer text (amber-300 on gray-900)', fg: 'amber-300', bg: 'gray-900', context: 'normal-text' },
    { desc: 'Warning icon (amber-400 on gray-900)', fg: 'amber-400', bg: 'gray-900', context: 'ui-component' },

    // === Code blocks (dark) ===
    { desc: 'Code text (gray-200 on gray-700)', fg: 'gray-200', bg: 'gray-700', context: 'normal-text' },

    // === Border contrast (dark) ===
    { desc: 'Card border (gray-700 on gray-800)', fg: 'gray-700', bg: 'gray-800', context: 'ui-component' },
    { desc: 'Input border (gray-600 on gray-700)', fg: 'gray-600', bg: 'gray-700', context: 'ui-component' },
    { desc: 'Confidence bar track (gray-600 on gray-800)', fg: 'gray-600', bg: 'gray-800', context: 'ui-component' },

    // === Empty state (dark) ===
    { desc: 'Empty state icon (gray-400 on gray-800)', fg: 'gray-400', bg: 'gray-800', context: 'ui-component' },
    { desc: 'Empty state heading (gray-100 on gray-800)', fg: 'gray-100', bg: 'gray-800', context: 'large-text' },
    { desc: 'Empty state text (gray-400 on gray-800)', fg: 'gray-400', bg: 'gray-800', context: 'normal-text' },

    // === Select inputs (dark) ===
    { desc: 'Select text (gray-100 on gray-700)', fg: 'gray-100', bg: 'gray-700', context: 'normal-text' },

    // === Status bar (dark) ===
    { desc: 'Status bar text (gray-400 on gray-900)', fg: 'gray-400', bg: 'gray-900', context: 'normal-text' },

    // === Filter labels (dark) ===
    { desc: 'Filter label (gray-400 on gray-800)', fg: 'gray-400', bg: 'gray-800', context: 'normal-text' },
    { desc: 'Filter controls text (gray-300 on gray-800)', fg: 'gray-300', bg: 'gray-800', context: 'normal-text' },

    // === Primary buttons (dark) ===
    { desc: 'Button text (white on primary-600)', fg: 'white', bg: 'primary-600', context: 'normal-text' },

    // === Secondary links (dark) ===
    { desc: 'Predict link (gray-300 on gray-800)', fg: 'gray-300', bg: 'gray-800', context: 'normal-text' },
    { desc: 'Predict link border (gray-600 on gray-800)', fg: 'gray-600', bg: 'gray-800', context: 'ui-component' },

    // === Stale badge (dark) ===
    { desc: 'Stale badge (amber-400 on gray-800)', fg: 'amber-400', bg: 'gray-800', context: 'normal-text' },

    // === Timeframe badge (dark) ===
    { desc: 'Timeframe badge (gray-300 on gray-700)', fg: 'gray-300', bg: 'gray-700', context: 'normal-text' },

    // === Confidence bar text (dark) ===
    { desc: 'Confidence text (gray-400 on gray-800)', fg: 'gray-400', bg: 'gray-800', context: 'normal-text' },

    // === Number inputs (dark) ===
    { desc: 'Number input text (gray-100 on gray-700)', fg: 'gray-100', bg: 'gray-700', context: 'normal-text' },

    // === Chart legend (dark) ===
    { desc: 'Chart legend text (gray-400 on gray-800)', fg: 'gray-400', bg: 'gray-800', context: 'normal-text' },

    // === Forecast summary cards (dark) ===
    { desc: 'Summary card label (gray-400 on gray-800)', fg: 'gray-400', bg: 'gray-800', context: 'normal-text' },
    { desc: 'Summary card value (gray-100 on gray-800)', fg: 'gray-100', bg: 'gray-800', context: 'large-text' },

    // === Direction badge (dark) ===
    { desc: 'Direction badge (green-400 on gray-800)', fg: 'green-400', bg: 'gray-800', context: 'normal-text' },
    { desc: 'Direction badge (red-400 on gray-800)', fg: 'red-400', bg: 'gray-800', context: 'normal-text' },

    // === Model info (dark) ===
    { desc: 'Model info label (gray-400 on gray-800)', fg: 'gray-400', bg: 'gray-800', context: 'normal-text' },
    { desc: 'Model info value (gray-100 on gray-800)', fg: 'gray-100', bg: 'gray-800', context: 'normal-text' },

    // === Green pulse indicator ===
    { desc: 'Green pulse dot (green-500 on gray-900)', fg: 'green-500', bg: 'gray-900', context: 'ui-component' },

    // === Checkbox labels (dark) ===
    { desc: 'Checkbox label (gray-300 on gray-800)', fg: 'gray-300', bg: 'gray-800', context: 'normal-text' },

    // === R:R info (dark) ===
    { desc: 'RR info text (gray-400 on gray-800)', fg: 'gray-400', bg: 'gray-800', context: 'normal-text' },

    // === Table cell mono (dark) ===
    { desc: 'Table cell mono (gray-100 on gray-800)', fg: 'gray-100', bg: 'gray-800', context: 'normal-text' },
  ];
}

function getRequiredRatio(context) {
  switch (context) {
    case 'large-text':
      return WCAG_AA_LARGE_TEXT;
    case 'ui-component':
      return WCAG_AA_UI_COMPONENTS;
    default:
      return WCAG_AA_NORMAL_TEXT;
  }
}

function runAudit(pairs, mode) {
  console.log(`\n=== ${mode} Mode Audit ===\n`);
  const failures = [];
  let totalChecks = 0;

  for (const pair of pairs) {
    const fgHex = colors[pair.fg];
    const bgHex = colors[pair.bg];
    if (!fgHex || !bgHex) {
      console.log(`\u26A0 SKIP: Unknown color in "${pair.desc}"`);
      continue;
    }

    const ratio = contrastRatio(fgHex, bgHex);
    const required = getRequiredRatio(pair.context);
    totalChecks++;

    const status = ratio >= required ? '\u2705 PASS' : '\u274C FAIL';
    console.log(
      `${status} | ${pair.fg} on ${pair.bg} | ${ratio.toFixed(2)}:1 (need ${required}:1) | ${pair.context} | ${pair.desc}`
    );

    if (ratio < required) {
      failures.push({
        mode,
        desc: pair.desc,
        fg: pair.fg,
        bg: pair.bg,
        ratio,
        required,
        context: pair.context,
      });
    }
  }

  return { failures, totalChecks };
}

// Main
console.log('WCAG AA Contrast Compliance Audit');
console.log('===================================');
console.log(`Normal text minimum: ${WCAG_AA_NORMAL_TEXT}:1`);
console.log(`Large text minimum:  ${WCAG_AA_LARGE_TEXT}:1`);
console.log(`UI components min:   ${WCAG_AA_UI_COMPONENTS}:1`);

const lightResult = runAudit(getLightModePairs(), 'Light');
const darkResult = runAudit(getDarkModePairs(), 'Dark');

const allFailures = [...lightResult.failures, ...darkResult.failures];
const totalChecks = lightResult.totalChecks + darkResult.totalChecks;

console.log('\n=== Summary ===');
console.log(`Total checks: ${totalChecks}`);
console.log(`Passed: ${totalChecks - allFailures.length}`);
console.log(`Failed: ${allFailures.length}`);

if (allFailures.length > 0) {
  console.log('\n=== Failures ===');
  for (const f of allFailures) {
    console.log(
      `[${f.mode}] ${f.desc} — ${f.ratio.toFixed(2)}:1 (need ${f.required}:1, context: ${f.context})`
    );
  }
  console.log('\n\u274C WCAG AA audit FAILED — fix the contrast issues above.');
  process.exit(1);
} else {
  console.log('\n\u2705 All color pairs pass WCAG AA contrast requirements.');
  process.exit(0);
}
