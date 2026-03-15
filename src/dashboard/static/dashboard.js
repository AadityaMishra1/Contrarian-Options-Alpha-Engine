/**
 * COE Trading Dashboard — WebSocket client + Chart.js initialisation.
 * Connects via Socket.IO, handles real-time updates, and seeds initial state
 * from the REST endpoints on page load.
 *
 * All DOM construction uses createElement + textContent to avoid XSS risks
 * even though data originates from the local Python backend.
 */

'use strict';

// ── Socket.IO ────────────────────────────────────────────────────────────────

const socket = io({ reconnectionDelay: 1000, reconnectionDelayMax: 5000 });

const statusDot = document.getElementById('statusDot');

socket.on('connect',    () => statusDot.classList.add('connected'));
socket.on('disconnect', () => statusDot.classList.remove('connected'));

// ── Chart instances ──────────────────────────────────────────────────────────

/** @type {import('chart.js').Chart} */
let pnlChart;

/** @type {import('chart.js').Chart} */
let equityChart;

const CHART_DEFAULTS = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 250 },
    plugins: { legend: { display: false }, tooltip: { mode: 'index', intersect: false } },
};

function initCharts() {
    const pnlCtx = document.getElementById('pnlChart').getContext('2d');
    pnlChart = new Chart(pnlCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'P&L',
                data: [],
                borderColor: '#3fb950',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.3,
                fill: false,
            }],
        },
        options: {
            ...CHART_DEFAULTS,
            scales: {
                x: { display: false },
                y: {
                    display: true,
                    grid: { color: 'rgba(48,54,61,0.6)' },
                    ticks: { color: '#8b949e', font: { size: 10 }, callback: v => '$' + v.toFixed(0) },
                    border: { display: false },
                },
            },
        },
    });

    const equityCtx = document.getElementById('equityChart').getContext('2d');
    const gradient  = equityCtx.createLinearGradient(0, 0, 0, 300);
    gradient.addColorStop(0, 'rgba(88,166,255,0.3)');
    gradient.addColorStop(1, 'rgba(88,166,255,0.0)');

    equityChart = new Chart(equityCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Equity',
                data: [],
                borderColor: '#58a6ff',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.3,
                fill: true,
                backgroundColor: gradient,
            }],
        },
        options: {
            ...CHART_DEFAULTS,
            scales: {
                x: { display: false },
                y: {
                    display: true,
                    grid: { color: 'rgba(48,54,61,0.6)' },
                    ticks: { color: '#8b949e', font: { size: 10 }, callback: v => '$' + v.toFixed(0) },
                    border: { display: false },
                },
            },
        },
    });
}

// ── Utilities ────────────────────────────────────────────────────────────────

/**
 * Format a numeric value as a signed currency string.
 * @param {number|null|undefined} val
 * @returns {string}
 */
function formatCurrency(val) {
    if (val === null || val === undefined || isNaN(val)) return '\u2014';
    const abs = Math.abs(val).toFixed(2);
    return val >= 0 ? '+$' + abs : '-$' + abs;
}

/**
 * Format a timestamp (ms epoch or ISO string) as local HH:MM:SS.
 * @param {number|string} ts
 * @returns {string}
 */
function formatTime(ts) {
    if (!ts) return '\u2014';
    return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

/** Return the appropriate CSS class for a signed numeric value. */
function signClass(val) {
    return val > 0 ? 'pos' : val < 0 ? 'neg' : 'neutral';
}

/**
 * Create a <td> element with optional CSS class and text.
 * @param {string} text
 * @param {string} [className]
 * @returns {HTMLTableCellElement}
 */
function td(text, className) {
    const cell = document.createElement('td');
    cell.textContent = text;
    if (className) cell.className = className;
    return cell;
}

// ── Positions table ──────────────────────────────────────────────────────────

/**
 * Rebuild the positions table from a full positions array.
 * @param {Array<Object>} positions
 */
function updatePositionsTable(positions) {
    const tbody = document.getElementById('positionsBody');
    const badge = document.getElementById('pos-count-badge');
    badge.textContent = String(positions.length);

    while (tbody.firstChild) tbody.removeChild(tbody.firstChild);

    if (!positions.length) {
        const row = tbody.insertRow();
        const cell = row.insertCell();
        cell.colSpan = 6;
        cell.className = 'neutral';
        cell.style.textAlign = 'center';
        cell.style.padding = '20px';
        cell.textContent = 'No open positions';
        return;
    }

    positions.forEach(p => {
        const row = tbody.insertRow();
        const side   = (p.side || '').toUpperCase();
        const unreal = p.unrealized_pnl ?? 0;
        const entry  = p.entry_price   != null ? '$' + Number(p.entry_price).toFixed(2)   : '\u2014';
        const curr   = p.current_price != null ? '$' + Number(p.current_price).toFixed(2) : '\u2014';

        row.appendChild(td(p.symbol || '\u2014'));
        row.appendChild(td(side, side === 'LONG' ? 'side-long' : 'side-short'));
        row.appendChild(td(p.qty != null ? String(p.qty) : '\u2014'));
        row.appendChild(td(entry));
        row.appendChild(td(curr));
        row.appendChild(td(formatCurrency(unreal), signClass(unreal)));
    });
}

// ── Trades table ─────────────────────────────────────────────────────────────

const MAX_TRADE_ROWS = 50;

/**
 * Prepend a single trade row to the trades table (newest on top).
 * @param {Object} trade
 */
function prependTradeRow(trade) {
    const tbody = document.getElementById('tradesBody');

    // Remove empty-state placeholder if present
    const firstRow = tbody.rows[0];
    if (firstRow) {
        const firstCell = firstRow.cells[0];
        if (firstCell && firstCell.colSpan > 1) tbody.deleteRow(0);
    }

    const pnl  = trade.pnl ?? trade.realized_pnl ?? null;
    const side = (trade.side || '').toUpperCase();
    const price = trade.price != null ? '$' + Number(trade.price).toFixed(2) : '\u2014';

    const row = document.createElement('tr');
    row.appendChild(td(formatTime(trade.timestamp)));
    row.appendChild(td(trade.symbol || '\u2014'));
    row.appendChild(td(side, side === 'LONG' || side === 'BUY' ? 'side-long' : 'side-short'));
    row.appendChild(td(trade.qty != null ? String(trade.qty) : '\u2014'));
    row.appendChild(td(price));
    row.appendChild(td(pnl != null ? formatCurrency(pnl) : '\u2014', pnl != null ? signClass(pnl) : 'neutral'));

    tbody.prepend(row);

    while (tbody.rows.length > MAX_TRADE_ROWS) tbody.deleteRow(tbody.rows.length - 1);
}

/**
 * Seed the trades table from the initial REST payload.
 * @param {Array<Object>} trades
 */
function populateTradesTable(trades) {
    if (!trades.length) return;
    trades.slice(-MAX_TRADE_ROWS).forEach(prependTradeRow);
}

// ── Equity chart ─────────────────────────────────────────────────────────────

/**
 * Append a single equity point to the equity chart.
 * @param {{ timestamp: number|string, equity: number }} point
 */
function appendEquityPoint(point) {
    const ds = equityChart.data.datasets[0];
    equityChart.data.labels.push(formatTime(point.timestamp));
    ds.data.push(point.equity);

    const badge = document.getElementById('equity-badge');
    badge.textContent = point.equity != null ? '$' + Number(point.equity).toFixed(2) : '\u2014';

    if (ds.data.length > 500) {
        equityChart.data.labels.shift();
        ds.data.shift();
    }
    equityChart.update('none');
}

/**
 * Seed the equity chart from the initial REST payload.
 * @param {Array<Object>} points
 */
function populateEquityChart(points) {
    points.forEach(appendEquityPoint);
}

// ── P&L sparkline ────────────────────────────────────────────────────────────

/**
 * Update the P&L sparkline with the latest daily P&L value.
 * @param {number} tradeCount
 * @param {number} dailyPnl
 */
function updatePnlChart(tradeCount, dailyPnl) {
    const ds = pnlChart.data.datasets[0];
    ds.borderColor = dailyPnl >= 0 ? '#3fb950' : '#f85149';

    pnlChart.data.labels.push(String(tradeCount));
    ds.data.push(dailyPnl);

    if (ds.data.length > 200) {
        pnlChart.data.labels.shift();
        ds.data.shift();
    }
    pnlChart.update('none');

    document.getElementById('pnl-badge').textContent =
        tradeCount + ' trade' + (tradeCount !== 1 ? 's' : '');
}

// ── Signals log ──────────────────────────────────────────────────────────────

const MAX_SIGNAL_ENTRIES = 100;

/**
 * Prepend a signal entry to the signals feed (newest on top).
 * @param {Object} signal
 */
function prependSignalEntry(signal) {
    const list = document.getElementById('signalsList');
    const score = signal.score != null ? Number(signal.score).toFixed(3) : '\u2014';
    const scoreClass = signal.score >= 0.7 ? 'pos' : signal.score >= 0.4 ? 'neutral' : 'neg';

    const div = document.createElement('div');
    div.className = 'signal-entry';

    const timeSpan = document.createElement('span');
    timeSpan.className = 'signal-time';
    timeSpan.textContent = formatTime(signal.timestamp);

    const typeSpan = document.createElement('span');
    typeSpan.className = 'signal-type';
    typeSpan.textContent = signal.type || '\u2014';

    const symSpan = document.createElement('span');
    symSpan.className = 'signal-symbol';
    symSpan.textContent = signal.symbol || '\u2014';

    const scoreSpan = document.createElement('span');
    scoreSpan.className = 'signal-score ' + scoreClass;
    scoreSpan.textContent = score;

    div.appendChild(timeSpan);
    div.appendChild(typeSpan);
    div.appendChild(symSpan);
    div.appendChild(scoreSpan);

    list.prepend(div);

    while (list.children.length > MAX_SIGNAL_ENTRIES) list.removeChild(list.lastChild);
}

/**
 * Seed the signals log from the initial REST payload.
 * @param {Array<Object>} signals
 */
function populateSignalsLog(signals) {
    signals.slice(-MAX_SIGNAL_ENTRIES).forEach(prependSignalEntry);
}

// ── Metrics display ──────────────────────────────────────────────────────────

/**
 * Update all header metric pills from a metrics object.
 * @param {Object} metrics
 */
function updateMetricsDisplay(metrics) {
    const dailyPnl = metrics.daily_pnl       ?? 0;
    const totalPnl = metrics.total_pnl       ?? 0;
    const winRate  = metrics.win_rate         ?? 0;
    const trades   = metrics.trade_count      ?? 0;
    const openPos  = metrics.open_positions   ?? 0;

    const dailyEl = document.getElementById('hdr-daily-pnl');
    dailyEl.textContent = formatCurrency(dailyPnl);
    dailyEl.className = 'value ' + signClass(dailyPnl);

    const totalEl = document.getElementById('hdr-total-pnl');
    totalEl.textContent = formatCurrency(totalPnl);
    totalEl.className = 'value ' + signClass(totalPnl);

    document.getElementById('hdr-win-rate').textContent    = (winRate * 100).toFixed(1) + '%';
    document.getElementById('hdr-trade-count').textContent = String(trades);
    document.getElementById('hdr-open-pos').textContent    = String(openPos);

    updatePnlChart(trades, dailyPnl);
}

// ── Socket.IO event handlers ─────────────────────────────────────────────────

socket.on('positions_update', updatePositionsTable);
socket.on('trade_update',     prependTradeRow);
socket.on('equity_update',    appendEquityPoint);
socket.on('signal_update',    prependSignalEntry);
socket.on('metrics_update',   updateMetricsDisplay);

// ── Boot ─────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    initCharts();

    Promise.all([
        fetch('/api/positions').then(r => r.json()),
        fetch('/api/trades').then(r => r.json()),
        fetch('/api/equity').then(r => r.json()),
        fetch('/api/signals').then(r => r.json()),
        fetch('/api/metrics').then(r => r.json()),
    ]).then(([positions, trades, equity, signals, metrics]) => {
        updatePositionsTable(positions);
        populateTradesTable(trades);
        populateEquityChart(equity);
        populateSignalsLog(signals);
        updateMetricsDisplay(metrics);
    }).catch(err => {
        console.warn('Failed to fetch initial state:', err);
    });
});
