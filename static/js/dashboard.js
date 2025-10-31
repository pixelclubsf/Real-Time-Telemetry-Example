/**
 * Solar Boat Dashboard - JavaScript
 * Handles interactive dashboard functionality
 */

// ============================================================================
// DOM Elements
// ============================================================================

const loadSampleBtn = document.getElementById('loadSampleBtn');
const exportBtn = document.getElementById('exportBtn');
const durationInput = document.getElementById('durationInput');
const intervalInput = document.getElementById('intervalInput');
const loadingSpinner = document.getElementById('loadingSpinner');
const metricsPanel = document.getElementById('metricsPanel');
const chartsGrid = document.getElementById('chartsGrid');
const emptyState = document.getElementById('emptyState');
const errorMessage = document.getElementById('errorMessage');
const errorText = document.getElementById('errorText');

// Chart containers
const speedChartDiv = document.getElementById('speedChart');
const voltageChartDiv = document.getElementById('voltageChart');
const currentChartDiv = document.getElementById('currentChart');
const efficiencyChartDiv = document.getElementById('efficiencyChart');
const gpsChartDiv = document.getElementById('gpsChart');

// ============================================================================
// Event Listeners
// ============================================================================

loadSampleBtn.addEventListener('click', loadSampleData);
exportBtn.addEventListener('click', exportData);

// ============================================================================
// Functions
// ============================================================================

/**
 * Load sample data from the backend
 */
async function loadSampleData() {
    const duration = parseInt(durationInput.value);
    const interval = parseInt(intervalInput.value);

    if (duration < 30 || interval < 1) {
        showError('Invalid duration or interval');
        return;
    }

    try {
        showLoading(true);

        // Load sample data
        const loadResponse = await fetch('/api/load-sample-data', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ duration, interval })
        });

        if (!loadResponse.ok) {
            throw new Error('Failed to load sample data');
        }

        // Load charts
        await loadCharts();

        // Load metrics
        await loadMetrics();

        // Show UI elements
        emptyState.classList.add('hidden');
        metricsPanel.classList.remove('hidden');
        chartsGrid.classList.remove('hidden');
        exportBtn.disabled = false;

        showLoading(false);
        showNotification('Sample data loaded successfully!');

    } catch (error) {
        showError(`Error loading data: ${error.message}`);
        showLoading(false);
    }
}

/**
 * Load and display all charts
 */
async function loadCharts() {
    try {
        const response = await fetch('/api/charts');

        if (!response.ok) {
            throw new Error('Failed to load charts');
        }

        const data = await response.json();

        if (data.status === 'error') {
            throw new Error(data.message);
        }

        // Plot all charts
        Plotly.newPlot(speedChartDiv, data.speed_chart.data, data.speed_chart.layout);
        Plotly.newPlot(voltageChartDiv, data.voltage_chart.data, data.voltage_chart.layout);
        Plotly.newPlot(currentChartDiv, data.current_chart.data, data.current_chart.layout);
        Plotly.newPlot(efficiencyChartDiv, data.efficiency_chart.data, data.efficiency_chart.layout);
        Plotly.newPlot(gpsChartDiv, data.gps_chart.data, data.gps_chart.layout);

    } catch (error) {
        showError(`Error loading charts: ${error.message}`);
    }
}

/**
 * Load and display metrics
 */
async function loadMetrics() {
    try {
        const response = await fetch('/api/metrics');

        if (!response.ok) {
            throw new Error('Failed to load metrics');
        }

        const data = await response.json();

        if (data.status === 'error') {
            throw new Error(data.message);
        }

        // Update metric cards
        document.getElementById('maxSpeed').textContent = data.max_speed;
        document.getElementById('minSpeed').textContent = data.min_speed;
        document.getElementById('avgSpeed').textContent = data.avg_speed;
        document.getElementById('distance').textContent = data.distance;
        document.getElementById('duration').textContent = data.duration;
        document.getElementById('maxVoltage').textContent = data.max_voltage;
        document.getElementById('minVoltage').textContent = data.min_voltage;
        document.getElementById('maxCurrent').textContent = data.max_current;

    } catch (error) {
        showError(`Error loading metrics: ${error.message}`);
    }
}

/**
 * Export data as JSON file
 */
async function exportData() {
    try {
        showLoading(true);

        const response = await fetch('/api/export');

        if (!response.ok) {
            throw new Error('Failed to export data');
        }

        const data = await response.json();

        if (data.status === 'error') {
            throw new Error(data.message);
        }

        // Create blob and download
        const dataStr = JSON.stringify(data, null, 2);
        const blob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `solar-boat-data-${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);

        showLoading(false);
        showNotification('Data exported successfully!');

    } catch (error) {
        showError(`Error exporting data: ${error.message}`);
        showLoading(false);
    }
}

/**
 * Show loading spinner
 */
function showLoading(show) {
    if (show) {
        loadingSpinner.classList.remove('hidden');
        loadSampleBtn.disabled = true;
    } else {
        loadingSpinner.classList.add('hidden');
        loadSampleBtn.disabled = false;
    }
}

/**
 * Show error message
 */
function showError(message) {
    errorText.textContent = message;
    errorMessage.classList.remove('hidden');

    // Auto-hide after 5 seconds
    setTimeout(() => {
        closeError();
    }, 5000);
}

/**
 * Close error message
 */
function closeError() {
    errorMessage.classList.add('hidden');
}

/**
 * Show notification (optional)
 */
function showNotification(message) {
    console.log('Notification:', message);
    // You could enhance this with a toast notification
}

// ============================================================================
// Initialize
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    // Initialize button states
    exportBtn.disabled = true;
});

// Responsive chart resizing
window.addEventListener('resize', debounce(() => {
    if (!chartsGrid.classList.contains('hidden')) {
        Plotly.Plots.resize(speedChartDiv);
        Plotly.Plots.resize(voltageChartDiv);
        Plotly.Plots.resize(currentChartDiv);
        Plotly.Plots.resize(efficiencyChartDiv);
        Plotly.Plots.resize(gpsChartDiv);
    }
}, 250));

/**
 * Debounce utility function
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}
