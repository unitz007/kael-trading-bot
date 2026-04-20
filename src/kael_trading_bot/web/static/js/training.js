/**
 * Training page client-side logic.
 *
 * Handles:
 *  - Fetching available forex pairs
 *  - Starting a training job
 *  - Polling active job status
 *  - Displaying trained models
 */

"use strict";

// ---------------------------------------------------------------------------
// DOM references
// ---------------------------------------------------------------------------

const pairSelect = document.getElementById("pair-select");
const startBtn = document.getElementById("start-btn");
const startError = document.getElementById("start-error");
const activeJobsDiv = document.getElementById("active-jobs");
const modelsTable = document.getElementById("models-table");
const modelsTbody = document.getElementById("models-tbody");
const modelsPlaceholder = document.getElementById("models-placeholder");

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let pollTimer = null;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function api(url, options = {}) {
    const resp = await fetch(url, {
        headers: { "Content-Type": "application/json" },
        ...options,
    });
    if (!resp.ok) {
        const body = await resp.json().catch(() => ({}));
        const err = new Error(body.detail?.message || body.detail || resp.statusText);
        err.status = resp.status;
        err.body = body.detail;
        throw err;
    }
    return resp.json();
}

function escHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}

function fmtPercent(val) {
    if (val == null) return "—";
    return (val * 100).toFixed(2) + "%";
}

function hide(el) {
    el.classList.add("hidden");
}

function show(el) {
    el.classList.remove("hidden");
}

// ---------------------------------------------------------------------------
// Pairs
// ---------------------------------------------------------------------------

async function loadPairs() {
    try {
        const data = await api("/api/pairs");
        pairSelect.innerHTML = "";
        if (data.pairs.length === 0) {
            pairSelect.innerHTML = '<option value="">No pairs available</option>';
            startBtn.disabled = true;
            return;
        }
        data.pairs.forEach((p) => {
            const opt = document.createElement("option");
            opt.value = p.ticker;
            opt.textContent = p.display;
            pairSelect.appendChild(opt);
        });
        startBtn.disabled = false;
        pairSelect.disabled = false;
    } catch (e) {
        pairSelect.innerHTML = '<option value="">Failed to load pairs</option>';
        console.error("Failed to load pairs:", e);
    }
}

// ---------------------------------------------------------------------------
// Start training
// ---------------------------------------------------------------------------

async function startTraining() {
    const pair = pairSelect.value;
    if (!pair) return;

    startBtn.disabled = true;
    startBtn.textContent = "Starting…";
    hide(startError);

    try {
        await api(`/api/training/start?pair=${encodeURIComponent(pair)}`, {
            method: "POST",
        });
        // Refresh jobs and models immediately, then keep polling
        await refreshJobs();
        loadModels();
        startPolling();
    } catch (e) {
        if (e.status === 409) {
            showConflictMessage(e.body);
        } else {
            showError(e.message);
        }
        startBtn.disabled = false;
    } finally {
        startBtn.textContent = "Start Training";
        // Re-check if there's a running job for this pair after a brief delay
        setTimeout(refreshStartButtonState, 500);
    }
}

function showConflictMessage(detail) {
    if (detail && detail.job) {
        startError.textContent =
            detail.message || "A training job is already running for this pair.";
    } else {
        startError.textContent =
            "A training job is already running for this pair.";
    }
    show(startError);
}

function showError(msg) {
    startError.textContent = msg;
    show(startError);
}

// ---------------------------------------------------------------------------
// Polling active jobs
// ---------------------------------------------------------------------------

function startPolling() {
    if (pollTimer) return;
    pollTimer = setInterval(async () => {
        const hasActive = await refreshJobs();
        if (!hasActive) {
            stopPolling();
            loadModels(); // reload models once all jobs finish
            refreshStartButtonState();
        }
    }, 3000);
}

function stopPolling() {
    if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
    }
}

async function refreshJobs() {
    try {
        const data = await api("/api/training/status");
        renderJobs(data.jobs || []);

        // Return true if there are any pending or running jobs
        return data.jobs.some(
            (j) => j.status === "pending" || j.status === "running"
        );
    } catch (e) {
        console.error("Failed to refresh jobs:", e);
        return false;
    }
}

function renderJobs(jobs) {
    // Only show non-terminal jobs (pending, running) and recently completed/failed
    const visible = jobs.filter((j) => {
        if (j.status === "pending" || j.status === "running") return true;
        // Show completed/failed for 5 minutes
        if (!j.completed_at) return false;
        const completed = new Date(j.completed_at);
        const diff = Date.now() - completed.getTime();
        return diff < 5 * 60 * 1000;
    });

    if (visible.length === 0) {
        activeJobsDiv.innerHTML = '<p class="placeholder-text">No active training jobs.</p>';
        return;
    }

    activeJobsDiv.innerHTML = visible
        .map((job) => {
            const displayPair = job.pair.replace("=X", "");
            const statusClass = job.status;
            const statusLabel = job.status;

            let messageHtml = "";
            if (job.status === "running" || job.status === "pending") {
                messageHtml = `<span class="spinner"></span>${escHtml(job.message || "Processing…")}`;
            } else {
                messageHtml = escHtml(job.message || "");
            }

            let metricsHtml = "";
            if (job.metrics && Object.keys(job.metrics).length > 0) {
                metricsHtml = `<div class="job-metrics">
                    ${Object.entries(job.metrics)
                        .map(
                            ([k, v]) => `<div class="metric">
                        <span class="metric-label">${escHtml(k)}</span>
                        <span class="metric-value">${fmtPercent(v)}</span>
                    </div>`
                        )
                        .join("")}
                </div>`;
            }

            return `<div class="job-card">
                <div class="job-header">
                    <span class="job-pair">${escHtml(displayPair)}</span>
                    <span class="job-status ${statusClass}">${statusLabel}</span>
                </div>
                <div class="job-message">${messageHtml}</div>
                ${metricsHtml}
            </div>`;
        })
        .join("");
}

// ---------------------------------------------------------------------------
// Disable start button when a job is running for the selected pair
// ---------------------------------------------------------------------------

async function refreshStartButtonState() {
    const pair = pairSelect.value;
    if (!pair) {
        startBtn.disabled = true;
        return;
    }
    try {
        const data = await api("/api/training/status");
        const hasRunning = (data.jobs || []).some(
            (j) =>
                j.pair === pair &&
                (j.status === "pending" || j.status === "running")
        );
        startBtn.disabled = hasRunning;
    } catch {
        startBtn.disabled = false;
    }
}

// ---------------------------------------------------------------------------
// Models table
// ---------------------------------------------------------------------------

async function loadModels() {
    try {
        const data = await api("/api/models");
        const models = data.models || [];

        if (models.length === 0) {
            hide(modelsTable);
            modelsPlaceholder.textContent = "No trained models yet.";
            show(modelsPlaceholder);
            return;
        }

        hide(modelsPlaceholder);
        modelsTbody.innerHTML = models
            .map((m) => {
                const pair = m.name ? m.name.replace(/_/g, " ").toUpperCase() : "—";
                return `<tr>
                    <td>${escHtml(pair)}</td>
                    <td>${escHtml(m.version || "—")}</td>
                    <td>${escHtml(m.model_type || "—")}</td>
                    <td>${escHtml(m.trained_at || "—")}</td>
                    <td>${fmtPercent(m.metrics?.accuracy)}</td>
                    <td>${fmtPercent(m.metrics?.precision)}</td>
                    <td>${fmtPercent(m.metrics?.recall)}</td>
                    <td>${fmtPercent(m.metrics?.f1)}</td>
                </tr>`;
            })
            .join("");
        show(modelsTable);
    } catch (e) {
        console.error("Failed to load models:", e);
        modelsPlaceholder.textContent = "Failed to load models.";
        show(modelsPlaceholder);
        hide(modelsTable);
    }
}

// ---------------------------------------------------------------------------
// Initialise
// ---------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
    loadPairs();
    refreshJobs().then((hasActive) => {
        if (hasActive) startPolling();
    });
    loadModels();

    startBtn.addEventListener("click", startTraining);
    pairSelect.addEventListener("change", refreshStartButtonState);
});
