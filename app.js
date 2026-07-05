(function () {
  const FIELDS = [
    { key: "initSal", label: "Initial Salinity", input: "initSal" },
    { key: "mc", label: "MC", input: "mc" },
    { key: "wr", label: "WR", input: "wr" },
    { key: "hc", label: "HC", input: "hc" },
    { key: "initCond", label: "Initial Conductivity", input: "initCond" },
    { key: "finCond", label: "Final Conductivity", input: "finCond" },
    { key: "initTds", label: "Initial TDS", input: "initTds" },
    { key: "finTds", label: "Final TDS", input: "finTds" },
    { key: "initPh", label: "Initial pH", input: "initPh" },
    { key: "finPh", label: "Final pH", input: "finPh" }
  ];

  const LEADERBOARD = [
    { model: "ETR", r2: 0.978519, rmse: 2.056682, mae: 1.363303, smape: 0.083535 },
    { model: "SVR", r2: 0.970827, rmse: 2.327423, mae: 1.159532, smape: 0.063451 },
    { model: "XGB", r2: 0.969496, rmse: 2.461862, mae: 1.577077, smape: 0.094204 },
    { model: "KNN", r2: 0.968434, rmse: 2.489512, mae: 1.520336, smape: 0.081022 },
    { model: "GBR", r2: 0.966742, rmse: 2.560559, mae: 1.746675, smape: 0.105077 },
    { model: "HGBR", r2: 0.963368, rmse: 2.680731, mae: 1.828667, smape: 0.118701 },
    { model: "LGBM", r2: 0.961849, rmse: 2.744554, mae: 1.816675, smape: 0.116852 },
    { model: "RF", r2: 0.952990, rmse: 3.065294, mae: 2.019579, smape: 0.118850 }
  ];

  const DATA = window.PW_DATA || [];
  const stats = buildStats(DATA);
  let batchResults = [];

  document.getElementById("datasetStatus").textContent =
    `${DATA.length.toLocaleString()} datapoints from Data points.xlsx`;
  document.getElementById("singleTab").addEventListener("click", () => setMode("single"));
  document.getElementById("batchTab").addEventListener("click", () => setMode("batch"));
  document.getElementById("singleModeRadio").addEventListener("change", () => setMode("single"));
  document.getElementById("batchModeRadio").addEventListener("change", () => setMode("batch"));
  document.getElementById("predictForm").addEventListener("submit", (event) => {
    event.preventDefault();
    predict();
  });
  document.getElementById("resetBtn").addEventListener("click", resetForm);
  document.getElementById("csvUpload").addEventListener("change", handleBatchUpload);
  document.getElementById("downloadBatchBtn").addEventListener("click", downloadBatchResults);

  function buildStats(rows) {
    const result = {};
    [...FIELDS, { key: "re" }].forEach((field) => {
      const values = rows.map((row) => row[field.key]).filter(Number.isFinite);
      const min = Math.min(...values);
      const max = Math.max(...values);
      const mean = values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1);
      const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / Math.max(values.length - 1, 1);
      result[field.key] = { min, max, mean, sd: Math.sqrt(variance) || 1 };
    });
    return result;
  }

  function setMode(mode) {
    const isBatch = mode === "batch";
    document.getElementById("predictForm").classList.toggle("hidden", isBatch);
    document.getElementById("batchPanel").classList.toggle("hidden", !isBatch);
    document.getElementById("singleTab").classList.toggle("active", !isBatch);
    document.getElementById("batchTab").classList.toggle("active", isBatch);
    document.getElementById("singleModeRadio").checked = !isBatch;
    document.getElementById("batchModeRadio").checked = isBatch;
  }

  function fmt(value, digits = 2) {
    return Number(value).toLocaleString(undefined, {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits
    });
  }

  function readInputs() {
    const sample = {};
    FIELDS.forEach((field) => {
      sample[field.key] = Number(document.getElementById(field.input).value);
    });
    return sample;
  }

  function distance(row, sample) {
    return Math.sqrt(FIELDS.reduce((sum, field) => {
      const zRow = (row[field.key] - stats[field.key].mean) / stats[field.key].sd;
      const zSample = (sample[field.key] - stats[field.key].mean) / stats[field.key].sd;
      return sum + (zRow - zSample) ** 2;
    }, 0));
  }

  function weightedQuantile(items, key, quantile) {
    const sorted = [...items].sort((a, b) => a.row[key] - b.row[key]);
    const total = sorted.reduce((sum, item) => sum + item.weight, 0);
    let cumulative = 0;
    for (const item of sorted) {
      cumulative += item.weight;
      if (cumulative / total >= quantile) return item.row[key];
    }
    return sorted[sorted.length - 1].row[key];
  }

  function computePrediction(sample, neighborCount = 18) {
    const k = Math.max(5, Math.min(50, Number(neighborCount) || 18));
    const ranked = DATA
      .map((row) => ({ row, d: distance(row, sample) }))
      .sort((a, b) => a.d - b.d)
      .slice(0, k)
      .map((item) => ({ ...item, weight: 1 / Math.max(item.d, 0.000001) ** 2 }));
    const weightTotal = ranked.reduce((sum, item) => sum + item.weight, 0);
    const re = ranked.reduce((sum, item) => sum + item.row.re * item.weight, 0) / weightTotal;
    const lower = weightedQuantile(ranked, "re", 0.1);
    const upper = weightedQuantile(ranked, "re", 0.9);
    const outside = FIELDS.filter((field) => sample[field.key] < stats[field.key].min || sample[field.key] > stats[field.key].max);
    return { re, lower, upper, ranked, outside };
  }

  function predict() {
    if (!DATA.length) return;
    const result = computePrediction(readInputs());
    renderPrediction(result);
    renderMatches(result.ranked.slice(0, 10));
    renderNeighborSummary(result);
  }

  function renderPrediction(result) {
    const rounded = fmt(result.re, 2);
    document.getElementById("predictionValue").textContent = rounded;
    document.querySelector(".meter-arc").style.setProperty("--score", `${Math.max(0, Math.min(100, result.re))}%`);
    document.getElementById("rangeFill").style.width = `${Math.max(0, Math.min(100, result.re))}%`;
    document.getElementById("predictionSummary").textContent =
      `Estimated RE is ${rounded}% using inverse-distance weighting from the nearest dataset records. Nearby range: ${fmt(result.lower)}% to ${fmt(result.upper)}%.`;

    const badge = document.getElementById("confidenceBadge");
    if (result.outside.length) {
      badge.textContent = "Outside dataset range";
      badge.style.color = "var(--warn)";
      document.getElementById("rangeWarning").textContent =
        `Caution: ${result.outside.map((field) => field.label).join(", ")} is outside the supplied dataset range.`;
    } else {
      const minDistance = result.ranked[0] ? result.ranked[0].d : 0;
      badge.textContent = minDistance < 0.2 ? "Exact/near match" : minDistance < 0.75 ? "High similarity" : "Moderate similarity";
      badge.style.color = minDistance < 0.75 ? "var(--good)" : "var(--warn)";
      document.getElementById("rangeWarning").textContent = "";
    }
  }

  function renderNeighborSummary(result) {
    const matches = result.ranked;
    const meanDistance = matches.reduce((sum, item) => sum + item.d, 0) / Math.max(matches.length, 1);
    const meanRe = matches.reduce((sum, item) => sum + item.row.re, 0) / Math.max(matches.length, 1);
    const cards = [
      { label: "Nearest records", value: matches.length, helper: "Rows used for local estimate", width: 100 },
      { label: "Closest distance", value: fmt(matches[0] ? matches[0].d : 0, 3), helper: "Standardized feature distance", width: Math.max(5, 100 - (matches[0] ? matches[0].d * 35 : 0)) },
      { label: "Mean neighbor RE", value: `${fmt(meanRe)}%`, helper: "Unweighted average of selected records", width: meanRe },
      { label: "Mean distance", value: fmt(meanDistance, 3), helper: "Average standardized distance", width: Math.max(5, 100 - meanDistance * 25) }
    ];
    document.getElementById("contributionCards").innerHTML = cards.map((item) => `
      <div class="inferred-card">
        <span>${item.label}</span>
        <strong>${item.value}</strong>
        <small>${item.helper}</small>
        <div class="mini-bar"><i style="width:${Math.max(0, Math.min(100, item.width))}%"></i></div>
      </div>
    `).join("");
  }

  function renderMatches(matches) {
    document.getElementById("matchesBody").innerHTML = matches.map((item, index) => {
      const row = item.row;
      return `
        <tr>
          <td>${index + 1}</td>
          <td>${fmt(row.initSal, 3)}</td>
          <td>${fmt(row.mc, 3)}</td>
          <td>${fmt(row.wr, 3)}</td>
          <td>${fmt(row.hc, 3)}</td>
          <td>${fmt(row.initCond, 3)}</td>
          <td>${fmt(row.finCond, 3)}</td>
          <td>${fmt(row.initTds, 3)}</td>
          <td>${fmt(row.finTds, 3)}</td>
          <td>${fmt(row.initPh, 3)}</td>
          <td>${fmt(row.finPh, 3)}</td>
          <td>${fmt(row.re, 2)}%</td>
        </tr>
      `;
    }).join("");
  }

  function renderModelStats() {
    document.getElementById("modelStats").innerHTML = `
      <dt>Dataset rows</dt><dd>${DATA.length.toLocaleString()}</dd>
      <dt>Estimator</dt><dd>Weighted KNN</dd>
      <dt>RE range</dt><dd>${fmt(stats.re.min)}-${fmt(stats.re.max)}%</dd>
      <dt>Notebook best</dt><dd>ETR</dd>
      <dt>ETR RMSE</dt><dd>2.0567</dd>
    `;
  }

  function renderLeaderboard() {
    document.getElementById("leaderboardBody").innerHTML = LEADERBOARD.map((row, index) => `
      <tr>
        <td>${index + 1}</td>
        <td>${row.model}</td>
        <td>${fmt(row.r2, 4)}</td>
        <td>${fmt(row.rmse, 4)}</td>
        <td>${fmt(row.mae, 4)}</td>
        <td>${fmt(row.smape, 4)}</td>
      </tr>
    `).join("");
  }

  function resetForm() {
    const defaults = DATA[Math.floor(DATA.length / 2)] || {
      initSal: 1.98,
      mc: 1.45,
      wr: 48,
      hc: 1.01,
      initCond: 11.57,
      finCond: 7.8,
      initTds: 6.74,
      finTds: 9.005,
      initPh: 8.79,
      finPh: 7.055
    };
    FIELDS.forEach((field) => {
      document.getElementById(field.input).value = defaults[field.key];
    });
    predict();
  }

  function normalizeHeader(header) {
    return String(header || "")
      .trim()
      .toLowerCase()
      .replace(/\s+/g, "")
      .replace(/_/g, "")
      .replace(/[^a-z0-9]/g, "");
  }

  function splitCsvLine(line) {
    const cells = [];
    let current = "";
    let inQuotes = false;
    for (let i = 0; i < line.length; i += 1) {
      const char = line[i];
      const next = line[i + 1];
      if (char === '"' && inQuotes && next === '"') {
        current += '"';
        i += 1;
      } else if (char === '"') {
        inQuotes = !inQuotes;
      } else if (char === "," && !inQuotes) {
        cells.push(current.trim());
        current = "";
      } else {
        current += char;
      }
    }
    cells.push(current.trim());
    return cells;
  }

  function parseCsv(text) {
    const aliases = {
      initSal: ["initsal", "initialsalinity"],
      mc: ["mc"],
      wr: ["wr"],
      hc: ["hc"],
      initCond: ["initcond", "initialconductivity"],
      finCond: ["fincond", "finalconductivity"],
      initTds: ["inittds", "initialtds"],
      finTds: ["fintds", "finaltds"],
      initPh: ["initph", "initialph"],
      finPh: ["finph", "finalph"]
    };
    const lines = text.replace(/^\uFEFF/, "").split(/\r?\n/).filter((line) => line.trim());
    if (lines.length < 2) throw new Error("CSV must include a header row and at least one data row.");
    const normalizedHeaders = splitCsvLine(lines[0]).map(normalizeHeader);
    const indexes = {};
    FIELDS.forEach((field) => {
      indexes[field.key] = normalizedHeaders.findIndex((header) => aliases[field.key].includes(header));
    });
    const missing = FIELDS.filter((field) => indexes[field.key] === -1);
    if (missing.length) throw new Error(`Missing columns: ${missing.map((field) => field.label).join(", ")}.`);
    return lines.slice(1).map((line, index) => {
      const cells = splitCsvLine(line);
      const sample = {};
      FIELDS.forEach((field) => {
        sample[field.key] = Number(cells[indexes[field.key]]);
      });
      return { rowNumber: index + 2, sample };
    });
  }

  function handleBatchUpload(event) {
    const file = event.target.files && event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const rows = parseCsv(String(reader.result || ""));
        batchResults = rows.map((row) => {
          const invalid = FIELDS.filter((field) => !Number.isFinite(row.sample[field.key]));
          if (invalid.length) {
            return { rowNumber: row.rowNumber, re: null, lower: null, upper: null, distance: null, status: `Invalid: ${invalid.map((field) => field.label).join(", ")}` };
          }
          const result = computePrediction(row.sample);
          return {
            rowNumber: row.rowNumber,
            re: result.re,
            lower: result.lower,
            upper: result.upper,
            distance: result.ranked[0] ? result.ranked[0].d : null,
            status: result.outside.length ? "Outside range" : "OK"
          };
        });
        renderBatchResults(batchResults);
        document.getElementById("downloadBatchBtn").disabled = !batchResults.some((row) => Number.isFinite(row.re));
        document.getElementById("batchStatus").textContent =
          `${batchResults.length} rows processed. Showing first ${Math.min(100, batchResults.length)} rows.`;
      } catch (error) {
        batchResults = [];
        renderBatchResults([]);
        document.getElementById("downloadBatchBtn").disabled = true;
        document.getElementById("batchStatus").textContent = error.message;
      }
    };
    reader.readAsText(file);
  }

  function renderBatchResults(rows) {
    document.getElementById("batchBody").innerHTML = rows.slice(0, 100).map((row) => `
      <tr>
        <td>${row.rowNumber}</td>
        <td>${Number.isFinite(row.re) ? fmt(row.re, 2) : "-"}</td>
        <td>${Number.isFinite(row.lower) ? `${fmt(row.lower)}-${fmt(row.upper)}%` : "-"}</td>
        <td>${Number.isFinite(row.distance) ? fmt(row.distance, 3) : "-"}</td>
        <td>${row.status}</td>
      </tr>
    `).join("");
  }

  function csvEscape(value) {
    if (value === null || value === undefined || Number.isNaN(value)) return "";
    const text = typeof value === "number" ? String(Number(value.toFixed(6))) : String(value);
    return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  }

  function downloadBatchResults() {
    if (!batchResults.length) return;
    const headers = ["Input Row", "Predicted RE (%)", "Lower Band (%)", "Upper Band (%)", "Closest Distance", "Status"];
    const rows = batchResults.map((row) => [row.rowNumber, row.re, row.lower, row.upper, row.distance, row.status]);
    const csv = [headers, ...rows].map((row) => row.map(csvEscape).join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "removal_efficiency_batch_predictions.csv";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  renderModelStats();
  renderLeaderboard();
  resetForm();
})();
