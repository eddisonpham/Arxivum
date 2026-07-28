/* ── Research Library MCP — Visual Panel JS ─────────────────────────── */

const API = "/api/v1";

// ── Helpers ──────────────────────────────────────────────────────────────

async function api(path, opts = {}) {
  const res = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.message || data.error || `HTTP ${res.status}`);
  return data;
}

function el(id) { return document.getElementById(id); }
function show(el) { if (el) el.style.display = ""; }
function hide(el) { if (el) el.style.display = "none"; }
function esc(s) {
  const d = document.createElement("div");
  d.textContent = s || "";
  return d.innerHTML;
}
function truncate(s, n) { return s && s.length > n ? s.slice(0, n) + "…" : s || ""; }

function setStatus(msg, type) {
  const e = el("search-status");
  if (!e) return;
  e.textContent = msg;
  e.className = "status-msg " + (type || "");
}

// ── Model indicator ──────────────────────────────────────────────────────

async function updateModelIndicator() {
  const indicator = el("model-indicator");
  if (!indicator) return;
  try {
    const h = await api("/health");
    const ms = h.model_state || {};
    const dot = indicator.querySelector(".dot");
    const text = indicator.querySelector(".model-text");
    if (ms.llm_loaded) {
      dot.className = "dot dot-ready";
      text.textContent = "LLM Ready";
    } else if (ms.resident) {
      dot.className = "dot dot-loading";
      text.textContent = `Loading ${ms.resident}…`;
    } else {
      dot.className = "dot dot-idle";
      text.textContent = "Idle";
    }
  } catch (_) { /* ignore */ }
}
setInterval(updateModelIndicator, 5000);

// ── Search & Import ──────────────────────────────────────────────────────

async function doSearch() {
  const query = el("search-query").value.trim();
  if (!query) { setStatus("Please enter a search query.", "error"); return; }
  const btn = el("search-btn");
  btn.disabled = true;
  btn.textContent = "Searching…";
  setStatus("Searching arXiv…", "");
  try {
    const body = {
      query,
      max_results: parseInt(el("search-max").value) || 10,
      primary_category: el("search-category").value,
      auto_enrich: el("search-enrich").checked,
    };
    const data = await api("/library/search", { method: "POST", body: JSON.stringify(body) });
    setStatus(`Imported ${data.total_found} paper(s) into the library.`, "success");
    loadLibrary();
    updateModelIndicator();
  } catch (e) {
    setStatus(`Search failed: ${e.message}`, "error");
  } finally {
    btn.disabled = false;
    btn.textContent = "Search & Import";
  }
}

// ── Query local library ──────────────────────────────────────────────────

async function doQuery() {
  const query = el("query-text").value.trim();
  if (!query) return;
  const btn = el("query-btn");
  btn.disabled = true;
  try {
    const body = {
      query,
      top_k: parseInt(el("query-topk").value) || 5,
      min_citations: parseInt(el("query-mincit").value) || 0,
      rerank: el("query-rerank").checked,
    };
    const data = await api("/library/query", { method: "POST", body: JSON.stringify(body) });
    renderQueryResults(data.results || []);
  } catch (e) {
    setStatus(`Query failed: ${e.message}`, "error");
  } finally {
    btn.disabled = false;
  }
}

function renderQueryResults(results) {
  const body = el("results-body");
  const title = el("table-title");
  if (!body) return;
  title.innerHTML = `Query Results <span class="badge">${results.length}</span>`;
  if (!results.length) {
    body.innerHTML = "";
    show(el("empty-state"));
    el("empty-state").querySelector("p").textContent = "No matching papers found in the library.";
    return;
  }
  hide(el("empty-state"));
  body.innerHTML = results.map(r => `
    <tr>
      <td><a class="paper-title" href="/paper/${esc(r.arxiv_id)}">${esc(r.title)}</a>
        <div class="text-xs text-muted">Score: ${(r.score * 100).toFixed(0)}% · ${esc(r.chunk_type)}</div></td>
      <td class="paper-authors" title="${esc(r.authors.join('; '))}">${esc(r.authors.join('; '))}</td>
      <td><span class="cat-badge">${esc(r.primary_category || '')}</span></td>
      <td class="cit-count">${r.citation_count >= 0 ? r.citation_count : '—'}</td>
      <td class="venue-text">${esc(r.venue || '—')}</td>
      <td class="text-xs text-muted">${esc(truncate(r.abstract_snippet, 80))}</td>
      <td></td>
    </tr>`).join("");
}

// ── Load library table ───────────────────────────────────────────────────

async function loadLibrary() {
  const body = el("results-body");
  const title = el("table-title");
  if (!body) return;
  show(el("loading-state"));
  hide(el("empty-state"));
  body.innerHTML = "";
  try {
    const sortBy = el("sort-by") ? el("sort-by").value : "created_at";
    const data = await api(`/library?limit=100&sort_by=${sortBy}`);
    const papers = data.papers || [];
    const count = el("paper-count");
    if (count) count.textContent = `(${data.total || 0})`;
    title.innerHTML = `Library <span class="badge">${data.total || 0}</span>`;
    if (!papers.length) {
      hide(el("loading-state"));
      show(el("empty-state"));
      el("empty-state").querySelector("p").textContent = "No papers in the library yet. Search arXiv above to get started.";
      return;
    }
    hide(el("loading-state"));
    hide(el("empty-state"));
    body.innerHTML = papers.map(p => `
      <tr>
        <td><a class="paper-title" href="/paper/${esc(p.arxiv_id)}">${esc(p.title)}</a></td>
        <td class="paper-authors" title="${esc(p.authors.join('; '))}">${esc(p.authors.join('; '))}</td>
        <td><span class="cat-badge">${esc(p.primary_category || '')}</span></td>
        <td class="cit-count">—</td>
        <td class="venue-text">—</td>
        <td class="text-xs text-muted">${esc(p.published || '')}</td>
        <td><button class="btn btn-ghost btn-sm" onclick="removePaper('${esc(p.arxiv_id)}')">Remove</button></td>
      </tr>`).join("");
  } catch (e) {
    hide(el("loading-state"));
    setStatus(`Failed to load library: ${e.message}`, "error");
  }
}

async function removePaper(arxivId) {
  if (!confirm(`Remove ${arxivId} from the library?`)) return;
  try {
    await api(`/library/${arxivId}`, { method: "DELETE" });
    loadLibrary();
  } catch (e) {
    alert(`Failed to remove: ${e.message}`);
  }
}

// ── Activity page ────────────────────────────────────────────────────────

async function loadActivity() {
  const feed = el("activity-feed");
  if (!feed) return;
  try {
    const data = await api("/activity?limit=100");
    const items = data.activities || [];
    if (!items.length) {
      feed.innerHTML = '<p class="text-muted">No activity yet.</p>';
      return;
    }
    const icons = { search: "🔍", import: "📥", summarize: "📝", idea: "💡", novelty: "🔍", query: "❓", remove: "🗑", enrich: "📊" };
    feed.innerHTML = items.map(a => `
      <div class="activity-item">
        <div class="activity-icon ${esc(a.action_type)}">${icons[a.action_type] || "•"}</div>
        <div class="activity-body">
          <div class="activity-type">${esc(a.action_type)}
            ${a.arxiv_id ? `<span class="text-muted text-xs">· ${esc(a.arxiv_id)}</span>` : ""}
            <span class="activity-status ${esc(a.status)}">${esc(a.status)}</span>
          </div>
          ${a.query ? `<div class="activity-query">${esc(a.query)}</div>` : ""}
          <div class="activity-time">${esc(a.created_at || '')}</div>
        </div>
      </div>`).join("");
  } catch (e) {
    feed.innerHTML = `<p class="text-muted">Failed to load activity: ${esc(e.message)}</p>`;
  }
}

// ── WebSocket for real-time activity ─────────────────────────────────────

function connectActivityWS() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${location.host}/ws/activity`);
  ws.onmessage = (ev) => {
    try {
      const msg = JSON.parse(ev.data);
      if (el("activity-feed")) loadActivity();
      if (msg.action_type === "summarize" && msg.status === "completed") {
        const path = location.pathname.match(/\/paper\/(.+)/);
        if (path && msg.arxiv_id && path[1] === msg.arxiv_id) {
          refreshSummaries(path[1]);
        }
      }
      if (msg.action_type === "idea" && msg.status === "completed") {
        const path = location.pathname.match(/\/paper\/(.+)/);
        if (path && msg.arxiv_id && path[1] === msg.arxiv_id) {
          refreshIdeas(path[1]);
        }
      }
    } catch (_) {}
  };
  ws.onclose = () => setTimeout(connectActivityWS, 5000);
}

// ── Paper detail page ────────────────────────────────────────────────────

async function loadPaperDetail(arxivId) {
  const container = el("paper-detail");
  if (!container) return;
  try {
    const p = await api(`/library/${arxivId}`);
    document.title = `${p.title} — Research Library`;
    const metrics = p.metrics || {};
    const summaries = p.summaries || [];
    const ideas = p.ideas || [];

    // Build summary tabs
    const sections = ["problem_statement", "methodology", "findings", "ablations", "discussion", "limitations", "overall"];
    const summaryMap = {};
    summaries.forEach(s => summaryMap[s.section] = s.content);

    container.innerHTML = `
      <div class="detail-header">
        <h1>${esc(p.title)}</h1>
        <div class="detail-meta">
          <span>${esc((p.authors || []).join(', '))}</span>
          <span><span class="cat-badge">${esc(p.primary_category || '')}</span></span>
          <span>${esc(p.published || '')}</span>
          ${p.pdf_url ? `<a href="${esc(p.pdf_url)}" target="_blank">PDF ↗</a>` : ""}
          ${p.abs_url ? `<a href="${esc(p.abs_url)}" target="_blank">arXiv ↗</a>` : ""}
        </div>
      </div>

      <div class="detail-grid">
        <div>
          <div class="card mb-16">
            <h2>Abstract</h2>
            <div class="abstract-box">${esc(p.abstract)}</div>
          </div>
          <div class="card">
            <div class="table-header"><h2>Summaries</h2>
              <button class="btn btn-primary btn-sm" id="gen-summary-btn">Generate</button>
            </div>
            <div class="tabs" id="summary-tabs">
              ${sections.map((s, i) => `<button class="tab ${i === 0 ? 'active' : ''}" data-section="${s}">${s.replace(/_/g, ' ')}</button>`).join("")}
            </div>
            <div id="summary-contents">
              ${sections.map((s, i) => `<div class="tab-content ${i === 0 ? 'active' : ''}" data-section="${s}"><p>${esc(summaryMap[s] || 'Not generated yet. Click "Generate" above.')}</p></div>`).join("")}
            </div>
          </div>
        </div>
        <div>
          <div class="card metrics-card mb-16">
            <h2>Metrics</h2>
            <div class="metric-row"><span class="metric-label">Citations</span><span class="metric-value">${metrics.citation_count >= 0 ? metrics.citation_count : '—'}</span></div>
            <div class="metric-row"><span class="metric-label">Influential</span><span class="metric-value">${metrics.influential_citation_count >= 0 ? metrics.influential_citation_count : '—'}</span></div>
            <div class="metric-row"><span class="metric-label">Venue</span><span class="metric-value">${esc(metrics.venue || '—')}</span></div>
            <button class="btn btn-ghost btn-sm mt-16" id="enrich-btn">Enrich via S2</button>
          </div>
          <div class="card">
            <div class="table-header"><h2>Ideas</h2>
              <button class="btn btn-primary btn-sm" id="gen-ideas-btn">Generate</button>
            </div>
            <div id="ideas-list">
              ${ideas.length ? ideas.map(i => renderIdeaCard(i)).join("") : '<p class="text-muted text-sm">No ideas yet. Click "Generate" above.</p>'}
            </div>
          </div>
        </div>
      </div>`;

    // Tab switching
    container.querySelectorAll(".tab").forEach(tab => {
      tab.onclick = () => {
        container.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
        container.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
        tab.classList.add("active");
        container.querySelector(`.tab-content[data-section="${tab.dataset.section}"]`).classList.add("active");
      };
    });

    // Generate summary
    const genSumBtn = el("gen-summary-btn");
    if (genSumBtn) genSumBtn.onclick = async () => {
      genSumBtn.disabled = true;
      const timer = showGeneratingTimer(genSumBtn, "Generating");
      try {
        const data = await api(`/summaries/${arxivId}`, { method: "POST", body: JSON.stringify({}) });
        const sm = data.summaries || {};
        sections.forEach(s => {
          const c = container.querySelector(`.tab-content[data-section="${s}"] p`);
          if (c) c.textContent = sm[s] || "N/A";
        });
      } catch (e) { alert(`Failed: ${e.message}`); }
      finally { clearInterval(timer); genSumBtn.disabled = false; genSumBtn.textContent = "Generate"; }
    };

    // Generate ideas
    const genIdeasBtn = el("gen-ideas-btn");
    if (genIdeasBtn) genIdeasBtn.onclick = async () => {
      genIdeasBtn.disabled = true;
      const timer = showGeneratingTimer(genIdeasBtn, "Generating");
      try {
        const data = await api(`/ideas/${arxivId}`, { method: "POST", body: JSON.stringify({}) });
        el("ideas-list").innerHTML = (data.ideas || []).map(i => renderIdeaCard(i)).join("");
        attachIdeaActions(arxivId);
      } catch (e) { alert(`Failed: ${e.message}`); }
      finally { clearInterval(timer); genIdeasBtn.disabled = false; genIdeasBtn.textContent = "Generate"; }
    };

    // Enrich
    const enrichBtn = el("enrich-btn");
    if (enrichBtn) enrichBtn.onclick = async () => {
      enrichBtn.disabled = true;
      const timer = showGeneratingTimer(enrichBtn, "Enriching");
      try {
        const data = await api(`/library/${arxivId}/enrich`, { method: "POST" });
        loadPaperDetail(arxivId);
      } catch (e) { alert(`Enrich failed: ${e.message}`); }
      finally { clearInterval(timer); enrichBtn.disabled = false; enrichBtn.textContent = "Enrich via S2"; }
    };

    attachIdeaActions(arxivId);
  } catch (e) {
    container.innerHTML = `<div class="empty-state"><p>Failed to load paper: ${esc(e.message)}</p></div>`;
  }
}

function renderIdeaCard(i) {
  const statusClass = i.status === 'approved' ? 'status-approved' : i.status === 'rejected' ? 'status-rejected' : 'status-pending';
  return `
    <div class="idea-card">
      <div class="idea-title">${esc(i.idea_text || i.title || 'Untitled idea')}</div>
      <span class="status-badge ${statusClass}">${esc(i.status)}</span>
      <div class="idea-actions">
        <button class="btn btn-sm btn-ghost" onclick="approveIdea(${i.id})">Approve</button>
        <button class="btn btn-sm btn-ghost" onclick="rejectIdea(${i.id})">Reject</button>
        <button class="btn btn-sm btn-secondary" onclick="verifyNovelty(${i.id})">Verify Novelty</button>
      </div>
      <div id="novelty-result-${i.id}"></div>
    </div>`;
}

function attachIdeaActions(arxivId) {
  // Actions are wired via inline onclick handlers (approveIdea, rejectIdea, verifyNovelty)
}

async function approveIdea(id) {
  try {
    await api(`/ideas/${id}/status`, { method: "POST", body: JSON.stringify({ status: "approved" }) });
    const path = location.pathname.match(/\/paper\/(.+)/);
    if (path) loadPaperDetail(path[1]);
  } catch (e) { alert(e.message); }
}

async function rejectIdea(id) {
  try {
    await api(`/ideas/${id}/status`, { method: "POST", body: JSON.stringify({ status: "rejected" }) });
    const path = location.pathname.match(/\/paper\/(.+)/);
    if (path) loadPaperDetail(path[1]);
  } catch (e) { alert(e.message); }
}

async function verifyNovelty(id) {
  const resultEl = el(`novelty-result-${id}`);
  if (resultEl) resultEl.innerHTML = '<div class="inline-spinner"><div class="spinner spinner-sm"></div><span class="text-xs text-muted">Verifying novelty (searching arXiv + LLM judgment)…</span></div>';
  try {
    const data = await api(`/novelty/${id}`, { method: "POST", body: JSON.stringify({}) });
    const cls = data.verdict === 'likely_novel' ? 'status-novel' : data.verdict === 'similar_exists' ? 'status-similar' : 'status-review';
    if (resultEl) resultEl.innerHTML = `<p class="text-xs"><span class="status-badge ${cls}">${esc(data.verdict)}</span> ${esc(data.notes || '')}</p>`;
  } catch (e) { if (resultEl) resultEl.innerHTML = `<p class="text-xs">Failed: ${esc(e.message)}</p>`; }
}

// ── Generating timer (shows elapsed seconds on a button) ────────────────

function showGeneratingTimer(btn, label) {
  const start = Date.now();
  btn.innerHTML = `<span class="btn-spinner"></span>${label}… <span class="timer-text">0s</span>`;
  return setInterval(() => {
    const elapsed = Math.floor((Date.now() - start) / 1000);
    btn.innerHTML = `<span class="btn-spinner"></span>${label}… <span class="timer-text">${elapsed}s</span>`;
  }, 1000);
}

// ── Auto-refresh summaries/ideas on paper page via WebSocket ─────────────

async function refreshSummaries(arxivId) {
  try {
    const data = await api(`/summaries/${arxivId}`);
    const sm = data.summaries || {};
    const sections = ["problem_statement", "methodology", "findings", "ablations", "discussion", "limitations", "overall"];
    sections.forEach(s => {
      const c = document.querySelector(`.tab-content[data-section="${s}"] p`);
      if (c && sm[s]) c.textContent = sm[s];
    });
  } catch (_) {}
}

async function refreshIdeas(arxivId) {
  try {
    const data = await api(`/ideas/${arxivId}`);
    const list = el("ideas-list");
    if (list) {
      const ideas = data.ideas || [];
      if (ideas.length) {
        list.innerHTML = ideas.map(i => renderIdeaCard(i)).join("");
      }
    }
  } catch (_) {}
}

// ── Init ─────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  const path = location.pathname;
  if (path === "/" || path === "/index.html") {
    el("search-btn")?.addEventListener("click", doSearch);
    el("query-btn")?.addEventListener("click", doQuery);
    el("refresh-btn")?.addEventListener("click", loadLibrary);
    el("sort-by")?.addEventListener("change", loadLibrary);
    el("search-query")?.addEventListener("keydown", e => { if (e.key === "Enter") doSearch(); });
    el("query-text")?.addEventListener("keydown", e => { if (e.key === "Enter") doQuery(); });
    loadLibrary();
    updateModelIndicator();
  } else if (path.startsWith("/paper/")) {
    const arxivId = path.split("/paper/")[1].replace(/\/$/, "");
    loadPaperDetail(arxivId);
  } else if (path === "/activity" || path === "/activity/") {
    loadActivity();
  }
  connectActivityWS();
});
