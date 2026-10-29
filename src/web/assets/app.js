/* ekko web UI. Plain ES2020, no dependencies.
   Everything that comes from the archive is escaped before it touches the
   DOM — transcripts contain arbitrary text, including HTML. */
(() => {
  "use strict";

  // ---- tiny helpers -------------------------------------------------------
  const $ = (sel, el = document) => el.querySelector(sel);
  const $$ = (sel, el = document) => Array.from(el.querySelectorAll(sel));
  const esc = (s) => String(s ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  const el = (html) => { const t = document.createElement("template"); t.innerHTML = html.trim(); return t.content.firstElementChild; };
  const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
  const fmtDate = (iso) => {
    if (!iso) return "";
    const d = new Date(iso);
    if (isNaN(d)) return iso;
    const now = new Date();
    const sameYear = d.getFullYear() === now.getFullYear();
    return d.toLocaleString(undefined, sameYear
      ? { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }
      : { year: "numeric", month: "short", day: "numeric" });
  };
  const fmtDuration = (ms) => {
    if (!ms || ms <= 0) return "";
    const m = Math.round(ms / 60000);
    return m < 60 ? `${m}m` : `${Math.floor(m / 60)}h ${m % 60}m`;
  };
  const shortId = (id) => (id || "").slice(0, 8);
  const srcLabel = { claude_code: "Claude Code", vscode_copilot: "VS Code Copilot", github_copilot: "Copilot CLI", codex: "Codex" };
  const toast = (msg, isErr = false) => {
    const t = $("#toast");
    t.textContent = msg; t.className = "toast" + (isErr ? " err" : ""); t.hidden = false;
    clearTimeout(toast._t); toast._t = setTimeout(() => { t.hidden = true; }, isErr ? 5000 : 2200);
  };

  // Per-request-family abort controllers: a new list fetch cancels the
  // previous one, a new detail fetch cancels the previous detail fetch.
  const inflight = {};
  async function api(family, url, opts = {}) {
    if (inflight[family]) inflight[family].abort();
    const ctl = new AbortController();
    inflight[family] = ctl;
    try {
      const r = await fetch(url, { ...opts, signal: ctl.signal });
      const body = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(body.error || `${r.status} ${r.statusText}`);
      return body;
    } finally {
      if (inflight[family] === ctl) delete inflight[family];
    }
  }
  const isAbort = (e) => e && e.name === "AbortError";
  // Escape, then turn FTS excerpt markers (>>>match<<<) into <mark>.
  const escMarked = (s) => esc(s).replace(/&gt;&gt;&gt;([\s\S]+?)&lt;&lt;&lt;/g, "<mark>$1</mark>");
  // Tokenize a filter query the way the server does (db/web.rs
  // filter_terms): bare words are terms, double-quoted runs are phrase
  // terms, an unterminated quote takes the rest as a phrase.
  const filterTerms = (q) => {
    const terms = [];
    let cur = "", inPhrase = false;
    for (const c of q) {
      if (c === '"') {
        if (cur.trim()) terms.push(cur.trim());
        cur = ""; inPhrase = !inPhrase;
      } else if (/\s/.test(c) && !inPhrase) {
        if (cur) { terms.push(cur); cur = ""; }
      } else cur += c;
    }
    if (cur.trim()) terms.push(cur.trim());
    return terms;
  };
  // Escape, then <mark> every occurrence of every filter term. Both sides
  // are escaped identically, so matching on the escaped strings is sound.
  const escHl = (text, needle) => {
    const escaped = esc(text);
    const terms = filterTerms(needle || "");
    if (!terms.length) return escaped;
    const pat = terms
      .map((t) => esc(t).replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
      .join("|");
    return escaped.replace(new RegExp(pat, "gi"), (m) => `<mark>${m}</mark>`);
  };

  // ---- minimal safe markdown --------------------------------------------
  // Handles what ekko's summaries and snapshots actually use: headings,
  // bullets, numbered lists, fenced code, inline code, bold, links, tables,
  // blockquotes, "Label: text" lead-ins. Input is escaped first; the
  // patterns below only ever emit fixed tags.
  function inline(s) {
    return s
      .replace(/`([^`]+)`/g, (_, c) => `<code>${c}</code>`)
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[\s(])\*([^*\s][^*]*?)\*(?=[\s).,;:!?]|$)/g, "$1<em>$2</em>")
      .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  }
  function markdown(src) {
    const lines = esc(src || "").replace(/\r\n?/g, "\n").split("\n");
    const out = [];
    let i = 0, list = null, para = [];
    const flushPara = () => { if (para.length) { out.push(`<p>${inline(para.join(" "))}</p>`); para = []; } };
    const closeList = () => { if (list) { out.push(`</${list}>`); list = null; } };
    while (i < lines.length) {
      let l = lines[i];
      if (/^```/.test(l)) {
        flushPara(); closeList();
        const buf = []; i++;
        while (i < lines.length && !/^```/.test(lines[i])) buf.push(lines[i++]);
        i++;
        out.push(`<pre><code>${buf.join("\n")}</code></pre>`);
        continue;
      }
      const h = /^(#{1,6})\s+(.*)$/.exec(l);
      if (h) { flushPara(); closeList(); const n = Math.min(h[1].length, 3); out.push(`<h${n}>${inline(h[2])}</h${n}>`); i++; continue; }
      if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(l)) { flushPara(); closeList(); out.push("<hr>"); i++; continue; }
      if (/^\|/.test(l) && i + 1 < lines.length && /^\|?\s*:?-+/.test(lines[i + 1])) {
        flushPara(); closeList();
        const cells = (r) => r.replace(/^\||\|$/g, "").split("|").map((c) => inline(c.trim()));
        const head = cells(l); i += 2;
        const rows = [];
        while (i < lines.length && /^\|/.test(lines[i])) rows.push(cells(lines[i++]));
        out.push(`<table><thead><tr>${head.map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>${
          rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table>`);
        continue;
      }
      const bq = /^&gt;\s?(.*)$/.exec(l);
      if (bq) { flushPara(); closeList(); out.push(`<blockquote>${inline(bq[1])}</blockquote>`); i++; continue; }
      const ul = /^\s*[-*•]\s+(.*)$/.exec(l);
      const ol = /^\s*\d+[.)]\s+(.*)$/.exec(l);
      if (ul || ol) {
        flushPara();
        const kind = ul ? "ul" : "ol";
        if (list !== kind) { closeList(); out.push(`<${kind}>`); list = kind; }
        out.push(`<li>${inline((ul || ol)[1])}</li>`); i++; continue;
      }
      if (l.trim() === "") { flushPara(); closeList(); i++; continue; }
      const label = /^([A-Z][A-Za-z -]{1,24}):\s+(.*)$/.exec(l);
      if (label && !para.length) { closeList(); out.push(`<p><span class="label">${label[1]}:</span> ${inline(label[2])}</p>`); i++; continue; }
      closeList(); para.push(l); i++;
    }
    flushPara(); closeList();
    return `<div class="md">${out.join("\n")}</div>`;
  }

  // ---- state --------------------------------------------------------------
  const PAGE = 50;
  const state = {
    q: "", mode: "filter", project: "", source: "", days: "", sort: "",
    offset: 0, total: 0, sessions: [], selected: null,
    // Keyset paging: the server hands back next_cursor; Prev replays the
    // stack of cursors that led here (null = first page).
    cursor: null, cursorStack: [], nextCursor: null,
    detailCache: new Map(), transcriptCache: new Map(),
    tab: "summary", warnedMode: null, tUpTo: null,
  };

  // Filters and selection live in the hash (#q=…&project=…&s=…) so a view
  // is bookmarkable and survives reload. Legacy "#s=<id>" still parses.
  const HASH_KEYS = ["q", "mode", "project", "source", "days", "sort", "s"];
  function readHash() {
    const h = location.hash.replace(/^#/, "");
    if (!h) return {};
    if (!h.includes("=")) return {};
    const p = new URLSearchParams(h);
    const out = {};
    for (const k of HASH_KEYS) if (p.has(k)) out[k] = p.get(k);
    return out;
  }
  function writeHash() {
    const p = new URLSearchParams();
    if (state.q.trim()) p.set("q", state.q.trim());
    if (state.mode !== "filter") p.set("mode", state.mode);
    if (state.project) p.set("project", state.project);
    if (state.source) p.set("source", state.source);
    if (state.days) p.set("days", state.days);
    if (state.sort) p.set("sort", state.sort);
    if (state.selected) p.set("s", state.selected);
    const h = p.toString();
    history.replaceState(null, "", h ? `#${h}` : location.pathname);
  }
  const cacheSet = (map, key, val, max) => { map.set(key, val); if (map.size > max) map.delete(map.keys().next().value); };

  // ---- list ---------------------------------------------------------------
  const listEl = $("#session-list");
  const setSeg = (rootSel, v) =>
    $$(rootSel + " .seg-b").forEach((b) => b.classList.toggle("active", b.dataset.v === v));
  // Meta text + sort visibility (sort is ignored by relevance-ranked search
  // modes, so the control hides there rather than lying).
  const setMeta = (html) => {
    $("#list-meta-text").innerHTML = html;
    $(".select.sort").hidden = state.mode !== "filter";
  };
  // Calendar bucket for the sidebar's date groups.
  function dateGroup(iso) {
    if (!iso) return "Older";
    const d = new Date(iso);
    if (isNaN(d)) return "Older";
    const now = new Date();
    const midnight = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const day = 86_400_000;
    if (d >= midnight) return "Today";
    if (d >= new Date(midnight - day)) return "Yesterday";
    if (d >= new Date(midnight - 6 * day)) return "This week";
    if (d >= new Date(midnight - 29 * day)) return "This month";
    return "Older";
  }

  function renderList() {
    listEl.innerHTML = "";
    if (!state.sessions.length) {
      listEl.appendChild(el(`<li class="muted" style="cursor:default">No sessions match.</li>`));
    }
    // Group headers only where the list is in date order: filter mode,
    // sorted by modified (default) or created. Relevance-ordered search
    // results and size sorts stay flat.
    const grouping = state.mode === "filter" && (state.sort === "" || state.sort === "created");
    const dateOf = (s) => (state.sort === "created" ? s.created_at : s.modified_at);
    const needle = state.mode === "filter" ? state.q.trim() : "";
    let lastGroup = null;
    for (const s of state.sessions) {
      if (grouping) {
        const g = dateGroup(dateOf(s));
        if (g !== lastGroup) {
          lastGroup = g;
          listEl.appendChild(el(`<li class="group" aria-hidden="true">${esc(g)}</li>`));
        }
      }
      const li = el(`<li role="option" data-id="${esc(s.session_id)}" class="${s.session_id === state.selected ? "active" : ""}">
        <div class="s-title">${escHl(s.title || s.first_prompt || "(untitled)", needle)}</div>
        <div class="s-desc">${needle ? escHl(s.short_desc || "", needle) : escMarked(s.short_desc || s.excerpt || s.relevance || "")}</div>
        <div class="s-meta">
          <span class="pill src-${esc(s.cli_source)}">${esc(srcLabel[s.cli_source] || s.cli_source)}</span>
          <span class="proj" title="${esc(s.project_path || "")}">${esc(projName(s.project_path))}</span>
          <span>${esc(fmtDate(dateOf(s) || s.modified_at))}</span>
          ${state.sort === "duration" && s.duration_ms ? `<span>${esc(fmtDuration(s.duration_ms))}</span>` : ""}
          ${s.message_count != null ? `<span>${s.message_count} msgs</span>` : ""}
          ${Number.isFinite(Number(s.score)) && s.score != null ? `<span title="relevance">★ ${Number(s.score).toFixed(2)}</span>` : ""}
        </div></li>`);
      li.addEventListener("click", () => select(s.session_id));
      listEl.appendChild(li);
    }
    const from = state.total ? state.offset + 1 : 0;
    const to = Math.min(state.offset + state.sessions.length, state.total);
    setMeta(state.mode === "filter"
      ? `${from}–${to} of ${state.total}${state.q ? " · filtered" : ""}`
      : `${state.sessions.length} results · ${esc(state.mode)}${state.tiers ? " · " + esc(state.tiers.join("→")) : ""}`);
    renderPager();
  }
  const projName = (p) => (p || "").split(/[\\/]/).filter(Boolean).pop() || p || "";
  function pageNext() {
    if (!state.nextCursor) return Promise.resolve();
    state.cursorStack.push(state.cursor);
    state.cursor = state.nextCursor;
    return loadList();
  }
  function pagePrev() {
    if (!state.cursorStack.length) return Promise.resolve();
    state.cursor = state.cursorStack.pop();
    return loadList();
  }
  function renderPager() {
    const pg = $("#pager");
    pg.innerHTML = "";
    if (state.mode !== "filter" || state.total <= PAGE) return;
    const pages = Math.ceil(state.total / PAGE), cur = state.cursorStack.length + 1;
    const b = (label, fn, dis) => { const x = el(`<button class="btn ghost" ${dis ? "disabled" : ""}>${label}</button>`); x.onclick = fn; return x; };
    pg.appendChild(b("‹", pagePrev, !state.cursorStack.length));
    pg.appendChild(el(`<span>page ${cur} / ${pages}</span>`));
    pg.appendChild(b("›", pageNext, !state.nextCursor));
  }

  async function loadList() {
    setMeta(`<span class="spinner"></span>loading…`);
    try {
      if (state.mode === "filter" || !state.q.trim()) {
        const p = new URLSearchParams({ limit: PAGE });
        if (state.cursor) p.set("cursor", state.cursor);
        if (state.sort) p.set("sort", state.sort);
        if (state.q.trim()) p.set("q", state.q.trim());
        if (state.project) p.set("project", state.project);
        if (state.source) p.set("source", state.source);
        if (state.days) p.set("days", state.days);
        const r = await api("list", `/api/sessions?${p}`);
        state.sessions = r.sessions; state.total = r.total; state.tiers = null;
        state.nextCursor = r.next_cursor || null;
        state.offset = PAGE * state.cursorStack.length;
        fillSources(r.source_counts);
      } else {
        const p = new URLSearchParams({ q: state.q.trim(), mode: state.mode, limit: 50 });
        if (state.project) p.set("directory", state.project);
        if (state.days) p.set("days", state.days);
        const r = await api("list", `/api/search?${p}`);
        state.sessions = (r.results || []).map((x) => ({ ...x, short_desc: x.excerpt || x.relevance || x.summary }));
        state.total = state.sessions.length; state.tiers = r.tiers_used || null;
        if (r.requested_mode !== r.mode && state.warnedMode !== state.mode) {
          state.warnedMode = state.mode;
          toast(`${r.requested_mode} unavailable (no LLM backend); used ${r.mode}`);
        }
        if (state.source) state.sessions = state.sessions.filter((s) => s.cli_source === state.source);
      }
      writeHash();
      renderList();
    } catch (e) {
      if (isAbort(e)) return;
      setMeta(`<span class="warn">${esc(e.message)}</span>`);
      toast(e.message, true);
    }
  }

  function fillSources(counts) {
    for (const b of $$("#f-source .seg-b")) {
      if (!b.dataset.v) continue;
      const n = (counts || {})[b.dataset.v] || 0;
      let ct = b.querySelector(".ct");
      if (!ct) { ct = document.createElement("span"); ct.className = "ct"; b.appendChild(ct); }
      ct.textContent = n || "";
      b.classList.toggle("zero", !n);
    }
  }
  function loadProjects() {
    return (async () => {
    try {
      const r = await api("projects", "/api/projects");
      const sel = $("#f-project");
      sel.innerHTML = `<option value="">All projects</option>`;
      for (const p of r.projects) sel.appendChild(el(`<option value="${esc(p.path)}" title="${esc(p.path)}">${esc(p.name)} (${p.count})</option>`));
      sel.value = state.project;
    } catch (e) { if (!isAbort(e)) toast(e.message, true); }
    })();
  }

  // ---- detail -------------------------------------------------------------
  async function select(id) {
    state.selected = id;
    markActive();
    $(".layout").classList.add("showing-detail");
    $("#empty-state").hidden = true;
    $("#session").hidden = false;
    let d = state.detailCache.get(id);
    if (!d) {
      $("#s-title").innerHTML = `<span class="spinner"></span>Loading…`;
      try { d = await api("detail", `/api/sessions/${encodeURIComponent(id)}`); }
      catch (e) { if (isAbort(e)) return; $("#s-title").textContent = e.message; return; }
      cacheSet(state.detailCache, d.session_id, d, 40);
    }
    if (state.selected !== id) return;
    // The server resolves prefixes: from here on the full id is the truth,
    // or the transcript tab / list highlight / j-k would compare a prefix
    // against full ids forever.
    if (d.session_id && d.session_id !== id) {
      state.selected = d.session_id;
      markActive();
    }
    state.tUpTo = null;
    writeHash();
    renderDetail(d);
  }
  function markActive() {
    let seen = false;
    $$("#session-list li").forEach((li) => {
      const on = li.dataset.id === state.selected;
      li.classList.toggle("active", on);
      if (on && !seen) { seen = true; li.scrollIntoView({ block: "nearest" }); }
    });
  }

  function backToList() {
    $(".layout").classList.remove("showing-detail");
  }

  function renderDetail(d) {
    const m = d.metadata || {};
    $("#s-title").textContent = m.title || d.first_prompt || "(untitled)";
    const meta = [];
    meta.push(`<span class="pill src-${esc(m.cli_source)}">${esc(srcLabel[m.cli_source] || m.cli_source)}</span>`);
    if (m.project_path) meta.push(`<span class="mono copy" data-copy="${esc(m.project_path)}" title="click to copy project path">${esc(m.project_path)}</span>`);
    if (m.git_branch) meta.push(`<span title="branch">⎇ ${esc(m.git_branch)}</span>`);
    if (m.created_at) meta.push(`<span title="${esc(m.created_at)}">${esc(fmtDate(m.created_at))}</span>`);
    if (d.duration_ms) meta.push(`<span>${esc(fmtDuration(d.duration_ms))}</span>`);
    meta.push(`<span>${m.message_count ?? 0} messages</span>`);
    if (d.token_count) meta.push(`<span>${Number(d.token_count).toLocaleString()} tokens</span>`);
    if (d.model) meta.push(`<span>${esc(d.model)}</span>`);
    meta.push(`<span class="mono copy" title="click to copy full id">${esc(shortId(d.session_id))}…</span>`);
    $("#s-meta").innerHTML = meta.join("");
    $$("#s-meta .copy").forEach((elx) => {
      elx.onclick = () => {
        const text = elx.dataset.copy || d.session_id;
        navigator.clipboard.writeText(text).then(() => toast(elx.dataset.copy ? "path copied" : "session id copied"));
      };
    });
    $("#s-short").textContent = d.short_desc || "";

    const files = d.files_touched || [];
    $(`.tab[data-tab="files"]`).innerHTML = `Files <span class="count">${files.length}</span>`;
    $(`.tab[data-tab="snapshot"]`).innerHTML = `Snapshot${d.has_snapshot ? "" : ' <span class="count">—</span>'}`;
    $(`.tab[data-tab="transcript"]`).innerHTML = `Transcript <span class="count">${d.event_count ?? ""}</span>`;

    const sum = $(`.panel[data-panel="summary"]`);
    sum.innerHTML = d.summary ? markdown(d.summary) : `<p class="none">No summary yet — the indexer hasn't reached this session (or it's still being written).</p>`;
    const kw = parseKeywords(d.keywords);
    if (kw.length) sum.appendChild(el(`<p class="kw">${kw.map((k) => `<span class="pill">${esc(k)}</span>`).join(" ")}</p>`));
    if (d.first_prompt) sum.appendChild(el(`<details><summary class="muted">First prompt</summary><pre>${esc(d.first_prompt)}</pre></details>`));

    const snap = $(`.panel[data-panel="snapshot"]`);
    snap.innerHTML = d.has_snapshot ? markdown(d.snapshot) : `<p class="none">${esc(d.suggestion || "No snapshot yet.")}</p>`;

    const fp = $(`.panel[data-panel="files"]`);
    fp.innerHTML = files.length
      ? `<table class="files-table">${files.map((f) => `<tr><td class="op ${esc(f.operation)}">${esc(f.operation)}</td><td>${esc(f.path)}</td><td class="acts"><button class="act" data-copy="${esc(f.path)}" title="copy path">copy</button><a class="act" href="vscode://file${esc(f.path)}" title="open in VS Code">open</a></td></tr>`).join("")}</table>`
      : `<p class="none">No file operations recorded.</p>`;
    $$(".files-table .act[data-copy]", fp).forEach((b) => {
      b.onclick = () => navigator.clipboard.writeText(b.dataset.copy).then(() => toast("path copied"));
    });

    $("#t-body").innerHTML = "";
    $("#t-count").textContent = "";
    setTab(state.tab, d);
  }

  // Keywords are stored as a JSON array string; tolerate comma lists too.
  function parseKeywords(k) {
    if (!k) return [];
    if (Array.isArray(k)) return k;
    try { const v = JSON.parse(k); if (Array.isArray(v)) return v.map(String); } catch {}
    return String(k).split(",").map((x) => x.trim()).filter(Boolean);
  }

  function setTab(tab, d) {
    state.tab = tab;
    $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === tab));
    $$(".panel").forEach((p) => { p.hidden = p.dataset.panel !== tab; });
    if (tab === "transcript") loadTranscript(d || state.detailCache.get(state.selected));
  }

  // ---- transcript ---------------------------------------------------------
  const T_CHUNK = 150;
  async function loadTranscript(d) {
    if (!d) return;
    const body = $("#t-body");
    let t = state.transcriptCache.get(d.session_id);
    if (!t) {
      if (d.transcript_exists === false) { body.innerHTML = `<p class="none">Transcript is not in the archive and the file is no longer on disk.</p>`; return; }
      body.innerHTML = `<p class="muted"><span class="spinner"></span>Loading transcript…</p>`;
      try { t = await api("transcript", `/api/sessions/${encodeURIComponent(d.session_id)}/transcript`); }
      catch (e) { if (isAbort(e)) return; body.innerHTML = `<p class="none">${esc(e.message)}</p>`; return; }
      cacheSet(state.transcriptCache, d.session_id, t, 6);
    }
    if (state.selected !== d.session_id) return;
    renderTranscript(t);
  }
  function visibleEvents(t) {
    const showTools = $("#t-show-tools").checked, showResults = $("#t-show-results").checked;
    return t.events.filter((e) =>
      e.type === "message" ? true
      : e.type === "tool_call" ? showTools
      : e.type === "tool_result" ? showResults
      : e.type === "artifact" || e.type === "error");
  }
  function renderTranscript(t, upTo = state.tUpTo || T_CHUNK, keepScroll = false) {
    state.tUpTo = upTo;
    const body = $("#t-body");
    const scroll = keepScroll ? $(".detail").scrollTop : null;
    const evs = visibleEvents(t);
    $("#t-count").textContent = `${evs.length} of ${t.event_count} events`;
    body.innerHTML = "";
    if (!t.event_count) {
      body.innerHTML = `<p class="none">No events could be parsed from this transcript.</p>`;
      return;
    }
    const frag = document.createDocumentFragment();
    for (const e of evs.slice(0, upTo)) frag.appendChild(eventNode(e));
    body.appendChild(frag);
    if (evs.length > upTo) {
      const more = el(`<div class="load-more"><button class="btn">Show ${Math.min(T_CHUNK, evs.length - upTo)} more</button> <button class="btn ghost">Show all ${evs.length} (enables ⌘F)</button></div>`);
      const [showMore, showAll] = more.querySelectorAll("button");
      showMore.onclick = () => renderTranscript(t, upTo + T_CHUNK, true);
      showAll.onclick = () => renderTranscript(t, evs.length, true);
      body.appendChild(more);
    }
    if (scroll != null) $(".detail").scrollTop = scroll;
  }
  function eventNode(e) {
    const who = e.type === "message" ? (e.role || "message")
      : e.type === "tool_call" ? `⚙ ${e.tool_name || "tool"}`
      : e.type === "tool_result" ? `↳ ${e.tool_success === false ? "failed" : "result"}`
      : e.type;
    const cls = e.type === "message" ? (e.role || "system") : e.type;
    const fail = e.type === "tool_result" && e.tool_success === false ? " fail" : "";
    const ts = e.timestamp ? `<span title="${esc(e.timestamp)}">${esc(fmtDate(e.timestamp))}</span>` : "";
    let bodyHtml;
    if (e.type === "tool_call") {
      bodyHtml = esc(toolInputText(e.tool_name, e.tool_input));
    } else if (e.type === "tool_result") {
      bodyHtml = esc(e.tool_output || e.content || "");
    } else if (e.type === "message" && e.role === "assistant") {
      bodyHtml = markdown(e.content);
    } else {
      bodyHtml = esc(e.content || e.artifact_path || "");
    }
    const cut = e.truncated
      ? `<div class="ev-body trunc">… truncated at ingest${e.full_len ? ` (${Number(e.full_len).toLocaleString()} chars originally)` : ""}</div>`
      : "";
    const n = el(`<div class="ev ${cls}${fail}"><div class="ev-head"><span class="who">${esc(who)}</span>${ts}<span class="grow"></span><span class="muted">#${e.sequence}</span></div><div class="ev-body"></div>${cut}</div>`);
    n.querySelector(".ev-body").innerHTML = bodyHtml;
    return n;
  }
  function toolInputText(name, input) {
    if (input == null) return "";
    if (typeof input === "string") return input;
    // Common shapes first so Bash/Read/Edit read like a shell log.
    if (input.command && Object.keys(input).length <= 3) return `$ ${input.command}${input.description ? `   # ${input.description}` : ""}`;
    if (input.file_path && !input.old_string && Object.keys(input).length <= 3) return input.file_path;
    if (input.old_string != null && input.new_string != null) return `${input.file_path || ""}\n--- old\n${input.old_string}\n+++ new\n${input.new_string}`;
    try { return JSON.stringify(input, null, 2); } catch { return String(input); }
  }

  // ---- ask ----------------------------------------------------------------
  async function ask() {
    const q = $("#ask-q").value.trim();
    if (!q) return;
    const out = $("#ask-out"), btn = $("#ask-go");
    btn.disabled = true;
    out.innerHTML = `<p class="muted"><span class="spinner"></span>Thinking — this reads summaries first, then transcripts if needed…</p>`;
    try {
      const body = { question: q };
      const dir = $("#ask-dir").value.trim(); if (dir) body.directory = dir;
      const days = $("#ask-days").value; if (days) body.days = Number(days);
      const r = await api("ask", "/api/ask", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
      const sources = (r.sources || []).map((s) => {
        const id = s.session_id || "";
        return `<li><a href="#s=${esc(id)}" data-id="${esc(id)}">${esc(s.title || shortId(id))}</a>${s.relevance ? ` — <span class="muted">${esc(s.relevance)}</span>` : ""}</li>`;
      });
      out.innerHTML = `${markdown(r.answer || "")}${sources.length ? `<div class="sources"><strong>Sources</strong><ul>${sources.join("")}</ul></div>` : ""}
        <div class="usage">${esc(r.tier_used ? `tier: ${r.tier_used}` : "")}${r.usage && r.usage.cost_display ? ` · ${esc(r.usage.cost_display)}` : ""}</div>`;
      $$("#ask-out a[data-id]").forEach((a) => a.onclick = (ev) => { ev.preventDefault(); $("#dlg-ask").close(); select(a.dataset.id); });
    } catch (e) {
      if (!isAbort(e)) out.innerHTML = `<p class="warn">${esc(e.message)}</p>`;
    } finally { btn.disabled = false; }
  }

  // ---- status -------------------------------------------------------------
  async function showStatus() {
    const out = $("#status-out");
    out.innerHTML = `<span class="spinner"></span>`;
    $("#dlg-status").showModal();
    try {
      const s = await api("status", "/api/status");
      const c = s.config || {}, idx = c.index || {}, qry = c.query || {};
      const row = (k, v, warn) => `<div class="k">${esc(k)}</div><div class="${warn ? "warn" : ""}">${v}</div>`;
      out.innerHTML = `<div class="status-grid">
        ${row("version", esc(s.version))}
        ${row("database", `<span class="mono">${esc(s.database?.path)}</span> (${esc(s.database?.size_mb)} MB)`)}
        ${row("sessions", `${esc(s.sessions?.total)} total · ${esc(s.sessions?.with_summary)} summarized · ${esc(s.sessions?.with_snapshot)} with snapshot · ${esc(s.sessions?.needing_summary)} pending`)}
        ${row("indexer", `${esc(s.indexer?.status)} <span class="muted">(the MCP server indexes; this page doesn't)</span>`)}
        ${row("index backend", `${esc(idx.backend)} / ${esc(idx.model)} <span class="muted">— ${esc(idx.reason)}</span>`, idx.available === false)}
        ${row("query backend", `${esc(qry.backend)} / ${esc(qry.model)} <span class="muted">— ${esc(qry.reason)}</span>`, qry.available === false)}
        ${idx.warning ? row("", esc(idx.warning), true) : ""}
        ${row("cost (7d)", `${esc(s.costs?.last_7_days?.cost_display)} <span class="muted">${esc(idx.cost_basis === "api" ? "" : idx.cost_basis)}</span>`)}
        ${Array.isArray(s.indexer?.recent_errors) && s.indexer.recent_errors.length
          ? row("recent errors", s.indexer.recent_errors.map((x) => `<div class="mono-sm">${esc(typeof x === "string" ? x : JSON.stringify(x))}</div>`).join(""), true)
          : ""}
      </div>`;
    } catch (e) { if (!isAbort(e)) out.innerHTML = `<p class="warn">${esc(e.message)}</p>`; }
  }

  // ---- theme --------------------------------------------------------------
  function applyTheme(t) {
    if (t) document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme;
    try { t ? localStorage.setItem("ekko-theme", t) : localStorage.removeItem("ekko-theme"); } catch {}
  }
  function toggleTheme() {
    const cur = document.documentElement.dataset.theme
      || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    applyTheme(cur === "dark" ? "light" : "dark");
  }

  // ---- wiring -------------------------------------------------------------
  function init() {
    try { const t = localStorage.getItem("ekko-theme"); if (t) applyTheme(t); } catch {}
    const q = $("#q");
    const reload = () => {
      state.offset = 0; state.cursor = null; state.cursorStack = []; state.nextCursor = null;
      loadList();
    };
    // semantic/hybrid cost an LLM call per search — those run on Enter,
    // not per keystroke. filter/keyword/transcript stay as-you-type.
    const searchOnEnter = () => state.mode === "semantic" || state.mode === "hybrid";
    q.addEventListener("input", debounce(() => {
      state.q = q.value;
      if (searchOnEnter()) {
        setMeta(`<span class="muted">press Enter to run ${esc(state.mode)} search</span>`);
        return;
      }
      reload();
    }, 200));
    q.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && searchOnEnter()) { state.q = q.value; reload(); }
    });
    const setMode = (mode) => {
      state.mode = mode;
      $$("#search-mode .mode").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
      if (searchOnEnter() && state.q.trim()) {
        setMeta(`<span class="muted">press Enter to run ${esc(state.mode)} search</span>`);
        return;
      }
      reload();
    };
    $$("#search-mode .mode").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
    $("#f-project").addEventListener("change", (e) => { state.project = e.target.value; reload(); });
    $("#f-source").addEventListener("click", (e) => {
      const b = e.target.closest(".seg-b"); if (!b) return;
      state.source = b.dataset.v; setSeg("#f-source", b.dataset.v); reload();
    });
    $("#f-days").addEventListener("click", (e) => {
      const b = e.target.closest(".seg-b"); if (!b) return;
      state.days = b.dataset.v; setSeg("#f-days", b.dataset.v); reload();
    });
    $("#f-sort").addEventListener("change", (e) => { state.sort = e.target.value; reload(); });
    $$(".tab").forEach((t) => t.addEventListener("click", () => setTab(t.dataset.tab)));
    const retoggle = () => {
      const t = state.transcriptCache.get(state.selected);
      if (t) renderTranscript(t, state.tUpTo || T_CHUNK, true); else setTab("transcript");
    };
    $("#t-show-tools").addEventListener("change", retoggle);
    $("#t-show-results").addEventListener("change", retoggle);
    $("#btn-ask").onclick = () => { $("#dlg-ask").showModal(); $("#ask-q").focus(); };
    $("#ask-form").addEventListener("submit", (e) => { e.preventDefault(); ask(); });
    $("#ask-close").onclick = () => $("#dlg-ask").close();
    $("#btn-back").onclick = backToList;
    $("#ask-q").addEventListener("keydown", (e) => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter") { e.preventDefault(); ask(); } });
    $("#btn-status").onclick = showStatus;
    $("#btn-theme").onclick = toggleTheme;
    $("#btn-help").onclick = () => $("#dlg-help").showModal();

    document.addEventListener("keydown", (e) => {
      const inField = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName) || document.querySelector("dialog[open]");
      if (e.key === "Escape" && !document.querySelector("dialog[open]")) {
        document.activeElement?.blur();
        backToList();
        return;
      }
      if (inField) return;
      if (e.key === "/") { e.preventDefault(); q.focus(); q.select(); return; }
      if (e.key === "a") { e.preventDefault(); $("#btn-ask").click(); return; }
      if (e.key === "?") { $("#dlg-help").showModal(); return; }
      if (e.key === "j" || e.key === "k") {
        const ids = state.sessions.map((s) => s.session_id);
        if (!ids.length) return;
        const i = ids.indexOf(state.selected);
        const next = e.key === "j" ? i + 1 : i - 1;
        if (i === -1) { select(ids[0]); return; }
        if (next >= 0 && next < ids.length) { select(ids[next]); return; }
        // Walk across page boundaries in filter mode.
        if (state.mode === "filter") {
          if (next >= ids.length && state.nextCursor) {
            pageNext().then(() => state.sessions.length && select(state.sessions[0].session_id));
          } else if (next < 0 && state.cursorStack.length) {
            pagePrev().then(() => state.sessions.length && select(state.sessions[state.sessions.length - 1].session_id));
          }
        }
        return;
      }
      const tabs = { 1: "summary", 2: "snapshot", 3: "transcript", 4: "files" };
      if (tabs[e.key] && state.selected) setTab(tabs[e.key]);
    });

    // Deep link / bookmarked view: #q=…&project=…&s=… (legacy "#s=<id>"
    // has no other keys and still lands on the session).
    const legacy = /^#s=([\w.-]+)$/.exec(location.hash);
    const h = legacy ? { s: legacy[1] } : readHash();
    if (h.q) { state.q = h.q; q.value = h.q; }
    if (h.mode) {
      state.mode = h.mode;
      $$("#search-mode .mode").forEach((b) => b.classList.toggle("active", b.dataset.mode === h.mode));
    }
    if (h.project) { state.project = h.project; }
    if (h.source) { state.source = h.source; }
    if (h.days) { state.days = h.days; setSeg("#f-days", h.days); }
    if (h.sort) { state.sort = h.sort; $("#f-sort").value = h.sort; }
    loadProjects().then(() => { $("#f-project").value = state.project; });
    if (h.source) setSeg("#f-source", state.source);
    loadList().then(() => {
      if (h.s) select(h.s);
    });
    window.addEventListener("hashchange", () => {
      const lg = /^#s=([\w.-]+)$/.exec(location.hash);
      const hh = lg ? { s: lg[1] } : readHash();
      if (hh.s && hh.s !== state.selected) select(hh.s);
    });
    fetch("/api/status").then((r) => r.json()).then((s) => { $("#brand-sub").textContent = `${s.sessions?.total ?? ""} sessions`; }).catch(() => {});
  }
  document.addEventListener("DOMContentLoaded", init);
})();
