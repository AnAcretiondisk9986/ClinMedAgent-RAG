/* 医学教材工作台前端逻辑（无框架，直接操作 DOM）。 */
"use strict";

const $ = (selector, element = document) => element.querySelector(selector);
const $$ = (selector, element = document) => Array.from(element.querySelectorAll(selector));

const STAGE_DEFS = [
  { key: "text", label: "文字层解析", hint: (b) => (b.workflow === "text" ? "从 PDF 文字层直接解析，无需 OCR" : "该 PDF 没有可用文字层，不推荐"), done: (b) => b.workflow === "text" && b.status.structure.complete },
  { key: "ocr", label: "OCR 文字识别", hint: (b) => `已识别 ${b.status.ocr.done}/${b.status.ocr.total || "?"} 页`, done: (b) => b.status.ocr.complete },
  { key: "layout", label: "版面/表格页检测", hint: (b) => (b.status.layout.found ? `检测到 ${b.status.layout.table_pages} 个表格页` : "尚未检测"), done: (b) => b.status.layout.complete },
  { key: "tables", label: "表格结构还原", hint: (b) => (b.status.layout.found ? `已还原 ${b.status.tables.done}/${b.status.tables.total} 个表格页` : "需要先完成版面检测"), done: (b) => b.status.tables.complete },
  { key: "fix", label: "OCR 定向纠错", hint: () => "修正聘→腭、挠→桡等无歧义错字（可重复运行）", done: () => false },
  { key: "structure", label: "版面结构化", hint: (b) => (b.status.structure.found ? `已生成 ${b.status.structure.chapters} 章` : "尚未结构化"), done: (b) => b.status.structure.complete },
  { key: "index", label: "建立检索索引", hint: (b) => (b.status.index.found ? `索引 ${b.status.index.chunks} 个证据块` : "尚未建立索引"), done: (b) => b.status.index.complete },
];

const STAGE_LABELS = Object.fromEntries(STAGE_DEFS.map((stage) => [stage.key, stage.label]));

function stageDone(book, key) {
  const definition = STAGE_DEFS.find((stage) => stage.key === key);
  return definition ? definition.done(book) : false;
}

/** “全部完成”只看该工作流必经的阶段（fix 是可选的轻量阶段，不计入）。 */
function requiredStages(book) {
  return book.workflow === "text" ? ["text", "index"] : ["ocr", "layout", "tables", "structure", "index"];
}

/** 处理弹窗展示的阶段：文字层工作流只有 解析 + 索引；OCR 工作流保留全部 6 个阶段。 */
function workflowStageDefs(book) {
  const keys = book.workflow === "text" ? ["text", "index"] : ["ocr", "layout", "tables", "fix", "structure", "index"];
  return STAGE_DEFS.filter((stage) => keys.includes(stage.key));
}

const state = {
  books: [],
  tasks: [],
  root: "",
  selected: null,
  page: 1,
  offset: 0,
  totalPdfPages: 0,
  totalPrintedPages: 0,
  zoomFactor: 1,
  pageImageWidth: 0,
  pendingScroll: null,
  taskPanelOpen: false,
  expandedTask: null,
  logCursor: {},
  knownStatus: {},
  uploadFile: null,
  textRequest: 0,
  taskEls: new Map(),
};

/* ------------------------------------------------------------------ 工具 */

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new Error(`HTTP ${response.status}`);
  }
  if (!response.ok || payload.ok === false) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload.data;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
}

function formatBytes(size) {
  const value = Number(size) || 0;
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  if (value < 1024 * 1024 * 1024) return `${(value / 1024 / 1024).toFixed(1)} MB`;
  return `${(value / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

function clamp(value, low, high) {
  return Math.min(Math.max(value, low), high);
}

function coverPlaceholder(text) {
  const div = document.createElement("div");
  div.className = "cover-placeholder";
  div.textContent = (text || "医").trim().charAt(0) || "医";
  return div;
}

let toastTimer = null;
function toast(message, type = "") {
  const element = $("#toast");
  element.textContent = message;
  element.className = "toast" + (type ? ` ${type}` : "");
  element.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { element.hidden = true; }, 3800);
}

async function copyText(text, message) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const area = document.createElement("textarea");
    area.value = text;
    document.body.appendChild(area);
    area.select();
    document.execCommand("copy");
    area.remove();
  }
  toast(message || "已复制", "ok");
}

/* ------------------------------------------------------------- 书库列表 */

function isBookRunning(bookId) {
  return state.tasks.some((task) => (task.status === "running" || task.status === "pending") && task.meta && task.meta.book_id === bookId);
}

function bookProgress(book) {
  const status = book.status;
  const flags = [
    status.pdf.found,
    status.ocr.complete,
    status.layout.complete,
    status.tables.complete,
    status.structure.complete,
    status.index.complete,
  ];
  return Math.round((flags.filter(Boolean).length / flags.length) * 100);
}

function bookBadges(book, running) {
  if (running) return `<span class="badge run">处理中</span>`;
  const chunks = book.status.index.chunks;
  if (chunks > 0) return `<span class="badge ok">已索引</span>`;
  if (book.status.structure.complete) return `<span class="badge warn">待建索引</span>`;
  if (book.status.pdf.found) return `<span class="badge warn">${book.workflow === "text" ? "可直接入库" : "待处理"}</span>`;
  return `<span class="badge plain">无 PDF</span>`;
}

function renderBookList() {
  const list = $("#book-list");
  $("#book-count").textContent = String(state.books.length);
  list.innerHTML = "";
  if (!state.books.length) {
    const hint = state.root ? `<br>也可以把 PDF 放到 <code>${escapeHtml(state.root)}/res/&lt;书名&gt;/PDF/</code> 下再刷新。` : "";
    list.innerHTML = `<div class="muted" style="padding:16px">书库还是空的，点右上角“导入教材”。${hint}</div>`;
    return;
  }
  for (const book of state.books) {
    const card = document.createElement("div");
    card.className = "book-card" + (state.selected && state.selected.id === book.id ? " active" : "");
    const pages = book.status.pdf.pages || book.quality.pages || 0;
    const chunks = book.status.index.chunks || 0;
    card.innerHTML = `
      <div class="book-info">
        <div class="book-name" title="${escapeHtml(book.index_title || book.title)}">${escapeHtml(book.title)}</div>
        <div class="book-sub">${pages ? `${pages} 页` : "页数未知"}${chunks ? ` · ${chunks} 块` : ""}</div>
        <div class="book-badges">${bookBadges(book, isBookRunning(book.id))}</div>
        <div class="mini-progress"><div style="width:${bookProgress(book)}%"></div></div>
      </div>`;
    if (book.pdf) {
      const img = document.createElement("img");
      img.className = "book-cover";
      img.loading = "lazy";
      img.alt = "";
      img.src = `/api/books/${book.id}/cover.png?w=160&v=${book.status.pdf.mtime || ""}`;
      img.addEventListener("error", () => img.replaceWith(coverPlaceholder(book.title)));
      card.prepend(img);
    } else {
      card.prepend(coverPlaceholder(book.title));
    }
    card.addEventListener("click", () => selectBook(book));
    list.appendChild(card);
  }
}

function renderWelcomeStats(data) {
  $("#welcome-stats").innerHTML = `
    <div class="stat"><b>${data.totals.books}</b><span>本教材</span></div>
    <div class="stat"><b>${data.totals.indexed}</b><span>已索引</span></div>
    <div class="stat"><b>${data.totals.chunks}</b><span>证据块</span></div>`;
  const hasBooks = data.totals.books > 0;
  $("#btn-welcome-open").hidden = !hasBooks;
  const importButton = $("#btn-welcome-import");
  importButton.textContent = hasBooks ? "导入教材" : "导入第一本教材";
  importButton.classList.toggle("primary", !hasBooks);
}

function renderEnv(data) {
  const interpreters = data.interpreters;
  $("#root-path").textContent = `${data.root}${data.port ? ` · 端口 ${data.port}` : ""}`;
  $("#env-info").innerHTML = `
    <div class="env-line"><span>索引块</span><span>${data.totals.chunks}</span></div>
    <div class="env-line"><span>OCR 环境</span><span class="${interpreters.venv_ocr ? "ok" : "bad"}">${interpreters.venv_ocr ? ".venv-ocr" : "主环境"}</span></div>
    <div class="env-line"><span>Paddle 环境</span><span class="${interpreters.venv_ocr312 ? "ok" : "bad"}">${interpreters.venv_ocr312 ? ".venv-ocr312" : "主环境"}</span></div>
    <div class="env-line" title="${escapeHtml(interpreters.ocr_python)}"><span>OCR Python</span><span>${escapeHtml(interpreters.ocr_python)}</span></div>`;
}

/* ------------------------------------------------------------- 教材详情 */

function statusChips(book) {
  const status = book.status;
  const chip = (name, value, kind) => `<div class="status-chip ${kind}"><span class="dot"></span><span class="name">${name}</span><span class="value">${value}</span></div>`;
  const chips = [];
  const kind = status.text && status.text.kind;
  const kindLabels = { text: "文字层", scan: "图片型", mixed: "混合型" };
  let pdfValue = status.pdf.found ? `${status.pdf.pages ?? "?"} 页` : "缺失";
  if (status.pdf.found && kind) {
    pdfValue += ` · ${kindLabels[kind] || kind}`;
    const chars = (status.text && status.text.chars) || 0;
    if (kind === "text" && chars) {
      pdfValue += `（${chars >= 10000 ? `${(chars / 10000).toFixed(1)} 万字` : `${chars} 字`}）`;
    }
  }
  chips.push(chip("原始 PDF", pdfValue, status.pdf.found ? "done" : "missing"));
  chips.push(chip("OCR 识别", status.ocr.total ? `${status.ocr.done}/${status.ocr.total}` : `${status.ocr.done} 页`, status.ocr.complete ? "done" : status.ocr.done ? "running" : "missing"));
  chips.push(chip("版面检测", status.layout.found ? `${status.layout.table_pages} 个表格页` : "未完成", status.layout.complete ? "done" : "missing"));
  chips.push(chip("表格还原", status.tables.total ? `${status.tables.done}/${status.tables.total}` : status.layout.found ? "无表格" : "未完成", status.tables.complete ? "done" : "missing"));
  chips.push(chip("版面结构化", status.structure.found ? `${status.structure.chapters} 章 · 偏移 ${status.structure.page_offset ?? 0}` : "未完成", status.structure.complete ? "done" : "missing"));
  chips.push(chip("检索索引", status.index.found ? `${status.index.chunks} 块 / ${status.index.chunk_pages} 页` : "未建立", status.index.complete ? "done" : "missing"));
  return chips.join("");
}

function renderDetail(book) {
  $("#detail-title").textContent = book.title;
  const parts = [];
  if (book.index_title && book.index_title !== book.title) parts.push(`索引：${book.index_title}`);
  parts.push(book.rel_dir);
  if (book.status.pdf.size) parts.push(formatBytes(book.status.pdf.size));
  $("#detail-subtitle").textContent = parts.join(" · ");
  $("#status-grid").innerHTML = statusChips(book);

  const cover = $("#detail-cover");
  if (book.pdf) {
    cover.hidden = false;
    cover.src = `/api/books/${book.id}/cover.png?w=480&v=${book.status.pdf.mtime || ""}`;
  } else {
    cover.hidden = true;
  }
  const allComplete = Boolean(book.pdf) && requiredStages(book).every((key) => stageDone(book, key));
  $("#btn-process").disabled = !book.pdf;
  $("#btn-process").textContent = allComplete ? "重新处理" : "处理教材";
  $("#detail-hint").textContent = allComplete
    ? "全部处理阶段已完成，一般无需再次处理。"
    : book.workflow === "text"
      ? "检测到文字层 PDF，可直接解析入库（几秒完成）。"
      : "";
  $("#btn-reindex").disabled = !book.status.structure.complete;
  $(".preview").hidden = !book.pdf;
  $("#input-printed").disabled = !state.offset;
  $("#printed-total").textContent = state.totalPrintedPages ? `/ ${state.totalPrintedPages}` : "/ —";
  $("#pdf-total").textContent = state.totalPdfPages ? `/ ${state.totalPdfPages}` : "/ —";
}

function showDetailView() {
  $("#welcome").hidden = true;
  $("#search-results").hidden = true;
  $("#detail").hidden = false;
}

function showWelcome() {
  state.selected = null;
  state.page = 1;
  $("#detail").hidden = true;
  $("#search-results").hidden = true;
  $("#welcome").hidden = false;
  renderBookList();
}

async function selectBook(book, options = {}) {
  state.selected = book;
  state.zoomFactor = 1;
  state.pendingScroll = null;
  state.page = options.keepPage ? state.page : 0;
  state.offset = Number(book.quality && book.quality.page_offset) || 0;
  showDetailView();
  renderBookList();
  renderDetail(book);
  try {
    const fresh = await api(`/api/books/${book.id}`);
    if (!state.selected || state.selected.id !== fresh.id) return;
    state.selected = fresh;
    state.offset = Number(fresh.quality && fresh.quality.page_offset) || 0;
    state.totalPdfPages = fresh.status.pdf.pages || fresh.quality.pages || 0;
    state.totalPrintedPages = state.offset ? Math.max(0, state.totalPdfPages - state.offset) : 0;
    if (!state.page) {
      const min = fresh.page_range && fresh.page_range.min;
      state.page = min ? min + state.offset : 1;
    }
    state.page = clamp(state.page, 1, state.totalPdfPages || 1);
    renderDetail(fresh);
    renderBookList();
    loadPage();
  } catch (error) {
    toast(`读取教材详情失败：${error.message}`, "error");
  }
}

async function refreshBooks() {
  try {
    const books = await api("/api/books");
    state.books = books;
    renderSearchScope();
    if (state.selected) {
      const fresh = books.find((item) => item.id === state.selected.id);
      if (fresh) {
        state.selected = fresh;
        state.totalPdfPages = fresh.status.pdf.pages || fresh.quality.pages || 0;
        state.offset = Number(fresh.quality && fresh.quality.page_offset) || 0;
        state.totalPrintedPages = state.offset ? Math.max(0, state.totalPdfPages - state.offset) : 0;
        if (!$("#detail").hidden) renderDetail(fresh);
      }
    }
    renderBookList();
  } catch {
    /* 静默重试 */
  }
}

async function refreshDetail() {
  if (!state.selected) return;
  try {
    const fresh = await api(`/api/books/${state.selected.id}`);
    state.selected = fresh;
    state.offset = Number(fresh.quality && fresh.quality.page_offset) || 0;
    state.totalPdfPages = fresh.status.pdf.pages || fresh.quality.pages || 0;
    state.totalPrintedPages = state.offset ? Math.max(0, state.totalPdfPages - state.offset) : 0;
    renderDetail(fresh);
    loadPage();
  } catch {
    /* 忽略 */
  }
}

/* ------------------------------------------------------------- 翻页预览 */

function setPage(pageNumber) {
  const total = state.totalPdfPages || 1;
  const next = clamp(Number(pageNumber) || 1, 1, total);
  if (next === state.page && state.page !== 0) return;
  state.page = next;
  loadPage();
}

function jumpToPrinted(printed) {
  if (!Number.isFinite(printed) || !state.offset) return;
  setPage(printed + state.offset);
}

function jumpToPdf(pdfPage) {
  if (!Number.isFinite(pdfPage)) return;
  setPage(pdfPage);
}

const ZOOM_STEPS = [1, 1.25, 1.5, 2, 2.5, 3, 4];

/** 适应屏幕时的显示宽度（CSS px）：受容器宽/高与页面宽高比约束。 */
function fitDisplayWidth() {
  const stage = $("#page-stage");
  const image = $("#page-image");
  const cssWidth = Math.max(200, stage.clientWidth - 24);
  const cssHeight = Math.max(200, stage.clientHeight - 24);
  const aspect = image && image.naturalWidth && image.naturalHeight ? image.naturalWidth / image.naturalHeight : 0.72;
  return Math.max(200, Math.min(cssWidth, cssHeight * aspect));
}

/** 按缩放倍率与屏幕像素比决定服务端渲染宽度（900–2600）；放大时自动取更高清版本。 */
function renderWidthFor(displayWidth) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  return Math.max(900, Math.min(2600, Math.round(displayWidth * state.zoomFactor * dpr)));
}

function captureScroll(stage) {
  return {
    x: stage.scrollWidth > stage.clientWidth ? (stage.scrollLeft + stage.clientWidth / 2) / stage.scrollWidth : 0.5,
    y: stage.scrollHeight > stage.clientHeight ? (stage.scrollTop + stage.clientHeight / 2) / stage.scrollHeight : 0.5,
  };
}

function restoreScroll(position) {
  const stage = $("#page-stage");
  if (!position) return;
  stage.scrollLeft = Math.max(0, position.x * stage.scrollWidth - stage.clientWidth / 2);
  stage.scrollTop = Math.max(0, position.y * stage.scrollHeight - stage.clientHeight / 2);
}

/** 应用当前缩放：适应时不设固定宽高；放大时给图片显式宽度并允许滚动。 */
function applyZoom(options = {}) {
  const stage = $("#page-stage");
  const image = $("#page-image");
  if (!stage || !image) return;
  const before = captureScroll(stage);
  const fitWidth = fitDisplayWidth();
  if (state.zoomFactor <= 1) {
    image.style.width = "";
    image.style.height = "";
    stage.classList.remove("zoomed");
  } else {
    image.style.width = `${Math.round(fitWidth * state.zoomFactor)}px`;
    image.style.height = "auto";
    stage.classList.add("zoomed");
  }
  $("#zoom-label").textContent = state.zoomFactor <= 1 ? "适应" : `${Math.round(state.zoomFactor * 100)}%`;
  const needed = renderWidthFor(fitWidth);
  if (!options.silent && state.pageImageWidth && needed !== state.pageImageWidth) {
    state.pendingScroll = before; // 换更高清图后按比例恢复视线中心
    reloadPageImage();
  } else {
    restoreScroll(before);
  }
}

function reloadPageImage() {
  const book = state.selected;
  if (!book || !book.pdf) return;
  const image = $("#page-image");
  const loading = $("#page-loading");
  loading.textContent = "加载中…";
  loading.hidden = false;
  state.pageImageWidth = renderWidthFor(fitDisplayWidth());
  image.src = `/api/books/${book.id}/page/${state.page}.png?w=${state.pageImageWidth}&v=${book.status.pdf.mtime || ""}`;
}

function setZoom(factor) {
  if (!state.selected || !state.selected.pdf) return;
  state.zoomFactor = factor;
  applyZoom();
}

function zoomBy(direction) {
  const current = ZOOM_STEPS.findIndex((step) => Math.abs(step - state.zoomFactor) < 0.001);
  const next = Math.min(ZOOM_STEPS.length - 1, Math.max(0, (current === -1 ? 0 : current) + direction));
  setZoom(ZOOM_STEPS[next]);
}

function toggleFullscreen() {
  const preview = $(".preview");
  if (!preview) return;
  const active = document.fullscreenElement || document.webkitFullscreenElement;
  if (active) {
    const exit = document.exitFullscreen || document.webkitExitFullscreen;
    if (exit) exit.call(document);
    return;
  }
  const request = preview.requestFullscreen || preview.webkitRequestFullscreen;
  if (request) request.call(preview);
}

function onFullscreenChange() {
  const active = Boolean(document.fullscreenElement || document.webkitFullscreenElement);
  const button = $("#btn-fullscreen");
  if (button) button.textContent = active ? "✕ 退出全屏" : "⛶ 全屏";
  // 容器尺寸变化后按新的适应尺寸取图
  requestAnimationFrame(() => {
    if (state.selected && state.selected.pdf) applyZoom();
  });
}

/** 放大后按住鼠标拖拽平移。 */
function bindPan() {
  const stage = $("#page-stage");
  let drag = null;
  stage.addEventListener("mousedown", (event) => {
    if (state.zoomFactor <= 1 || event.button !== 0) return;
    drag = { x: event.clientX, y: event.clientY, left: stage.scrollLeft, top: stage.scrollTop };
    stage.classList.add("panning");
    event.preventDefault();
  });
  window.addEventListener("mousemove", (event) => {
    if (!drag) return;
    stage.scrollLeft = drag.left - (event.clientX - drag.x);
    stage.scrollTop = drag.top - (event.clientY - drag.y);
  });
  window.addEventListener("mouseup", () => {
    if (!drag) return;
    drag = null;
    stage.classList.remove("panning");
  });
}

function loadPage() {
  const book = state.selected;
  if (!book || !book.pdf) return;
  const image = $("#page-image");
  const loading = $("#page-loading");
  loading.textContent = "加载中…";
  loading.hidden = false;
  state.pendingScroll = null;
  state.pageImageWidth = renderWidthFor(fitDisplayWidth());
  image.src = `/api/books/${book.id}/page/${state.page}.png?w=${state.pageImageWidth}&v=${book.status.pdf.mtime || ""}`;
  image.alt = `${book.title} PDF 第 ${state.page} 页`;
  $("#input-pdf").value = String(state.page);
  const printed = state.page - state.offset;
  $("#input-printed").value = state.offset && printed >= 1 ? String(printed) : "";
  applyZoom({ silent: true });
  loadPageText(printed);
}

async function loadPageText(printed) {
  const box = $("#page-text");
  const book = state.selected;
  if (!book) return;
  if (!$("#toggle-text").checked) {
    box.hidden = true;
    return;
  }
  box.hidden = false;
  if (!state.offset || printed < 1 || !book.status.index.found) {
    box.innerHTML = `<div class="empty">${
      book.status.index.found
        ? "当前是前置页（封面 / 目录），没有对应的原书页文本。"
        : "这本教材还没有建立索引，先点“处理教材”。"
    }</div>`;
    return;
  }
  const bookId = book.id;
  const requestId = ++state.textRequest;
  box.innerHTML = `<div class="empty">加载文本…</div>`;
  try {
    const data = await api(`/api/books/${bookId}/text/${printed}`);
    if (requestId !== state.textRequest || !state.selected || state.selected.id !== bookId) return;
    if (!data.chunks.length) {
      box.innerHTML = `<div class="empty">原书第 ${printed} 页没有索引文本（可能是图页或已并入相邻页）。</div>`;
      return;
    }
    box.innerHTML = data.chunks
      .map(
        (chunk) => `
        <div class="text-chunk">
          <div class="section">${escapeHtml(chunk.section || "未标注章节")}</div>
          <div class="body">${escapeHtml(chunk.text)}</div>
          <div class="chunk-id" data-chunk="${escapeHtml(chunk.chunk_id)}">chunk_id=${escapeHtml(chunk.chunk_id)}</div>
        </div>`,
      )
      .join("");
    $$(".chunk-id", box).forEach((element) => element.addEventListener("click", () => copyText(element.dataset.chunk, "已复制 chunk_id")));
  } catch (error) {
    if (requestId === state.textRequest) box.innerHTML = `<div class="empty">文本加载失败：${escapeHtml(error.message)}</div>`;
  }
}

/* --------------------------------------------------------------- 检索 */

function renderSearchScope() {
  const select = $("#search-scope");
  if (!select) return;
  const current = select.value || "all";
  const options = ['<option value="all">全部教材</option>'];
  for (const book of state.books) {
    if (!book.status.index.complete) continue; // 未建索引的教材检索不到内容
    options.push(`<option value="${book.id}" title="${escapeHtml(book.index_title || book.title)}">${escapeHtml(book.title)}</option>`);
  }
  select.innerHTML = options.join("");
  select.value = state.books.some((book) => book.id === current) ? current : "all";
}

async function doSearch() {
  const query = $("#global-search-input").value.trim();
  if (!query) {
    toast("请输入检索词", "error");
    return;
  }
  const scopeValue = $("#search-scope").value;
  try {
    const data = await api("/api/search", {
      method: "POST",
      body: JSON.stringify({ query, limit: 8, book_id: scopeValue === "all" ? null : scopeValue }),
    });
    renderSearchResults(data);
  } catch (error) {
    toast(`检索失败：${error.message}`, "error");
  }
}

function renderSearchResults(data) {
  $("#detail").hidden = true;
  $("#welcome").hidden = true;
  $("#search-results").hidden = false;
  const scopeLabel = $("#search-scope-label");
  if (scopeLabel) scopeLabel.textContent = data.scope ? `· 范围：《${data.scope.title}》` : "· 范围：全部教材";
  const list = $("#search-results-list");
  if (!data.results.length) {
    const where = data.scope ? `《${escapeHtml(data.scope.title)}》里` : "本地教材";
    list.innerHTML = `<div class="muted">${where}没有检索到“${escapeHtml(data.query)}”的证据。</div>`;
    return;
  }
  list.innerHTML = "";
  for (const item of data.results) {
    const card = document.createElement("div");
    card.className = "result-card";
    card.innerHTML = `
      <div class="head">
        <div class="cite">《${escapeHtml(item.book)}》第 <em>${item.page}</em> 页 · ${escapeHtml(item.section)}</div>
        <button class="btn small" data-open>打开该页</button>
      </div>
      <div class="text">${escapeHtml(item.text)}</div>
      <div class="foot"><span class="chunk" data-chunk="${escapeHtml(item.chunk_id)}">chunk_id=${escapeHtml(item.chunk_id)}</span></div>`;
    card.querySelector("[data-open]").addEventListener("click", () => openSearchResult(item));
    card.querySelector("[data-chunk]").addEventListener("click", () => copyText(item.chunk_id, "已复制 chunk_id"));
    list.appendChild(card);
  }
}

async function openSearchResult(item) {
  const book = state.books.find((candidate) => candidate.index_title === item.book || candidate.title === item.book);
  if (!book) {
    toast("书库里没有找到这本教材", "error");
    return;
  }
  await selectBook(book);
  jumpToPrinted(item.page);
}

/* --------------------------------------------------------------- 任务 */

function buildTaskCard(task) {
  const root = document.createElement("div");
  root.className = "task-card";
  root.innerHTML = `
    <div class="task-title">
      <span class="name"></span>
      <span class="task-status"></span>
    </div>
    <div class="progress"><div class="progress-bar"></div></div>
    <div class="task-stage"></div>
    <div class="task-stages"></div>
    <div class="task-error" hidden></div>
    <pre class="task-logs" hidden></pre>
    <div class="task-actions">
      <button class="btn small" data-logs>日志</button>
      <button class="btn small" data-cancel>取消</button>
    </div>`;
  const element = {
    root,
    name: root.querySelector(".name"),
    status: root.querySelector(".task-status"),
    bar: root.querySelector(".progress-bar"),
    stage: root.querySelector(".task-stage"),
    stages: root.querySelector(".task-stages"),
    error: root.querySelector(".task-error"),
    logs: root.querySelector(".task-logs"),
  };
  element.name.textContent = task.label;
  root.querySelector("[data-logs]").addEventListener("click", () => {
    state.expandedTask = state.expandedTask === task.id ? null : task.id;
    renderTasks();
  });
  root.querySelector("[data-cancel]").addEventListener("click", async () => {
    try {
      await api(`/api/tasks/${task.id}/cancel`, { method: "POST", body: "{}" });
      toast("已请求取消任务", "ok");
    } catch (error) {
      toast(error.message, "error");
    }
  });
  return element;
}

function updateTaskCard(element, task) {
  const statusText = { pending: "排队中", running: `进行中 ${task.percent}%`, done: "完成", error: "失败", cancelled: "已取消" }[task.status] || task.status;
  element.status.textContent = statusText;
  element.status.className = `task-status ${task.status}`;
  element.bar.style.width = `${task.percent}%`;
  if (task.progress_message) {
    const stage = task.stages.find((item) => item.key === task.current_stage);
    element.stage.textContent = `${stage ? stage.label : ""} ${task.progress_message}`.trim();
  } else if (task.status === "done") {
    element.stage.textContent = "全部阶段已完成";
  } else if (task.status === "pending") {
    element.stage.textContent = "等待开始…";
  }
  element.stages.innerHTML = task.stages
    .map((stage) => {
      const mark = { done: "✓", running: "●", error: "✕", skipped: "–", pending: "·" }[stage.status] || "·";
      const detail = stage.total ? ` ${stage.done}/${stage.total}` : "";
      const message = stage.message ? ` · ${escapeHtml(stage.message)}` : "";
      return `<div class="task-stage-row ${stage.status}"><span class="mark">${mark}</span><span>${escapeHtml(stage.label)}${detail}${message}</span></div>`;
    })
    .join("");
  if (task.error) {
    element.error.hidden = false;
    element.error.textContent = task.error;
  } else {
    element.error.hidden = true;
  }
  element.logs.hidden = state.expandedTask !== task.id;
  element.root.querySelector("[data-cancel]").disabled = !(task.status === "running" || task.status === "pending");
}

function renderTasks() {
  const panel = $("#task-panel");
  const list = $("#task-list");
  if (!state.tasks.length) {
    panel.hidden = true;
    return;
  }
  const active = state.tasks.filter((task) => task.status === "running" || task.status === "pending");
  if (!state.taskPanelOpen && active.length) state.taskPanelOpen = true;
  panel.hidden = !state.taskPanelOpen;
  if (!state.expandedTask && active.length) state.expandedTask = active[0].id;

  const visible = state.tasks.slice(0, 5);
  for (const [id, element] of state.taskEls) {
    if (!visible.some((task) => task.id === id)) {
      element.root.remove();
      state.taskEls.delete(id);
    }
  }
  for (const task of visible) {
    let element = state.taskEls.get(task.id);
    if (!element) {
      element = buildTaskCard(task);
      state.taskEls.set(task.id, element);
    }
    updateTaskCard(element, task);
    list.appendChild(element.root);
  }
  if (state.expandedTask) fetchTaskLogs(state.expandedTask).catch(() => {});
}

async function fetchTaskLogs(taskId) {
  const since = state.logCursor[taskId] || 0;
  const snapshot = await api(`/api/tasks/${taskId}?since=${since}`);
  state.logCursor[taskId] = snapshot.log_cursor || since;
  const element = state.taskEls.get(taskId);
  if (element && snapshot.logs && snapshot.logs.length) {
    element.logs.textContent += `${snapshot.logs.join("\n")}\n`;
    element.logs.scrollTop = element.logs.scrollHeight;
  }
}

async function pollTasks() {
  let tasks;
  try {
    tasks = await api("/api/tasks");
  } catch {
    return;
  }
  let finishedAny = false;
  const nextKnown = {};
  for (const task of tasks) {
    nextKnown[task.id] = task.status;
    const previous = state.knownStatus[task.id];
    if (previous && previous !== task.status && ["done", "error", "cancelled"].includes(task.status)) {
      finishedAny = true;
      if (task.status === "done") toast(`任务完成：${task.label}`, "ok");
      else if (task.status === "error") toast(`任务失败：${task.label}`, "error");
    }
  }
  state.knownStatus = nextKnown;
  state.tasks = tasks;
  renderTasks();
  renderBookList();
  if (finishedAny) {
    await refreshBooks();
    if (state.selected) await refreshDetail();
  }
}

function openTaskPanel() {
  state.taskPanelOpen = true;
  $("#task-panel").hidden = false;
  pollTasks();
}

/* --------------------------------------------------------------- 导入 */

async function inspectPdfPath() {
  const path = $("#path-input").value.trim();
  const box = $("#inspect-result");
  if (!path || !/\.pdf$/i.test(path)) {
    box.hidden = true;
    return;
  }
  box.hidden = false;
  box.className = "inspect-result muted";
  box.textContent = "正在检测文字层…";
  try {
    const data = await api("/api/inspect", { method: "POST", body: JSON.stringify({ path }) });
    box.className = `inspect-result ${data.kind === "text" ? "ok" : "warn"}`;
    const stages = (data.recommended_stages || []).map((key) => STAGE_LABELS[key] || key).join(" → ");
    box.textContent = `${data.recommendation}推荐阶段：${stages}`;
  } catch (error) {
    box.className = "inspect-result warn";
    box.textContent = error.message;
  }
}

function openImportModal() {
  state.uploadFile = null;
  $("#file-info").textContent = "";
  $("#file-input").value = "";
  $("#import-progress").hidden = true;
  $("#inspect-result").hidden = true;
  $("#import-progress-bar").style.width = "0%";
  $("#import-progress-text").textContent = "";
  $("#import-modal").hidden = false;
}

function setUploadFile(file) {
  if (!file) return;
  if (!/\.pdf$/i.test(file.name)) {
    toast("请选择 PDF 文件", "error");
    return;
  }
  state.uploadFile = file;
  $("#file-info").textContent = `${file.name} · ${formatBytes(file.size)}`;
  if (!$("#upload-title").value) $("#upload-title").placeholder = file.name.replace(/\.pdf$/i, "");
}

function uploadViaXhr(url, file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("PUT", url);
    xhr.setRequestHeader("Content-Type", "application/pdf");
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(event.loaded, event.total);
    };
    xhr.onload = () => {
      let payload;
      try {
        payload = JSON.parse(xhr.responseText);
      } catch {
        reject(new Error(`上传失败：HTTP ${xhr.status}`));
        return;
      }
      if (xhr.status >= 200 && xhr.status < 300 && payload.ok !== false) resolve(payload.data);
      else reject(new Error(payload.error || `HTTP ${xhr.status}`));
    };
    xhr.onerror = () => reject(new Error("上传失败：网络错误"));
    xhr.send(file);
  });
}

async function submitImport() {
  const activeTab = $(".tab.active").dataset.tab;
  const button = $("#btn-import-submit");
  button.disabled = true;
  try {
    if (activeTab === "upload") {
      if (!state.uploadFile) throw new Error("请先选择 PDF 文件");
      const title = $("#upload-title").value.trim();
      const begin = await api("/api/import/begin", {
        method: "POST",
        body: JSON.stringify({ filename: state.uploadFile.name, title }),
      });
      $("#import-progress").hidden = false;
      await uploadViaXhr(begin.upload_url, state.uploadFile, (loaded, total) => {
        const percent = Math.round((loaded / total) * 100);
        $("#import-progress-bar").style.width = `${percent}%`;
        $("#import-progress-text").textContent = `上传中 ${formatBytes(loaded)} / ${formatBytes(total)}（${percent}%）`;
      });
      toast("上传完成，开始处理前请先确认状态", "ok");
    } else {
      const path = $("#path-input").value.trim();
      if (!path) throw new Error("请填写 PDF 路径");
      const title = $("#path-title").value.trim();
      await api("/api/import", { method: "POST", body: JSON.stringify({ path, title }) });
      toast("已开始导入", "ok");
    }
    $("#import-modal").hidden = true;
    openTaskPanel();
    await pollTasks();
    await refreshBooks();
  } catch (error) {
    toast(error.message, "error");
  } finally {
    button.disabled = false;
  }
}

/* --------------------------------------------------------------- 处理 */

function openProcessModal() {
  const book = state.selected;
  if (!book) return;
  $("#process-book-name").textContent = book.title;
  const textWorkflow = book.workflow === "text";
  const textInfo = book.status.text || {};
  $("#process-hint").textContent = textWorkflow
    ? `检测到文字层 PDF（${textInfo.text_pages}/${textInfo.pages} 页有文字，共 ${textInfo.chars} 字），推荐直接解析，无需 OCR。`
    : "未检测到可用文字层，将使用 OCR 流水线（图片型 PDF 耗时较长）。";
  $("#force-ocr").closest("label").hidden = textWorkflow;
  $(".advanced").hidden = textWorkflow;
  $("#force-ocr").checked = false;
  $('input[name="stage-mode"][value="auto"]').checked = true;
  renderStageList(book);
  updateStageMode();
  $("#process-modal").hidden = false;
}

function renderStageList(book) {
  const textWorkflow = book.workflow === "text";
  const stages = workflowStageDefs(book);
  const allDone = stages.every((stage) => stage.done(book));
  $("#stage-list").innerHTML = stages.map((stage) => {
    const done = stage.done(book);
    const checked = allDone
      ? textWorkflow
        ? ["text", "index"].includes(stage.key)
        : ["fix", "structure", "index"].includes(stage.key)
      : !done;
    return `
      <label class="stage-item" data-key="${stage.key}">
        <input type="checkbox" value="${stage.key}" ${checked ? "checked" : ""}>
        <span class="info">
          <span class="title">${stage.label}${done ? "（已完成）" : ""}</span>
          <span class="hint">${escapeHtml(stage.hint(book))}</span>
        </span>
      </label>`;
  }).join("");
}

function updateStageMode() {
  const mode = $('input[name="stage-mode"]:checked').value;
  const disabled = mode === "auto";
  $$("#stage-list input").forEach((input) => {
    input.disabled = disabled;
    input.closest(".stage-item").classList.toggle("disabled", disabled);
  });
}

async function submitProcess() {
  const book = state.selected;
  if (!book) return;
  const mode = $('input[name="stage-mode"]:checked').value;
  let stages = "auto";
  if (mode === "manual") {
    stages = $$("#stage-list input:checked").map((input) => input.value);
    if (!stages.length) {
      toast("请至少选择一个阶段", "error");
      return;
    }
  }
  const button = $("#btn-process-submit");
  button.disabled = true;
  try {
    await api("/api/process", {
      method: "POST",
      body: JSON.stringify({
        book_id: book.id,
        stages,
        force: $("#force-ocr").checked,
        dpi: Number($("#dpi-select").value),
        device: $("#device-select").value,
      }),
    });
    $("#process-modal").hidden = true;
    toast("已开始处理，进度见右下角任务面板", "ok");
    openTaskPanel();
    await pollTasks();
  } catch (error) {
    toast(error.message, "error");
  } finally {
    button.disabled = false;
  }
}

async function reindexSelected() {
  const book = state.selected;
  if (!book) return;
  try {
    await api("/api/reindex", { method: "POST", body: JSON.stringify({ book_id: book.id }) });
    toast("已开始重建索引", "ok");
    openTaskPanel();
    await pollTasks();
  } catch (error) {
    toast(error.message, "error");
  }
}

/* ------------------------------------------------------------- 事件绑定 */

function bindEvents() {
  $("#btn-refresh").addEventListener("click", () => {
    refreshBooks();
    pollTasks();
    toast("已刷新", "ok");
  });
  $("#btn-search").addEventListener("click", doSearch);
  $("#global-search-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter") doSearch();
  });
  $("#search-scope").addEventListener("change", () => {
    if ($("#global-search-input").value.trim() && !$("#search-results").hidden) doSearch();
  });
  $("#btn-search-close").addEventListener("click", () => {
    $("#search-results").hidden = true;
    if (state.selected) $("#detail").hidden = false;
    else $("#welcome").hidden = false;
  });
  $("#btn-import").addEventListener("click", openImportModal);
  $("#brand-home").addEventListener("click", showWelcome);
  $("#btn-welcome-open").addEventListener("click", () => {
    const first = state.books.find((book) => book.pdf) || state.books[0];
    if (first) selectBook(first);
  });
  $("#btn-welcome-import").addEventListener("click", openImportModal);
  $("#btn-welcome-guide").addEventListener("click", () => {
    $("#guide").open = true;
    $("#guide").scrollIntoView({ behavior: "smooth", block: "center" });
  });

  $("#btn-process").addEventListener("click", openProcessModal);
  $("#btn-reindex").addEventListener("click", reindexSelected);
  $("#btn-prev").addEventListener("click", () => setPage(state.page - 1));
  $("#btn-next").addEventListener("click", () => setPage(state.page + 1));
  $("#input-printed").addEventListener("change", (event) => jumpToPrinted(Number(event.target.value)));
  $("#input-printed").addEventListener("keydown", (event) => {
    if (event.key === "Enter") jumpToPrinted(Number(event.target.value));
  });
  $("#input-pdf").addEventListener("change", (event) => jumpToPdf(Number(event.target.value)));
  $("#input-pdf").addEventListener("keydown", (event) => {
    if (event.key === "Enter") jumpToPdf(Number(event.target.value));
  });
  $("#toggle-text").addEventListener("change", () => loadPage());
  $("#btn-zoom-in").addEventListener("click", () => zoomBy(1));
  $("#btn-zoom-out").addEventListener("click", () => zoomBy(-1));
  $("#zoom-label").addEventListener("click", () => setZoom(1));
  $("#btn-fullscreen").addEventListener("click", toggleFullscreen);
  document.addEventListener("fullscreenchange", onFullscreenChange);
  document.addEventListener("webkitfullscreenchange", onFullscreenChange);
  bindPan();

  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (state.selected && !$("#detail").hidden) loadPage();
    }, 250);
  });

  $("#page-image").addEventListener("load", () => {
    $("#page-loading").hidden = true;
    applyZoom({ silent: true });
    if (state.pendingScroll) {
      restoreScroll(state.pendingScroll);
      state.pendingScroll = null;
    }
  });
  $("#page-image").addEventListener("error", () => { $("#page-loading").textContent = "页面加载失败"; });
  $("#page-image").addEventListener("dblclick", () => setZoom(state.zoomFactor > 1 ? 1 : 2));
  $("#detail-cover").addEventListener("error", () => {
    const cover = $("#detail-cover");
    cover.replaceWith(coverPlaceholder(state.selected ? state.selected.title : "医"));
  });

  $$("[data-close]").forEach((button) => button.addEventListener("click", () => { $(`#${button.dataset.close}`).hidden = true; }));
  $$(".modal").forEach((modal) => modal.addEventListener("click", (event) => { if (event.target === modal) modal.hidden = true; }));
  $$(".tab").forEach((tab) => tab.addEventListener("click", () => {
    $$(".tab").forEach((item) => item.classList.toggle("active", item === tab));
    $$(".tab-body").forEach((body) => { body.hidden = body.dataset.tab !== tab.dataset.tab; });
  }));
  $$('input[name="stage-mode"]').forEach((radio) => radio.addEventListener("change", updateStageMode));

  const dropzone = $("#dropzone");
  dropzone.addEventListener("click", () => $("#file-input").click());
  dropzone.addEventListener("dragover", (event) => { event.preventDefault(); dropzone.classList.add("dragover"); });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
    setUploadFile(event.dataTransfer.files[0]);
  });
  $("#file-input").addEventListener("change", (event) => setUploadFile(event.target.files[0]));
  $("#btn-inspect").addEventListener("click", inspectPdfPath);
  $("#path-input").addEventListener("change", inspectPdfPath);
  $("#btn-import-submit").addEventListener("click", submitImport);
  $("#btn-process-submit").addEventListener("click", submitProcess);

  $("#btn-task-collapse").addEventListener("click", () => {
    state.taskPanelOpen = false;
    $("#task-panel").hidden = true;
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeAllModals();
      return;
    }
    const tag = document.activeElement && document.activeElement.tagName;
    if (["INPUT", "SELECT", "TEXTAREA"].includes(tag)) return;
    if ($("#detail").hidden) return;
    if (event.key === "ArrowLeft") setPage(state.page - 1);
    if (event.key === "ArrowRight") setPage(state.page + 1);
    if (event.key === "+" || event.key === "=") zoomBy(1);
    if (event.key === "-" || event.key === "_") zoomBy(-1);
    if (event.key === "0") setZoom(1);
    if (event.key === "f" || event.key === "F") toggleFullscreen();
  });
}

function closeAllModals() {
  ["import-modal", "process-modal"].forEach((id) => { const modal = $(`#${id}`); if (modal) modal.hidden = true; });
}

/* ------------------------------------------------------------------ 启动 */

async function init() {
  bindEvents();
  try {
    const data = await api("/api/overview");
    state.root = data.root;
    state.books = data.books;
    renderSearchScope();
    state.tasks = data.tasks;
    for (const task of data.tasks) state.knownStatus[task.id] = task.status;
    renderBookList();
    renderEnv(data);
    renderWelcomeStats(data);
    renderTasks();
    // 书库非空时直接打开第一本教材，欢迎页只在空书库或点击左上角 logo 时出现
    const first = state.books.find((book) => book.pdf) || state.books[0];
    if (first) await selectBook(first);
  } catch (error) {
    toast(`无法连接后端：${error.message}`, "error");
  }
  setInterval(pollTasks, 1200);
  setInterval(() => { if (!$("#welcome").hidden || state.selected) refreshBooks(); }, 6000);
}

document.addEventListener("DOMContentLoaded", init);
