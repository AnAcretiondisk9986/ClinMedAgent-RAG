"use strict";
(() => {
  const $ = (s) => document.querySelector(s), root = document.documentElement;
  function init() {
    $("#btn-sidebar")?.addEventListener("click", e => { document.body.classList.toggle("sidebar-collapsed"); e.currentTarget.setAttribute("aria-expanded", String(!document.body.classList.contains("sidebar-collapsed"))); });
    $("#btn-overview")?.addEventListener("click", () => $("#brand-home")?.click());
    $("#btn-tasks")?.addEventListener("click", () => typeof openTaskPanel === "function" && openTaskPanel());
    const themeButton = $("#btn-theme");
    const savedTheme = localStorage.getItem("clinmedagent-theme");
    if (savedTheme === "light" || savedTheme === "dark") document.documentElement.dataset.theme = savedTheme;
    const updateThemeButton = () => { const dark = document.documentElement.dataset.theme === "dark" || (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches); if(themeButton){ themeButton.textContent = dark ? "☀ 亮色" : "☾ 暗色"; themeButton.setAttribute("aria-pressed", String(dark)); } };
    updateThemeButton();
    themeButton?.addEventListener("click", () => { const dark = document.documentElement.dataset.theme !== "dark"; document.documentElement.dataset.theme = dark ? "dark" : "light"; localStorage.setItem("clinmedagent-theme", document.documentElement.dataset.theme); updateThemeButton(); });
    $("#btn-focus")?.addEventListener("click", e => { const on=document.body.classList.toggle("reading-focus"); e.currentTarget.textContent=on?"退出专注":"专注阅读"; e.currentTarget.setAttribute("aria-pressed",String(on)); });
    ["book-filter","book-status-filter"].forEach(id => $("#"+id)?.addEventListener(id==="book-filter"?"input":"change", () => typeof renderBookList === "function" && renderBookList()));
    document.addEventListener("keydown", e => { if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==="k"){e.preventDefault();$("#global-search-input")?.focus();} });
    const r=$("#text-resizer"); if(r){ let x,w,drag=false; const move=e=>{if(!drag)return;const n=Math.max(260,Math.min(640,w-(e.clientX-x)));root.style.setProperty("--text-width",n+"px");r.setAttribute("aria-valuenow",n)}; const stop=()=>{drag=false;window.removeEventListener("pointermove",move);window.removeEventListener("pointerup",stop)}; r.addEventListener("pointerdown",e=>{drag=true;x=e.clientX;w=parseInt(getComputedStyle(root).getPropertyValue("--text-width"))||360;r.setPointerCapture?.(e.pointerId);window.addEventListener("pointermove",move);window.addEventListener("pointerup",stop)}); r.addEventListener("keydown",e=>{if(!["ArrowLeft","ArrowRight"].includes(e.key))return;e.preventDefault();const n=Math.max(260,Math.min(640,(parseInt(getComputedStyle(root).getPropertyValue("--text-width"))||360)+(e.key==="ArrowLeft"?-20:20)));root.style.setProperty("--text-width",n+"px");r.setAttribute("aria-valuenow",n)}); }
    $("#dropzone")?.addEventListener("keydown",e=>{if(e.key==="Enter"||e.key===" "){$("#file-input")?.click();}});
  }
  addEventListener("DOMContentLoaded",init);
})();
