import { Terminal } from '/vendor/xterm.mjs';
import { FitAddon } from '/vendor/addon-fit.mjs';
const fragment = new URLSearchParams(location.hash.slice(1));
const nonce = fragment.get('nonce') || sessionStorage.getItem('tars-qa-nonce');
if (nonce) sessionStorage.setItem('tars-qa-nonce', nonce);
history.replaceState(null, '', location.pathname);
const status = document.querySelector('#status');
const error = document.querySelector('#error');
const restart = document.querySelector('#restart');
const term = new Terminal({ cols: 140, rows: 42, cursorBlink: true, convertEol: false, scrollback: 5000, fontSize: 15, lineHeight: 1.1, fontFamily: '"Cascadia Mono", Consolas, "Microsoft YaHei UI", monospace', theme: {background:'#101827',foreground:'#f2f5fb',cursor:'#f7d879'} });
// ConPTY owns Windows console capability negotiation. A browser DA response
// otherwise becomes literal input in Textual's Windows console driver.
// Suppress only DA queries; user onData and cursor-position replies stay intact.
term.parser.registerCsiHandler({final:'c'}, () => true);
term.parser.registerCsiHandler({prefix:'>', final:'c'}, () => true);
const fit = new FitAddon(); term.loadAddon(fit); term.open(document.querySelector('#terminal'));
window.tarsTerminal = term;
let cursor = 0, generation = null, inputChain = Promise.resolve();
async function api(route, payload) {
  const response = await fetch(route, {method:'POST', headers:{'Content-Type':'application/json','X-Tars-Bridge-Token':nonce || ''}, body:JSON.stringify(payload), credentials:'omit'});
  const result = await response.json(); if (!response.ok) throw new Error(result.error || response.status); return result;
}
function send(text) {
  // DA1/DA2 are renderer-generated replies, never Windows key encodings.
  // Keep all actual key and paste data, including ambiguous function-key/DSR forms.
  if (/^\x1b\[(?:\?|>)[0-9;]*c$/.test(text)) return;
  inputChain = inputChain.then(() => api('/api/input', {text})).catch(e => { error.textContent = e.message; });
}
term.onData(send);
term.attachCustomKeyEventHandler(e => { if (e.ctrlKey && ['q','x'].includes(e.key.toLowerCase())) e.preventDefault(); return true; });
let resizeTimer;
function resize() { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { fit.fit(); const rows=Math.max(10,Math.min(120,term.rows)); const cols=Math.max(40,Math.min(300,term.cols)); api('/api/resize',{rows,cols}).catch(e=>{error.textContent=e.message;}); },80); }
new ResizeObserver(resize).observe(document.querySelector('#terminal'));
restart.addEventListener('click', async () => { try { await api('/api/restart',{}); cursor=0; generation=null; term.reset(); error.textContent=''; term.focus(); } catch(e) {error.textContent=e.message;} });
async function poll() {
  try {
    const result = await api('/api/poll',{after:cursor});
    if (generation !== null && result.generation !== generation) { cursor=0; term.reset(); generation=result.generation; setTimeout(poll,10); return; }
    generation=result.generation;
    for (const chunk of result.chunks) term.write(chunk.text);
    cursor=result.cursor;
    status.textContent = result.running ? `已连接 · PID ${result.pid} · ${result.cols}×${result.rows}` : `TUI 已退出 · code ${result.exit_code ?? '?'} · 可显式重启`;
    restart.disabled=result.running;
    if(result.truncated) error.textContent='原始输出缓冲已截断；显式重启 TUI 可重建屏幕';
    if(result.error) error.textContent=result.error;
  } catch(e) {error.textContent=e.message;}
  setTimeout(poll,40);
}
term.focus(); resize(); poll();
