// 보고서용 화면 캡처. 헤드리스 Chrome을 DevTools 프로토콜로 직접 조종한다(추가 설치 없음, Node 22+).
//   node report/capture.mjs                      → 게시된 사이트를 찍는다
//   node report/capture.mjs http://localhost:8090/   → 로컬 site/ 를 찍는다
// 결과: report/captures/ (다시 돌리면 덮어쓴다)
import { spawn } from "node:child_process";
import { mkdirSync, rmSync, writeFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const BASE = (process.argv[2] || "https://gkfka1128.github.io/weather-subway-visualization/").replace(/\/?$/, "/");
const REPO = "https://github.com/gkfka1128/weather-subway-visualization";
const OUT =join(dirname(fileURLToPath(import.meta.url)), "captures");
const CHROME = ["C:/Program Files/Google/Chrome/Application/chrome.exe",
                "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"].find(existsSync);
const PORT = 9333, SCALE = 2, WIDE = 1280, PHONE = 390;
const sleep = ms => new Promise(r => setTimeout(r, ms));

// ── 브라우저 연결 ─────────────────────────────────────────────
const profile = join(tmpdir(), "capture-profile-" + process.pid);
const chrome = spawn(CHROME, ["--headless=new", `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`,
  "--hide-scrollbars", "--no-first-run", `--window-size=${WIDE},900`, "about:blank"], { stdio: "ignore" });
process.on("exit", () => chrome.kill());  // 중간에 오류로 끝나도 브라우저를 남기지 않는다

async function pageSocket() {
  for (let i = 0; i < 50; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
      const page = list.find(t => t.type === "page");
      if (page) return page.webSocketDebuggerUrl;
    } catch { /* 아직 안 떴다 */ }
    await sleep(200);
  }
  throw new Error("Chrome에 연결하지 못했습니다");
}
const ws = new WebSocket(await pageSocket());
await new Promise((ok, no) => { ws.onopen = ok; ws.onerror = no; });
let seq = 0;
const waiting = new Map(), listeners = new Map();
ws.onmessage = ev => {
  const m = JSON.parse(ev.data);
  if (m.id) { const w = waiting.get(m.id); waiting.delete(m.id); m.error ? w.no(new Error(m.error.message)) : w.ok(m.result); }
  else listeners.get(m.method)?.(m.params);
};
const send = (method, params = {}) => new Promise((ok, no) => { waiting.set(++seq, { ok, no }); ws.send(JSON.stringify({ id: seq, method, params })); });
const once = method => new Promise(ok => listeners.set(method, p => { listeners.delete(method); ok(p); }));

async function js(expr) {
  const r = await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true });
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text);
  return r.result.value;
}
async function viewport(width, mobile = false) {
  await send("Emulation.setDeviceMetricsOverride", { width, height: 900, deviceScaleFactor: SCALE, mobile });
}
// 그림·글꼴이 다 실릴 때까지 기다린다. 숨은 탭의 그림도 포함.
const settle = () => js(`(async () => {
  await document.fonts.ready;
  await Promise.all([...document.images].map(i => i.complete ? 0 : new Promise(r => { i.onload = i.onerror = r; })));
  await new Promise(r => setTimeout(r, 400));
  // 지금 보이는 탭에 지도(Plotly)가 있으면 지도 그림 조각이 다 실릴 때까지 조금 더 기다린다
  if ([...document.querySelectorAll(".maplibregl-map")].some(m => m.offsetParent)) await new Promise(r => setTimeout(r, 2500));
})()`);
async function open(page) {
  const loaded = once("Page.loadEventFired");
  await send("Page.navigate", { url: BASE + page });
  await loaded;
  // 화면 위에 붙는 탭 막대(sticky)는 긴 화면을 한 장으로 찍을 때 중간에 겹쳐 찍힌다 → 찍는 동안만 제자리에 둔다
  await js(`document.head.insertAdjacentHTML("beforeend", "<style>.tabs{position:static!important}</style>")`);
  await settle();
}
async function tab(id) {
  await js(`location.hash = "#${id}"`);
  await js(`scrollTo(0, 0)`);
  await settle();
}
// 예보를 받아 '오늘 예측'이 그려질 때까지(최대 20초)
async function forecastReady(sel) {
  for (let i = 0; i < 40; i++) {
    if (await js(`!/불러오는 중/.test(document.querySelector(${JSON.stringify(sel)})?.textContent ?? "불러오는 중")`)) break;
    await sleep(500);
  }
  await settle();
}

// ── 찍기 ─────────────────────────────────────────────────────
let count = 0;
async function shoot(file, clip) {
  // Chrome은 한 변이 16384px를 넘는 그림을 못 만든다 → 너무 길면 배율을 낮춘다
  const scale = clip.height * SCALE > 16000 ? 16000 / (clip.height * SCALE) : 1;
  const { data } = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: true, clip: { ...clip, scale } });
  const path = join(OUT, file);
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, Buffer.from(data, "base64"));
  count++;
  console.log(file);
}
// 선택자 하나 또는 여럿을 한 장에(여럿이면 모두 들어가는 가장 작은 사각형). pad = 둘레 여백(px)
async function shootEl(file, selectors, pad = 12) {
  const box = await js(`(() => {
    const els = ${JSON.stringify([].concat(selectors))}.map(s => document.querySelector(s)).filter(e => e && e.offsetParent);  // 숨겨진 요소(컨트롤 막대 등)는 뺀다
    if (!els.length) return null;
    const rs = els.map(e => e.getBoundingClientRect());
    const x0 = Math.min(...rs.map(r => r.left)), y0 = Math.min(...rs.map(r => r.top));
    const x1 = Math.max(...rs.map(r => r.right)), y1 = Math.max(...rs.map(r => r.bottom));
    return { x: x0 + scrollX, y: y0 + scrollY, width: x1 - x0, height: y1 - y0, pageW: document.documentElement.scrollWidth };
  })()`);
  if (!box) throw new Error("찾지 못함: " + selectors);
  const x = Math.max(0, box.x - pad), y = Math.max(0, box.y - pad);
  await shoot(file, { x, y, width: Math.min(box.pageW, box.x + box.width + pad) - x, height: box.y + box.height + pad - y });
}
async function shootPage(file) {
  const s = await js(`({ w: document.documentElement.scrollWidth, h: document.documentElement.scrollHeight })`);
  await shoot(file, { x: 0, y: 0, width: s.w, height: s.h });
}
const safe = s => s.replace(/[\\/:*?"<>|]/g, "").replace(/\s+/g, " ").trim();

// ── 실행 ─────────────────────────────────────────────────────
rmSync(OUT, { recursive: true, force: true });
await send("Page.enable");
await viewport(WIDE);

// 1. 화면 구성: 첫 화면과 탭별 전체 화면
await open("index.html");
await forecastReady("#today-sentence");
const tabs = await js(`[...document.querySelectorAll(".tabs [data-tab]")].map(t => ({ id: t.dataset.tab, name: t.textContent.replace(/^\\d+/, "").trim() }))`);
await shootPage("1_화면/0_첫 화면 전체(오늘 예측 탭).png");
await shootEl("1_화면/0_머리말과 전체 데이터 요약.png", ["header", ".overview"]);
for (const [i, t] of tabs.entries()) {
  await tab(t.id);
  await shootEl(`1_화면/${i + 1}_탭 ${safe(t.name)}.png`, [".tabs", ".controls", "#" + t.id]);
}

// 2. 그래프: 카드 전체(제목·그래프·설명)와 그래프만
const cards = await js(`[...document.querySelectorAll(".panel[id^=s]")].flatMap((p, si) =>
  [...p.querySelectorAll(".card")].map((c, ci) => ({ tab: p.id, id: c.id, no: (si + 1) + "-" + (ci + 1), title: c.querySelector("h2").firstChild.textContent.trim() })))`);
let cur = "";
for (const c of cards) {
  if (c.tab !== cur) { await tab(c.tab); cur = c.tab; }
  const name = `${c.no}_${safe(c.title)}.png`;
  await shootEl("2_그래프(설명 포함)/" + name, "#" + c.id, 0);
  await shootEl("3_그래프만/" + name, `#${c.id} .plot`, 4);
}

// 3. 컨트롤: 호선 칩·요일 콤보박스를 바꾸면 그래프가 바뀐다
const pages = await js(`PAGES`);
const pick = "5호선|토요일";
// 컨트롤 바로 아래 첫 카드(#corr)까지 한 장에 담아 '고른 값 → 바뀐 그래프'가 같이 보이게 한다
const changed = ["heatmap", "lines", "factors"];  // 호선·요일에 따라 바뀌는 카드 가운데 차이가 잘 보이는 것
await tab("s3");
await shootEl("4_컨트롤/1_컨트롤 막대.png", [".tabs", ".controls"]);
await tab("s2");  // 호선·요일에 따라 바뀌는 그래프가 없는 탭에서는 컨트롤 막대가 숨는다
await shootEl("4_컨트롤/1_컨트롤이 없는 탭(서울의 날씨).png", [".tabs", "#s2 .sec"]);
await tab("s3");
await shootEl("4_컨트롤/2_바꾸기 전(전체·전체).png", [".tabs", ".controls", "#corr"]);
if (pages[pick]) {
  await open(pages[pick] + "#s3");
  await tab("s3");
  await shootEl("4_컨트롤/3_바꾼 뒤(5호선·토요일).png", [".tabs", ".controls", "#corr"]);
  for (const c of cards.filter(c => changed.includes(c.id))) {  // 바꾸기 전 그림은 '2_그래프(설명 포함)'의 같은 번호
    await tab(c.tab);
    await shootEl(`4_컨트롤/4_바꾼 뒤(5호선·토요일) ${c.no}_${safe(c.title)}.png`, "#" + c.id, 0);
  }
  await tab("today");
  await forecastReady("#today-sentence");
  await shootEl("4_컨트롤/5_바꾼 뒤(5호선·토요일) 오늘 예측.png", [".tabs", ".controls", "#today"]);
}

// 4. 예측 서비스 화면
await open("today.html");
await forecastReady("#app");
await shootPage("5_예측 서비스/1_지하철 붐빌까(전체 화면).png");

// 5. 휴대폰 폭
await viewport(PHONE, true);
await open("index.html");
await forecastReady("#today-sentence");
await shootPage("6_휴대폰/1_첫 화면(오늘 예측).png");
await tab("s4");
await shootEl("6_휴대폰/2_그래프 카드(교통수단별 날씨 반응).png", [".tabs", ".controls", "#modes"]);
await tab("end");
await shootEl("6_휴대폰/3_결론 통념 검증 표.png", "#end");
await open("today.html");
await forecastReady("#app");
await shootPage("6_휴대폰/4_예측 서비스.png");

// 6. GitHub 저장소와 자동 배포 기록 (화면에 보이는 만큼만)
await viewport(WIDE);
for (const [file, path] of [["1_저장소 첫 화면.png", ""], ["2_자동 배포(Actions) 기록.png", "/actions"]]) {
  const loaded = once("Page.loadEventFired");
  await send("Page.navigate", { url: REPO + path });
  await loaded;
  await settle();
  await sleep(1500);
  await shoot("7_GitHub/" + file, { x: 0, y: 0, width: WIDE, height: 900 });
}

console.log(`\n${count}장 → ${OUT}`);
ws.close();
chrome.kill();
await sleep(500);
rmSync(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 300 });
