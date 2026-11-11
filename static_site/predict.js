// 정적 사이트(GitHub Pages)용 예측: app.py의 predict()와 service/serve.py를 브라우저로 옮긴 것.
// model.json(평소 이용량·날씨 보정 계수, build_static.py가 생성) × Open-Meteo 예보(브라우저에서 직접 호출).
"use strict";

const KST = "Asia/Seoul";
const FORECAST_URL = "https://api.open-meteo.com/v1/forecast?latitude=37.5665&longitude=126.9780" +
  "&forecast_days=7&hourly=temperature_2m,precipitation,snowfall&timezone=Asia%2FSeoul";
const LEVELS = [[0.75, "매우 많음", "lv4"], [0.5, "많음", "lv3"], [0.3, "보통", "lv2"], [0, "한산", "lv1"]];

// ---------- 데이터 ----------
async function loadModel() {
  const r = await fetch("model.json");
  if (!r.ok) throw new Error("model.json " + r.status);
  return r.json();
}

async function loadForecast() {
  try {  // 30분 캐시 (저장소가 막혀 있어도 동작)
    const c = JSON.parse(sessionStorage.getItem("fc") || "null");
    if (c && Date.now() - c.at < 1800e3) return c.rows;
  } catch (e) { /* 무시 */ }
  const r = await fetch(FORECAST_URL);
  if (!r.ok) throw new Error("forecast " + r.status);
  const h = (await r.json()).hourly;
  const rows = h.time.map((t, i) => ({
    date: t.slice(0, 10), hour: +t.slice(11, 13),
    temp: h.temperature_2m[i], rain: h.precipitation[i], snow: h.snowfall[i],
  }));
  try { sessionStorage.setItem("fc", JSON.stringify({ at: Date.now(), rows })); } catch (e) { /* 무시 */ }
  return rows;
}

// ---------- 날짜 (서울 기준) ----------
function kstNow() {
  const p = {};
  new Intl.DateTimeFormat("en-CA", { timeZone: KST, year: "numeric", month: "2-digit", day: "2-digit",
    hour: "2-digit", hourCycle: "h23" }).formatToParts(new Date()).forEach(x => { p[x.type] = x.value; });
  return { ymd: `${p.year}-${p.month}-${p.day}`, hour: +p.hour };
}

function addDays(ymd, n) {
  const [y, m, d] = ymd.split("-").map(Number);
  const t = new Date(Date.UTC(y, m - 1, d + n));
  return { ymd: t.toISOString().slice(0, 10), wd: (t.getUTCDay() + 6) % 7, m: t.getUTCMonth() + 1, d: t.getUTCDate() };
}

function dayType(M, day) {
  if (M.holidays.includes(day.ymd)) return "일·공휴일";
  return day.wd < 5 ? "평일" : day.wd === 5 ? "토요일" : "일·공휴일";
}

// ---------- 보정 계수 (analysis.py와 같은 구간) ----------
const rainLevel = mm => mm <= 0.1 ? "무강수" : mm < 1 ? "약한 비(<1mm)" : mm < 5 ? "비(1~5mm)" : "강한 비(5mm+)";
const tempBand = c => c <= 0 ? "0℃ 이하" : c <= 10 ? "0~10℃" : c <= 20 ? "10~20℃" : c <= 28 ? "20~28℃" : "28℃ 초과";

function weatherFactor(M, temp, rain, snow) {
  const p = snow > 0 ? M.snow_f : (M.rain_f[rainLevel(rain)] ?? 1);
  return p * (M.temp_f[tempBand(temp)] ?? 1);
}

function usualFor(M, line, dtype, day) {  // 2025년 같은 시기(가장 가까운 기준일)의 시간대별 평소 이용량
  const doy = (Date.UTC(2025, day.m - 1, day.d) - Date.UTC(2025, 0, 1)) / 864e5;
  const rows = M.usual[line][dtype];
  return rows[Math.min(rows.length - 1, Math.round(doy / M.anchor_step))];
}

// offset일 뒤의 시간대별 예측 → { day, dtype, rows:[{hour,temp,rain,snow,base,pred}], pct }
function predict(M, fc, line, offset) {
  const day = addDays(kstNow().ymd, offset);
  const dtype = dayType(M, day);
  const base = usualFor(M, line, dtype, day);
  const rows = fc.filter(r => r.date === day.ymd && base[r.hour] != null).map(r => {
    const b = base[r.hour];
    return { ...r, base: b, pred: b * weatherFactor(M, r.temp, r.rain, r.snow) };
  });
  const sum = k => rows.reduce((s, r) => s + r[k], 0);
  return { day, dtype, rows, pct: rows.length ? (sum("pred") / sum("base") - 1) * 100 : 0 };
}

const hourKey = h => (h < 5 ? h + 24 : h);  // 0시(자정 이후)는 맨 뒤로
const signed = v => (v >= 0 ? "+" : "") + v.toFixed(1);
const levelOf = ratio => LEVELS.find(([th]) => ratio >= th);
const pad = n => String(n).padStart(2, "0");

// ---------- 대시보드의 '오늘 예측' 카드 ----------
async function dashToday(line) {
  const box = document.getElementById("today-sentence");
  try {
    const [M, fc] = await Promise.all([loadModel(), loadForecast()]);
    const p = predict(M, fc, line, 0);
    if (!p.rows.length) throw new Error("예보 없음");
    const temps = p.rows.map(r => r.temp), rain = p.rows.reduce((s, r) => s + r.rain, 0),
      snow = p.rows.reduce((s, r) => s + r.snow, 0);
    const sky = snow > 0 ? `눈(합계 ${snow.toFixed(1)}cm)` : rain > 0.1 ? `비(합계 ${rain.toFixed(1)}mm)` : "강수 없음";
    const word = Math.abs(p.pct) < 1 ? "비슷할" : p.pct < 0 ? "적을" : "많을";
    const mag = Math.abs(p.pct) < 1 ? "" : ` (약 ${Math.abs(p.pct).toFixed(1)}%)`;
    box.textContent = `오늘(${pad(p.day.m)}/${pad(p.day.d)} ${M.weekdays[p.day.wd]}, ${p.dtype}) 서울 날씨는 기온 ` +
      `${Math.round(Math.min(...temps))}~${Math.round(Math.max(...temps))}℃, ${sky}입니다. ` +
      `그래서 지하철 이용량은 평소 ${p.dtype}보다 ${word} 것으로 예측됩니다${mag}.`;
    document.getElementById("today-plot").innerHTML = lineChart(p.rows);
  } catch (e) {
    box.textContent = `오늘 예보를 불러오지 못했습니다 (${e.message}).`;
    box.classList.add("warn");
  }
}

function lineChart(rows) {  // 평소 vs 예측, 단위 천 명 (의존성 없는 SVG)
  const pts = rows.map(r => ({ x: hourKey(r.hour), base: r.base / 1000, pred: r.pred / 1000 })).sort((a, b) => a.x - b.x);
  const W = 640, H = 260, L = 50, R = 12, T = 28, B = 36;
  const ymax = Math.max(...pts.map(p => Math.max(p.base, p.pred))) * 1.08;
  const X = x => L + (x - 5) / (24 - 5) * (W - L - R), Y = y => H - B - y / ymax * (H - T - B);
  const path = k => pts.map((p, i) => `${i ? "L" : "M"}${X(p.x).toFixed(1)},${Y(p[k]).toFixed(1)}`).join("");
  const step = ymax > 400 ? 100 : ymax > 100 ? 50 : ymax > 40 ? 10 : 5;
  let g = "";
  for (let v = 0; v <= ymax; v += step) {
    g += `<line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}" stroke="#ebe8e1"/>` +
      `<text x="${L - 6}" y="${Y(v) + 4}" text-anchor="end">${v}</text>`;
  }
  for (let x = 6; x <= 24; x += 3) g += `<text x="${X(x)}" y="${H - B + 16}" text-anchor="middle">${x}</text>`;
  const dots = k => pts.map(p => `<circle cx="${X(p.x)}" cy="${Y(p[k])}" r="2.5"/>`).join("");
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="오늘의 시간대별 이용량 예측" style="width:100%;height:auto;font-size:11px;fill:#55534e">
    ${g}
    <text x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">시간 (24=자정 이후)</text>
    <text x="12" y="${(T + H - B) / 2}" transform="rotate(-90 12 ${(T + H - B) / 2})" text-anchor="middle">이용량 (천 명)</text>
    <path d="${path("base")}" fill="none" stroke="#a9a59b" stroke-width="2" stroke-dasharray="4 3"/>
    <path d="${path("pred")}" fill="none" stroke="#0f6b5c" stroke-width="2.2"/><g fill="#0f6b5c">${dots("pred")}</g>
    <g transform="translate(${L + 8},14)"><rect width="14" height="2" y="-4" fill="#a9a59b"/><text x="20">평소 (같은 계절 무강수일 평균)</text>
      <rect x="214" width="14" height="2" y="-4" fill="#0f6b5c"/><text x="234">날씨 반영 예측</text></g>
  </svg>`;
}

// ---------- 서비스 페이지 (today.html) ----------
async function renderService() {
  const app = document.getElementById("app");
  const q = new URLSearchParams(location.search);
  try {
    const [M, fc] = await Promise.all([loadModel(), loadForecast()]);
    const line = M.lines.includes(q.get("line")) ? q.get("line") : "전체";
    const offset = Math.min(Math.max(parseInt(q.get("day"), 10) || 0, 0), 6);
    const url = kw => "today.html?" + new URLSearchParams({ line, day: offset, ...kw });

    const p = predict(M, fc, line, offset);
    if (!p.rows.length) throw new Error("예보 데이터 없음");
    const rows = Object.fromEntries(p.rows.map(r => [r.hour, r]));
    const hours = p.rows.map(r => r.hour).sort((a, b) => hourKey(a) - hourKey(b));
    const now = kstNow();
    const defHour = offset === 0 ? now.hour : 8;
    let hour = parseInt(q.get("hour"), 10);
    if (Number.isNaN(hour)) hour = defHour;
    if (!(hour in rows)) hour = hours.reduce((b, h) => (Math.abs(h - hour) < Math.abs(b - hour) ? h : b));

    const tabs = [...Array(7).keys()].map(o => {
      const d = addDays(now.ymd, o);
      const label = o === 0 ? "오늘" : o === 1 ? "내일" : `${pad(d.m)}/${pad(d.d)}(${M.weekdays[d.wd]})`;
      return `<a class="${o === offset ? "on" : ""}" href="${url({ day: o, hour: "" })}">${label}</a>`;
    }).join("");

    const peak = Math.max(...p.rows.map(r => r.pred));
    const bars = hours.map(h => {
      const ratio = rows[h].pred / peak, [, name, css] = levelOf(ratio);
      return `<a class="bar ${css} ${h === hour ? "cur" : ""}" href="${url({ hour: h })}" title="${h}시 ${name}">` +
        `<span style="height: ${Math.round(ratio * 100)}%"></span><i>${h}</i></a>`;
    }).join("");

    const cur = rows[hour], [, level, css] = levelOf(cur.pred / peak);
    const sky = cur.snow > 0 ? "눈" : cur.rain > 0.1 ? "비" : "강수 없음";

    // 추천: 앞뒤 1시간 중 가장 한산한 시간
    const cand = [hour - 1, hour, hour + 1].filter(h => h in rows);
    const best = cand.reduce((b, h) => (rows[h].pred < rows[b].pred ? h : b));
    const gain = (1 - rows[best].pred / cur.pred) * 100;
    const tip = best === hour || gain < 5 ? "이 시간대가 앞뒤 한 시간 중 가장 한산하거나 비슷합니다."
      : `${Math.abs(best - hour)}시간 ${best < hour ? "일찍" : "늦게"}(${best}시) 타면 약 ${Math.round(gain)}% 덜 붐빕니다.`;

    // 호선별 비교 (그 호선 하루 최대 대비, 한산한 순)
    const cmp = M.lines.map(ln => {
      const o = predict(M, fc, ln, offset).rows, at = o.find(r => r.hour === hour);
      if (!at) return null;
      const ratio = at.pred / Math.max(...o.map(r => r.pred)), [, lv, c] = levelOf(ratio);
      return { ln, pct: Math.round(ratio * 100), lv, c };
    }).filter(Boolean).sort((a, b) => a.pct - b.pct).map(c =>
      `<li class="${c.ln === line ? "sel" : ""}"><a href="today.html?${new URLSearchParams({ line: c.ln, day: offset, hour })}">` +
      `<span class="ln">${c.ln}</span><span class="track"><span class="fill ${c.c}" style="width: ${c.pct}%"></span></span>` +
      `<span class="lv ${c.c}">${c.pct}%</span></a></li>`).join("");

    const rain = p.rows.reduce((s, r) => s + r.rain, 0), snow = p.rows.reduce((s, r) => s + r.snow, 0);
    const temps = p.rows.map(r => r.temp);
    const wsky = snow > 0 ? `눈 ${snow.toFixed(1)}cm` : rain > 0.1 ? `비 ${rain.toFixed(1)}mm` : "강수 없음";
    const effect = Math.abs(p.pct) < 1 ? "날씨 영향은 거의 없어요"
      : `평소보다 하루 이용량 약 ${Math.round(Math.abs(p.pct))}% ${p.pct < 0 ? "적을" : "많을"} 전망`;
    const ev = M.evidence;
    const opt = (v, sel, text = v) => `<option value="${v}" ${v === sel ? "selected" : ""}>${text}</option>`;

    app.innerHTML = `
    <nav class="tabs">${tabs}</nav>
    <p class="date">${pad(p.day.m)}월 ${pad(p.day.d)}일 (${M.weekdays[p.day.wd]}, ${p.dtype})</p>
    <section class="card weather">
      <div class="wmain">${Math.round(Math.min(...temps))}~${Math.round(Math.max(...temps))}℃ · ${wsky}</div>
      <div class="small">${effect}</div></section>
    <form class="pick" onsubmit="return false">
      <label>호선 <select id="sel-line">${opt("전체", line)}${M.lines.map(l => opt(l, line)).join("")}</select></label>
      <label>시간 <select id="sel-hour">${hours.map(h => opt(h, hour, h + "시")).join("")}</select></label>
    </form>
    <section class="card result ${css}"><div class="small">${hour}시 · ${line}</div><div class="big">${level}</div>
      <div class="meter" aria-hidden="true"><i></i><i></i><i></i><i></i></div>
      <div class="small">기온 ${Math.round(cur.temp)}℃ · ${sky}<br>평소보다 ${signed((cur.pred / cur.base - 1) * 100)}%</div></section>
    <p class="card tip"><span class="lbl">추천</span>${tip}</p>
    <section class="card"><h2>시간대별 이용 수준</h2><div class="bars">${bars}</div></section>
    <section class="card"><h2>${hour}시, 호선별 비교 <small>(한산한 순)</small></h2><ul class="cmp">${cmp}</ul>
      <p class="small">100%는 그 호선이 그날 가장 붐비는 시간이에요. 출근·퇴근 피크에는 대부분 호선이 100%에 가깝고, 호선끼리 승객 수를 비교한 것은 아니에요.</p></section>
    <p class="card small"><span class="lbl">근거</span>2025년 서울 지하철 승하차 데이터에서 ${ev[0].label} <b>${ev[0].value}</b>,
      ${ev[1].label} <b>${ev[1].value}</b>. 날씨 영향은 방향은 일관되지만 크기는 작아서, 이 서비스의 수준 표시는 대부분 요일·시간대 패턴에서 나옵니다.</p>`;

    document.getElementById("sel-line").onchange = e => { location.href = "today.html?" + new URLSearchParams({ line: e.target.value, day: offset, hour }); };
    document.getElementById("sel-hour").onchange = e => { location.href = url({ hour: e.target.value }); };
  } catch (e) {
    app.innerHTML = `<p class="card warn">예보를 불러오지 못했습니다 (${e.message}). 잠시 후 다시 시도해 주세요.</p>`;
  }
}
