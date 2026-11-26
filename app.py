"""날씨는 서울 시민의 이동을 어떻게 바꾸는가(지하철·버스·따릉이) — Flask + matplotlib/seaborn + mpld3 대시보드.

분석·보정 계수는 analysis.py, 그래프는 charts.py, 여기서는 날씨 예보 → 오늘의 예측과 웹 라우트를 맡는다.
예보는 기상청 단기예보 API(인증키 KMA_SERVICE_KEY가 있을 때)를 쓰고, 기상청이 주지 않는 4일 뒤부터와 키가 없을 때는 Open-Meteo를 쓴다.
실행: python app.py  →  http://127.0.0.1:5000
"""
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote, unquote
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from flask import Flask, jsonify, render_template, request

import charts
from analysis import (DAY_TYPES, DF, LINES, WEEKDAYS, day_type_of, usual_by_hour,  # noqa: F401 (service가 core.로 사용)
                      weather_factor)

KST = ZoneInfo("Asia/Seoul")
app = Flask(__name__)

# 인증키는 저장소에 올리지 않는 .env 파일(또는 환경변수)에서 읽는다. 형식은 .env.example 참고.
_env = Path(__file__).with_name(".env")
if _env.exists():
    for _line in _env.read_text(encoding="utf-8-sig").splitlines():  # utf-8-sig: 메모장이 붙이는 BOM 제거
        _k, _sep, _v = _line.partition("=")
        if _sep and not _k.lstrip().startswith("#"):
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))


# ---------- 날씨 예보 (오늘 ~ 6일 뒤) ----------
_forecast_cache = {"at": 0.0, "data": None}
KMA_URL = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
KMA_GRID = {"nx": 60, "ny": 127}   # 서울 중구(서울시청)의 기상청 격자 좌표
KMA_SKY = {1: 10, 3: 50, 4: 90}    # 하늘상태 코드(맑음·구름많음·흐림) → 구름량(%)
FC_COLS = ["temp_c", "precip_mm", "snowfall_cm", "cloud", "pop", "humidity", "wind_ms"]


def _kma_amount(v) -> float:
    """기상청 강수량(PCP)·적설(SNO) 값은 글자다: '강수없음' → 0, '1mm 미만' → 0.5, '3.2mm' → 3.2, '30.0~50.0mm' → 30."""
    m = re.search(r"[\d.]+", str(v))
    if not m:
        return 0.0
    return float(m.group()) / 2 if "미만" in str(v) else float(m.group())


def _kma_items(key: str, base: datetime) -> pd.DataFrame:
    """기상청 단기예보 한 번(발표 시각 base)을 시간별 표로. 열은 FC_COLS, 색인은 (date, hour)."""
    r = requests.get(KMA_URL, timeout=10, params={
        "serviceKey": unquote(key), "dataType": "JSON", "pageNo": 1, "numOfRows": 2000,  # 한 번 발표가 1000개 남짓이라 한 쪽에 다 받는다
        "base_date": f"{base:%Y%m%d}", "base_time": f"{base:%H}00", **KMA_GRID})
    r.raise_for_status()
    body = r.json()["response"]
    if body["header"]["resultCode"] != "00":
        raise ValueError(f"기상청 API 오류 {body['header']['resultCode']}: {body['header']['resultMsg']}")
    it = pd.DataFrame(body["body"]["items"]["item"])
    w = it.pivot_table(index=["fcstDate", "fcstTime"], columns="category", values="fcstValue", aggfunc="first")
    # 발표 끝자락에 일부 항목이 빠진 시각이 있어도 전체를 버리지 않는다(없는 강수·적설은 0, 하늘상태는 구름많음)
    w = w.reindex(columns=["TMP", "PCP", "SNO", "SKY", "POP", "REH", "WSD"])
    w[["PCP", "SNO"]] = w[["PCP", "SNO"]].fillna("0")
    w["SKY"] = w["SKY"].fillna("3")
    d = pd.to_datetime(w.index.get_level_values(0) + w.index.get_level_values(1), format="%Y%m%d%H%M")
    out = pd.DataFrame({"temp_c": w["TMP"].astype(float).to_numpy(), "precip_mm": w["PCP"].map(_kma_amount).to_numpy(),
                        "snowfall_cm": w["SNO"].map(_kma_amount).to_numpy(), "cloud": w["SKY"].astype(int).map(KMA_SKY).to_numpy(),
                        "pop": w["POP"].astype(float).to_numpy(), "humidity": w["REH"].astype(float).to_numpy(),
                        "wind_ms": w["WSD"].astype(float).to_numpy()},
                       index=pd.MultiIndex.from_arrays([d.strftime("%Y-%m-%d"), d.hour], names=["date", "hour"]))
    return out.dropna(subset=["temp_c"])


def fetch_kma(key: str, now: datetime) -> tuple[pd.DataFrame, datetime]:
    """기상청 단기예보. 하루 8번(02·05·…·23시) 발표되고 10분 뒤부터 조회된다.
    최신 발표는 그 시각 이후만 담고 있어, 오늘 0시부터 채우려고 어제 23시 발표를 먼저 깔고 최신 발표로 덮어쓴다."""
    t = now - timedelta(minutes=10)
    latest = t.replace(hour=(t.hour - 2) // 3 * 3 + 2, minute=0, second=0, microsecond=0) if t.hour >= 2 \
        else (t - timedelta(days=1)).replace(hour=23, minute=0, second=0, microsecond=0)
    first = (now - timedelta(days=1)).replace(hour=23, minute=0, second=0, microsecond=0)
    try:
        newest = _kma_items(key, latest)
    except ValueError:  # 발표 직후 아직 안 올라온 경우(NO_DATA) → 바로 전 발표를 쓴다
        latest -= timedelta(hours=3)
        newest = _kma_items(key, latest)
    frames = [newest] if first == latest else [_kma_items(key, first), newest]
    both = pd.concat(frames)
    return both[~both.index.duplicated(keep="last")], latest


def fetch_forecast():
    if _forecast_cache["data"] is not None and time.time() - _forecast_cache["at"] < 1800:
        return _forecast_cache["data"]
    r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=10, params={
        "latitude": 37.5665, "longitude": 126.9780, "forecast_days": 7, "wind_speed_unit": "ms",
        "hourly": "temperature_2m,precipitation,snowfall,cloud_cover,precipitation_probability,relative_humidity_2m,wind_speed_10m",
        "timezone": "Asia/Seoul"})
    r.raise_for_status()
    h = r.json()["hourly"]
    t = pd.to_datetime(h["time"])
    data = pd.DataFrame({"temp_c": h["temperature_2m"], "precip_mm": h["precipitation"], "snowfall_cm": h["snowfall"],
                         "cloud": h["cloud_cover"], "pop": h["precipitation_probability"], "humidity": h["relative_humidity_2m"],
                         "wind_ms": h["wind_speed_10m"]},
                        index=pd.MultiIndex.from_arrays([t.strftime("%Y-%m-%d"), t.hour], names=["date", "hour"]))
    data = data.apply(pd.to_numeric, errors="coerce")  # Open-Meteo가 빈 값(null)을 주는 시간이 있다
    data["source"] = "Open-Meteo 예보"
    key = os.environ.get("KMA_SERVICE_KEY", "").strip()
    if key:
        try:  # 기상청이 실패해도(키 오류·점검·응답 형식) Open-Meteo 값으로 계속 보여 준다
            kma, base = fetch_kma(key, datetime.now(KST).replace(tzinfo=None))
            hit = data.index.intersection(kma.index)
            data.loc[hit, FC_COLS] = kma.loc[hit, FC_COLS].to_numpy()
            data.loc[hit, "source"] = f"기상청 단기예보 · {base:%H}시 발표"
        except Exception as e:  # 오류 문장에 요청 주소(=인증키)가 들어갈 수 있어 키를 가리고 남긴다
            msg = str(e)
            for k in {key, unquote(key), quote(unquote(key), safe="")}:
                msg = msg.replace(k, "***")
            app.logger.warning("기상청 예보 실패, Open-Meteo로 대체: %s: %s", type(e).__name__, msg)
    data = data.reset_index()
    _forecast_cache.update(at=time.time(), data=data)
    return data


def predict(line: str = "전체", offset: int = 0):
    """offset일 뒤(0=오늘)의 시간대별 예측. 반환: (날짜, 요일유형, 시간별 표, 하루 합계 변화율%)."""
    day = datetime.now(KST) + timedelta(days=offset)
    dtype = day_type_of(day)
    fc = fetch_forecast()
    out = fc[fc["date"] == f"{day:%Y-%m-%d}"].copy()
    out["base"] = out["hour"].map(usual_by_hour(line, dtype, f"{day:%m-%d}"))
    out = out.dropna(subset=["base"]).copy()
    if out.empty:  # 예보가 그날을 아직/더 이상 포함하지 않으면 nan 문장 대신 오류로 처리
        raise ValueError("예보 데이터 없음")
    out["factor"] = [weather_factor(a, b, c) for a, b, c in zip(out.temp_c, out.precip_mm, out.snowfall_cm)]
    out["pred"] = out["base"] * out["factor"]
    total_pct = (out["pred"].sum() / out["base"].sum() - 1) * 100
    return day, dtype, out, total_pct


def summary_sentence(now, dtype, out, pct) -> str:
    rain = out["precip_mm"].sum()
    snow = out["snowfall_cm"].sum()
    lo, hi = out["temp_c"].min(), out["temp_c"].max()
    sky = f"눈(합계 {snow:.1f}cm)" if snow > 0 else f"비(합계 {rain:.1f}mm)" if rain > 0.1 else "강수 없음"
    word = "비슷할" if abs(pct) < 1 else ("적을" if pct < 0 else "많을")
    mag = "" if abs(pct) < 1 else f" (약 {abs(pct):.1f}%)"
    return (f"오늘({now:%m/%d} {WEEKDAYS[now.weekday()]}, {dtype}) 서울 날씨는 기온 {lo:.0f}~{hi:.0f}℃, {sky}입니다. "
            f"그래서 지하철 이용량은 평소 {dtype}보다 {word} 것으로 예측됩니다{mag}.")


def sky_icon(rain, snow, cloud) -> str:
    """icons.svg 심볼 이름. 강수가 없으면 구름량(%)으로 맑음/구름 조금/흐림을 고른다."""
    if snow > 0:
        return "snow"
    if rain > 0.1:
        return "rain"
    return "sun" if cloud < 30 else "partly" if cloud < 70 else "cloud"


def today_panel(out, pct) -> dict:
    """대시보드 '오늘 예측'의 날씨 카드와 3시간 간격 날씨 띠."""
    rain, snow = out["precip_mm"].sum(), out["snowfall_cm"].sum()
    hours = []
    for h in (6, 9, 12, 15, 18, 21):
        r = out[out["hour"] == h]
        if len(r):
            r = r.iloc[0]
            hours.append({"hour": h, "temp": round(r.temp_c), "icon": sky_icon(r.precip_mm, r.snowfall_cm, r.cloud),
                          "amt": f"{r.snowfall_cm:.1f}cm" if r.snowfall_cm > 0 else f"{r.precip_mm:.1f}mm" if r.precip_mm > 0.1 else "–"})
    return {"icon": sky_icon(rain, snow, out["cloud"].mean()),
            "range": f"{out['temp_c'].min():.0f}~{out['temp_c'].max():.0f}℃",
            "sky": f"눈 {snow:.1f}cm" if snow > 0 else f"비 {rain:.1f}mm" if rain > 0.1 else "강수 없음",
            "pct": pct, "hours": hours, "source": out["source"].mode().iat[0],
            "extra": [("강수확률", f"최대 {out['pop'].max():.0f}%"), ("습도", f"평균 {out['humidity'].mean():.0f}%"),
                      ("바람", f"최대 {out['wind_ms'].max():.1f}m/s")]}


def lines_today() -> list:
    """오늘 호선별 예상 하루 이용량(만 명)과 평소 대비 변화."""
    rows = []
    for ln in LINES:
        try:  # 한 호선이 비어도 오늘 카드 전체를 잃지 않게 그 호선만 뺀다 (predict.js와 같은 동작)
            _, _, o, p = predict(ln)
        except ValueError:
            continue
        rows.append({"line": ln, "no": ln[:-2], "total": o["pred"].sum() / 10000, "pct": p})
    top = max((r["total"] for r in rows), default=0) or 1
    for r in rows:
        r["w"] = round(r["total"] / top * 100)
    return rows


# ---------- 라우트 ----------
@app.route("/")
def index():
    line = request.args.get("line", "전체")
    dtype = request.args.get("day_type", "전체")
    if line != "전체" and line not in LINES:
        line = "전체"
    if dtype != "전체" and dtype not in DAY_TYPES:
        dtype = "전체"

    pred_card, error = None, None
    try:
        now, pdtype, out, pct = predict(line)
        pred_card = {"sentence": summary_sentence(now, pdtype, out, pct), "html": charts.forecast_html(out),
                     "wx": today_panel(out, pct), "lines": lines_today()}
    except Exception as e:  # 예보 API 실패 시에도 나머지 페이지는 보여준다
        error = f"오늘 예보를 불러오지 못했습니다 ({type(e).__name__})."
    return render_template("index.html", sections=charts.sections(line, dtype), pred=pred_card, error=error,
                           lines=LINES, day_types=DAY_TYPES, sel_line=line, sel_dtype=dtype,
                           metrics=charts.headline(), verdicts=charts.verdicts())


@app.route("/api/predict")
def api_predict():
    line = request.args.get("line", "전체")
    if line != "전체" and line not in LINES:
        line = "전체"
    try:
        now, dtype, out, pct = predict(line)
    except Exception as e:
        return jsonify({"error": f"예보를 불러오지 못했습니다 ({type(e).__name__})"}), 503
    return jsonify({"date": f"{now:%Y-%m-%d}", "day_type": dtype, "line": line,
                    "change_pct": round(pct, 1), "message": summary_sentence(now, dtype, out, pct),
                    "hourly": [{"hour": int(r.hour), "expected": round(r.pred), "usual": round(r.base),
                                "temp_c": r.temp_c, "precip_mm": r.precip_mm} for r in out.itertuples()]})


if __name__ == "__main__":
    app.run(debug=True)
