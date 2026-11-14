"""날씨가 서울 지하철 이용량에 미치는 영향 — Flask + matplotlib/seaborn + mpld3 대시보드.

분석·보정 계수는 analysis.py, 그래프는 charts.py, 여기서는 날씨 예보 → 오늘의 예측과 웹 라우트를 맡는다.
실행: python app.py  →  http://127.0.0.1:5000
"""
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from flask import Flask, jsonify, render_template, request

import charts
from analysis import (DAY_TYPES, DF, LINES, WEEKDAYS, day_type_of, usual_by_hour,  # noqa: F401 (service가 core.로 사용)
                      weather_factor)

KST = ZoneInfo("Asia/Seoul")
app = Flask(__name__)


# ---------- 날씨 예보 (오늘 ~ 6일 뒤) ----------
_forecast_cache = {"at": 0.0, "data": None}


def fetch_forecast():
    if _forecast_cache["data"] is not None and time.time() - _forecast_cache["at"] < 1800:
        return _forecast_cache["data"]
    r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=10, params={
        "latitude": 37.5665, "longitude": 126.9780, "forecast_days": 7,
        "hourly": "temperature_2m,precipitation,snowfall,cloud_cover", "timezone": "Asia/Seoul"})
    r.raise_for_status()
    h = r.json()["hourly"]
    t = pd.to_datetime(h["time"])
    data = pd.DataFrame({"date": t.strftime("%Y-%m-%d"), "hour": t.hour, "temp_c": h["temperature_2m"],
                         "precip_mm": h["precipitation"], "snowfall_cm": h["snowfall"], "cloud": h["cloud_cover"]})
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
            "pct": pct, "hours": hours}


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
                           metrics=charts.headline())


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
