"""이용자용 서비스 사이트 — "지하철, 이 시간에 얼마나 붐빌까".

발표용 대시보드(app.py)와 별개의 사이트. 예측 로직(평소 이용량 × 날씨 보정)은 app.py 것을 가져다 쓴다.
실행: python service/serve.py  →  http://127.0.0.1:5001
"""
import sys
from pathlib import Path
from urllib.parse import urlencode

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, render_template, request  # noqa: E402

import app as core  # noqa: E402  (데이터·보정 계수·날씨 예보)
import charts  # noqa: E402  (근거 숫자)

app = Flask(__name__, template_folder="templates", static_folder="static")

LEVELS = [(0.75, "매우 많음", "lv4"), (0.5, "많음", "lv3"), (0.3, "보통", "lv2"), (0.0, "한산", "lv1")]


def level_of(ratio: float):
    return next((name, css) for th, name, css in LEVELS if ratio >= th)


def advice(rows: dict, hour: int):
    """선택한 시간 앞뒤 1시간 중 이용량이 가장 적은 시간대를 추천한다."""
    cand = [h for h in (hour - 1, hour, hour + 1) if h in rows]
    best = min(cand, key=lambda h: rows[h]["pred"])
    here = rows[hour]["pred"]
    gain = (1 - rows[best]["pred"] / here) * 100 if here else 0
    if best == hour or gain < 5:
        return "이 시간대가 앞뒤 한 시간 중 가장 한산하거나 비슷합니다."
    when = "일찍" if best < hour else "늦게"
    return f"{abs(best - hour)}시간 {when}({best}시) 타면 약 {gain:.0f}% 덜 붐빕니다."


def weather_banner(out, pct):
    rain, snow = out["precip_mm"].sum(), out["snowfall_cm"].sum()
    lo, hi = out["temp_c"].min(), out["temp_c"].max()
    if snow > 0:
        icon, sky = "❄️", f"눈 {snow:.1f}cm"
    elif rain > 0.1:
        icon, sky = "☔", f"비 {rain:.1f}mm"
    else:
        icon, sky = "🌡️", "강수 없음"
    if abs(pct) < 1:
        effect = "날씨 영향은 거의 없어요"
    else:
        effect = f"평소보다 하루 이용량 약 {abs(pct):.0f}% {'적을' if pct < 0 else '많을'} 전망"
    return {"icon": icon, "sky": sky, "range": f"{lo:.0f}~{hi:.0f}℃", "effect": effect}


def day_label(offset: int, day) -> str:
    return "오늘" if offset == 0 else "내일" if offset == 1 else f"{day:%m/%d}({core.WEEKDAYS[day.weekday()]})"


@app.route("/")
def home():
    line = request.args.get("line", "전체")
    if line != "전체" and line not in core.LINES:
        line = "전체"
    try:
        offset = min(max(int(request.args.get("day", 0)), 0), 6)
    except ValueError:
        offset = 0

    def url(**kw):
        params = {"line": line, "day": offset, **kw}
        return "?" + urlencode(params)

    try:
        day, dtype, out, pct = core.predict(line, offset)
        if out.empty:
            raise ValueError("예보 데이터 없음")
        tabs = []
        for o in range(7):
            d = core.datetime.now(core.KST) + core.timedelta(days=o)
            tabs.append({"label": day_label(o, d), "url": url(day=o, hour=""), "active": o == offset})

        rows = {int(r.hour): {"pred": float(r.pred), "base": float(r.base), "temp": r.temp_c,
                              "rain": r.precip_mm, "snow": r.snowfall_cm} for r in out.itertuples()}
        hours = sorted(rows, key=lambda h: (h < 5, h))  # 0시(자정 이후)는 맨 뒤로
        default_hour = core.datetime.now(core.KST).hour if offset == 0 else 8
        try:
            hour = int(request.args.get("hour", default_hour))
        except ValueError:
            hour = default_hour
        hour = hour if hour in rows else min(hours, key=lambda h: abs(h - hour))

        peak = max(r["pred"] for r in rows.values())
        bars = []
        for h in hours:
            name, css = level_of(rows[h]["pred"] / peak)
            bars.append({"hour": h, "pct": round(rows[h]["pred"] / peak * 100), "css": css, "name": name,
                         "current": h == hour, "url": url(hour=h)})
        cur = rows[hour]
        name, css = level_of(cur["pred"] / peak)
        sky = "눈" if cur["snow"] > 0 else "비" if cur["rain"] > 0.1 else "강수 없음"

        # 호선별 비교: 선택한 시간이 그 호선의 하루 최대 대비 어느 정도인지 (한산한 순)
        compare = []
        for ln in core.LINES:
            _, _, o, _ = core.predict(ln, offset)
            if hour not in set(o["hour"]):
                continue
            ratio = float(o.loc[o["hour"] == hour, "pred"].iloc[0]) / float(o["pred"].max())
            lname, lcss = level_of(ratio)
            compare.append({"line": ln, "pct": round(ratio * 100), "level": lname, "css": lcss,
                            "selected": ln == line, "url": f"?{urlencode({'line': ln, 'day': offset, 'hour': hour})}"})
        compare.sort(key=lambda c: c["pct"])

        return render_template(
            "home.html", error=None, lines=core.LINES, sel_line=line, sel_day=offset, hour=hour, hours=hours,
            bars=bars, level=name, css=css, temp=cur["temp"], sky=sky, tip=advice(rows, hour),
            vs_usual=(cur["pred"] / cur["base"] - 1) * 100, tabs=tabs, compare=compare,
            banner=weather_banner(out, pct), evidence=charts.headline()[:2],
            date_text=f"{day:%m월 %d일} ({core.WEEKDAYS[day.weekday()]}, {dtype})", is_today=offset == 0)
    except Exception as e:
        return render_template("home.html", error=f"예보를 불러오지 못했습니다 ({type(e).__name__}). 잠시 후 다시 시도해 주세요.",
                               lines=core.LINES, sel_line=line, sel_day=offset)


if __name__ == "__main__":
    app.run(port=5001, debug=True)
