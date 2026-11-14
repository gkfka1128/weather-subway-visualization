"""GitHub Pages용 정적 사이트 빌드 → site/

Pages는 서버(Flask)를 돌릴 수 없으므로
- 대시보드: 호선(전체+8) × 요일 유형(전체+3) = 36개 화면을 미리 HTML로 만든다(그래프는 Flask 버전과 같다).
- 오늘 예측·서비스: 평소 이용량과 날씨 보정 계수를 model.json으로 내보내고,
  브라우저가 Open-Meteo 예보를 직접 받아 계산한다(static_site/predict.js).

실행: python build_static.py  →  site/index.html (로컬 확인: python -m http.server -d site)
"""
import json
import shutil
from datetime import date, timedelta
from pathlib import Path

from flask import render_template

import charts
from analysis import (DAY_TYPES, HOLIDAYS_2026, LINES, RAIN_F, SNOW_F, TEMP_F, WEEKDAYS, WIN,
                      usual_by_hour)
from app import app

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "site"
SRC = ROOT / "static_site"
ALL_LINES = ["전체"] + LINES
ALL_DTYPES = ["전체"] + DAY_TYPES
ANCHOR_STEP = 7  # 평소 이용량 기준일: 1/1부터 7일 간격(앞뒤 4주 창이라 가장 가까운 기준일로 충분)


def page_name(li: int, di: int) -> str:
    return "index.html" if li == 0 and di == 0 else f"dash-{li}-{di}.html"


def build_model() -> dict:
    """브라우저 예측에 필요한 숫자: 기준일별·시간별 평소 이용량 + 날씨 보정 계수."""
    anchors = [date(2025, 1, 1) + timedelta(days=ANCHOR_STEP * k) for k in range(366 // ANCHOR_STEP + 1)]
    usual = {}
    for line in ALL_LINES:
        usual[line] = {}
        for dtype in DAY_TYPES:
            rows = []
            for a in anchors:
                s = usual_by_hour(line, dtype, f"{a:%m-%d}")
                rows.append([round(float(s[h])) if h in s.index else None for h in range(24)])
            usual[line][dtype] = rows
    return {
        "lines": LINES, "weekdays": WEEKDAYS, "anchor_step": ANCHOR_STEP, "window_days": WIN,
        "holidays": sorted(HOLIDAYS_2026),
        "rain_f": RAIN_F, "temp_f": TEMP_F, "snow_f": SNOW_F,
        "usual": usual,
        "evidence": charts.headline()[:2],
    }


def build_dashboards():
    pages = {f"{ln}|{dt}": page_name(li, di) for li, ln in enumerate(ALL_LINES) for di, dt in enumerate(ALL_DTYPES)}
    with app.test_request_context():
        for li, line in enumerate(ALL_LINES):
            for di, dtype in enumerate(ALL_DTYPES):
                html = render_template(
                    "index.html", static_site=True, css_href="style.css", pages=pages,
                    sections=charts.sections(line, dtype), pred=None, error=None,
                    lines=LINES, day_types=DAY_TYPES, sel_line=line, sel_dtype=dtype, metrics=charts.headline())
                (OUT / page_name(li, di)).write_text(html, encoding="utf-8")
                print(f"  {page_name(li, di):16s} {line} / {dtype}")


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    print("model.json")
    (OUT / "model.json").write_text(json.dumps(build_model(), ensure_ascii=False, separators=(",", ":"), allow_nan=False),
                                    encoding="utf-8")
    print("dashboards")
    build_dashboards()
    shutil.copy(ROOT / "static" / "style.css", OUT / "style.css")
    shutil.copy(ROOT / "static" / "icons.svg", OUT / "icons.svg")
    shutil.copy(ROOT / "service" / "static" / "service.css", OUT / "service.css")
    for f in SRC.iterdir():
        shutil.copy(f, OUT / f.name)
    (OUT / ".nojekyll").touch()  # Pages가 파일을 가공하지 않도록
    total = sum(f.stat().st_size for f in OUT.iterdir()) / 1e6
    print(f"done: {len(list(OUT.iterdir()))} files, {total:.1f} MB → {OUT}")


if __name__ == "__main__":
    main()
