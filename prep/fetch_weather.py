"""Open-Meteo에서 서울 2025년 시간별 날씨를 받아 data/weather_hourly.csv 로 저장하고,
2025년 공휴일 목록 data/holidays.csv 도 만든다. (B가 만든 파일이 있으면 같은 형식으로 교체하면 된다)

출력 열: date, hour, temp_c, precip_mm, snowfall_cm, humidity, wind_ms
"""
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data"

URL = "https://archive-api.open-meteo.com/v1/archive"
PARAMS = {
    "latitude": 37.5665, "longitude": 126.9780,
    "start_date": "2025-01-01", "end_date": "2025-12-31",
    "hourly": "temperature_2m,precipitation,snowfall,relative_humidity_2m,wind_speed_10m",
    "wind_speed_unit": "ms", "timezone": "Asia/Seoul",
}

# 2025년 법정 공휴일 + 대체·임시 공휴일
HOLIDAYS = [
    ("2025-01-01", "신정"), ("2025-01-27", "임시공휴일"),
    ("2025-01-28", "설날 연휴"), ("2025-01-29", "설날"), ("2025-01-30", "설날 연휴"),
    ("2025-03-01", "삼일절"), ("2025-03-03", "삼일절 대체"),
    ("2025-05-05", "어린이날·부처님오신날"), ("2025-05-06", "대체공휴일"),
    ("2025-06-03", "대통령선거"), ("2025-06-06", "현충일"),
    ("2025-08-15", "광복절"),
    ("2025-10-03", "개천절"), ("2025-10-05", "추석 연휴"), ("2025-10-06", "추석"),
    ("2025-10-07", "추석 연휴"), ("2025-10-08", "추석 대체"), ("2025-10-09", "한글날"),
    ("2025-12-25", "성탄절"),
]


def main():
    r = requests.get(URL, params=PARAMS, timeout=60)
    r.raise_for_status()
    h = r.json()["hourly"]
    t = pd.to_datetime(h["time"])
    df = pd.DataFrame({
        "date": t.strftime("%Y-%m-%d"), "hour": t.hour,
        "temp_c": h["temperature_2m"], "precip_mm": h["precipitation"],
        "snowfall_cm": h["snowfall"], "humidity": h["relative_humidity_2m"],
        "wind_ms": h["wind_speed_10m"],
    })
    df.to_csv(OUT / "weather_hourly.csv", index=False)
    pd.DataFrame(HOLIDAYS, columns=["date", "name"]).to_csv(OUT / "holidays.csv", index=False)
    print("weather_hourly", df.shape, "결측", int(df.isna().sum().sum()))
    print(df.describe().round(1).to_string())


if __name__ == "__main__":
    main()
