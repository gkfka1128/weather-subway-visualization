"""지하철 시간별 집계 + 날씨 + 공휴일 → data/merged.csv

단위: 날짜·시간·호선 (승객유형은 합산). 열:
date, hour, line, boardings, alightings, weekday(0=월), day_type(평일/토/일·공휴일),
temp_c, precip_mm, snowfall_cm, humidity, wind_ms, rain(강수 여부)
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data"


def main():
    sub = pd.read_csv(D / "subway_hourly.csv")
    sub = sub.groupby(["date", "hour", "line"], as_index=False)[["boardings", "alightings"]].sum()
    wx = pd.read_csv(D / "weather_hourly.csv")
    hol = set(pd.read_csv(D / "holidays.csv")["date"])

    m = sub.merge(wx, on=["date", "hour"], how="left", validate="many_to_one")
    assert m["temp_c"].notna().all(), "날씨와 매칭 안 된 행이 있음"

    dt = pd.to_datetime(m["date"])
    m["weekday"] = dt.dt.weekday
    m["day_type"] = "평일"
    m.loc[m["weekday"] == 5, "day_type"] = "토요일"
    m.loc[(m["weekday"] == 6) | m["date"].isin(hol), "day_type"] = "일·공휴일"
    m["rain"] = (m["precip_mm"] > 0.1).astype(int)

    m.to_csv(D / "merged.csv", index=False)
    print("subway rows", len(sub), "→ merged", m.shape)
    print(m["day_type"].value_counts().to_dict())


if __name__ == "__main__":
    main()
