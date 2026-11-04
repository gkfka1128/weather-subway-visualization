"""서울교통공사 1~8호선 역별 일별 시간대별 승객유형별 승하차인원(OA-22470) 원본을 가벼운 집계 파일로 줄인다.

입력 : data/raw/subway_*.csv  (cp949, 와이드: 시간대 19개 열)
출력 : data/subway_hourly.csv   date, hour, line, passenger_type, boardings, alightings
       data/subway_station.csv  date, line, station, boardings, alightings   (역별 일합계)

시간대 열 처리: '06시간대이전'→5시, 'HH-HH+1시간대'→HH시, '24시간대이후'→다음날 0시.
승객유형: 외국어 표기 유형(영어/일어/중국어 일반·어린이)은 일반·어린이에 합친다.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data"

TYPE_MAP = {
    "영어 일반": "일반", "일어 일반": "일반", "중국어 일반": "일반",
    "영어 어린이": "어린이", "일어 어린이": "어린이", "중국어 어린이": "어린이",
}


def hour_of(col: str) -> int:
    if col.startswith("06시간대이전"):
        return 5
    if col.startswith("24시간대이후"):
        return 24
    return int(col[:2])


def main():
    frames = [pd.read_csv(p, encoding="cp949") for p in sorted(RAW.glob("subway_*.csv"))]
    df = pd.concat(frames, ignore_index=True)
    hour_cols = [c for c in df.columns if "시간대" in c]

    df["passenger_type"] = df["승객유형"].replace(TYPE_MAP)
    df["kind"] = df["승하차구분"].map({"승차": "boardings", "하차": "alightings"})
    df = df.rename(columns={"수송일자": "date", "호선명": "line", "역명": "station"})

    # 역별 일합계
    st = df.groupby(["date", "line", "station", "kind"])[hour_cols].sum().sum(axis=1)
    st = st.unstack("kind").reset_index()
    st.to_csv(OUT / "subway_station.csv", index=False)

    # 시간별: 역을 합쳐 호선·승객유형 단위로
    g = df.groupby(["date", "line", "passenger_type", "kind"])[hour_cols].sum().reset_index()
    long = g.melt(id_vars=["date", "line", "passenger_type", "kind"], value_vars=hour_cols,
                  var_name="col", value_name="n")
    long["hour"] = long["col"].map(hour_of)
    # 24시 이후는 다음 날 0시로 이월
    late = long["hour"] == 24
    long.loc[late, "date"] = (pd.to_datetime(long.loc[late, "date"]) + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    long.loc[late, "hour"] = 0
    hourly = (long.groupby(["date", "hour", "line", "passenger_type", "kind"])["n"].sum()
              .unstack("kind").reset_index())
    hourly = hourly[hourly["date"] <= "2025-12-31"]  # 2026-01-01 0시 이월분은 날씨 범위 밖이라 제외
    hourly.to_csv(OUT / "subway_hourly.csv", index=False)

    print("subway_hourly", hourly.shape, hourly["date"].min(), hourly["date"].max())
    print("subway_station", st.shape)
    print("총 승차", int(hourly["boardings"].sum()), "총 하차", int(hourly["alightings"].sum()))


if __name__ == "__main__":
    main()
