"""서울교통공사 1~8호선 역별 일별 시간대별 승하차인원 원본을 가벼운 집계 파일로 줄인다.

입력 : data/raw/subway_*.csv  (cp949, 와이드: 시간대 20개 열)
       - 2025년: 서울 열린데이터광장 OA-22470 (승객유형 열 있음) → subway_1.csv, subway_2.csv
       - 2023·2024년: 공공데이터포털 15048032 과거 데이터 (승객유형 없음, 열 이름이 조금 다름)
         → subway_2023.csv, subway_2024.csv
출력 : data/subway_line_hourly.csv  date, hour, line, boardings, alightings          (2023~2025, 분석 본체)
       data/subway_hourly.csv       date, hour, line, passenger_type, boardings, alightings (승객유형 있는 2025년만)
       data/subway_station.csv      date, line, station, boardings, alightings      (역별 일합계, 2023~2025)

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


LAST_DATE = "2025-12-31"  # 마지막 날 24시 이후 → 다음 해 0시 이월분은 날씨 범위 밖이라 제외
# 원본 오류 구간: 2024년 4월 파일은 13-14시 칸에 13~18시 합계 비슷한 값이 들어가고 17시 이후 칸이
# 한 시간씩 밀려 있다(하루 합계가 평소의 약 1.4배, 8개 호선 30일 전부). 칸을 되돌릴 근거가 없어 통째로 뺀다.
BAD_DATES = ("2024-04-01", "2024-04-30")


def _read(p: Path) -> pd.DataFrame:
    """두 가지 원본 형식을 같은 열 이름으로 맞춘다."""
    df = pd.read_csv(p, encoding="cp949")
    df = df.rename(columns={"호선": "호선명"})
    df.columns = [c.replace("06시이전", "06시간대이전").replace("24시이후", "24시간대이후") for c in df.columns]
    if "승객유형" not in df.columns:
        df["승객유형"] = "전체"  # 공공데이터포털 과거 파일에는 승객유형 구분이 없다
    return df


def _to_hourly(g: pd.DataFrame, keys: list, hour_cols: list) -> pd.DataFrame:
    long = g.melt(id_vars=keys + ["kind"], value_vars=hour_cols, var_name="col", value_name="n")
    long["hour"] = long["col"].map(hour_of)
    # 24시 이후는 다음 날 0시로 이월
    late = long["hour"] == 24
    long.loc[late, "date"] = (pd.to_datetime(long.loc[late, "date"]) + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    long.loc[late, "hour"] = 0
    out = long.groupby(["date", "hour"] + [k for k in keys if k != "date"] + ["kind"])["n"].sum().unstack("kind").reset_index()
    return out[(out["date"] <= LAST_DATE) & ~out["date"].between(*BAD_DATES)]  # 전날 24시 이월분까지 제외


def main():
    frames = [_read(p) for p in sorted(RAW.glob("subway_*.csv"))]
    df = pd.concat(frames, ignore_index=True)
    bad = df["수송일자"].between(*BAD_DATES)
    print("원본 오류 구간 제외:", BAD_DATES, int(bad.sum()), "행")
    df = df[~bad]
    hour_cols = [c for c in df.columns if "시간대" in c]
    df[hour_cols] = df[hour_cols].fillna(0)

    df["passenger_type"] = df["승객유형"].replace(TYPE_MAP)
    df["kind"] = df["승하차구분"].map({"승차": "boardings", "하차": "alightings"})
    df = df.rename(columns={"수송일자": "date", "호선명": "line", "역명": "station"})

    # 역별 일합계
    st = df.groupby(["date", "line", "station", "kind"])[hour_cols].sum().sum(axis=1)
    st = st.unstack("kind").reset_index()
    st.to_csv(OUT / "subway_station.csv", index=False)

    # 시간별 호선 합계 (모든 해)
    g = df.groupby(["date", "line", "kind"])[hour_cols].sum().reset_index()
    line_hourly = _to_hourly(g, ["date", "line"], hour_cols)
    line_hourly.to_csv(OUT / "subway_line_hourly.csv", index=False)

    # 시간별 승객유형 (승객유형이 있는 해만)
    typed = df[df["passenger_type"] != "전체"]
    g = typed.groupby(["date", "line", "passenger_type", "kind"])[hour_cols].sum().reset_index()
    hourly = _to_hourly(g, ["date", "line", "passenger_type"], hour_cols)
    hourly.to_csv(OUT / "subway_hourly.csv", index=False)

    print("subway_line_hourly", line_hourly.shape, line_hourly["date"].min(), line_hourly["date"].max())
    print("subway_hourly", hourly.shape, hourly["date"].min(), hourly["date"].max())
    print("subway_station", st.shape)
    print("총 승차", int(hourly["boardings"].sum()), "총 하차", int(hourly["alightings"].sum()))


if __name__ == "__main__":
    main()
