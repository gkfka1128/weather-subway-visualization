"""역 성격(유형)을 데이터로 나누고, 역마다 주변 따릉이 대여소를 붙인다.

입력
  data/raw/subway_*.csv            지하철 원본 (aggregate_subway.py 와 같은 파일)
  data/raw/station_master.json     서울시 역사마스터 정보 (OA-21232, 역 좌표)
  data/raw/bike_stations_2512.xlsx 따릉이 대여소 정보 (OA-13252, 25.12월 기준 좌표)
출력
  data/station_info.csv : station, lat, lon, type, weekend_ratio, am_in_ratio, wd_daily, bike_ids
      type  업무지구   평일 아침(7~9시) 하차가 승차의 2배 이상 — 일하러 오는 역
            주거지     평일 아침 승차가 하차의 2배 이상 — 집에서 나가는 역
            여가·상업  주말 하루 이용이 평일의 75% 이상 — 주말에도 붐비는 역
            혼합       나머지
      bike_ids  역에서 300m 안의 따릉이 대여소 번호 (세미콜론 구분)
"""

import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "station_info.csv"
BAD_DATES = ("2024-04-01", "2024-04-30")  # aggregate_subway.py 와 같은 원본 오류 구간
HOLI = set(pd.read_csv(ROOT / "data" / "holidays.csv")["date"])
NEAR_M = 300
RENAMED = {"당고개": "불암산", "자양": "뚝섬유원지"}  # 승하차 원본 앞쪽 해는 옛 이름 → 새 이름으로 합친다


def base_name(s: str) -> str:
    """'서울역' / '서울(경부선)' / '고속터미널(3)' 처럼 제각각인 역 이름을 맞춘다."""
    s = re.sub(r"\(.*?\)", "", str(s)).strip()
    return s[:-1] if s.endswith("역") and len(s) > 2 else s


def profile() -> pd.DataFrame:
    """역별 평일 하루 평균, 주말 하루 평균, 평일 아침 7~9시 승차·하차 평균."""
    rows = []
    for p in sorted(RAW.glob("subway_*.csv")):
        df = pd.read_csv(p, encoding="cp949")
        df.columns = [c.replace("06시이전", "06시간대이전").replace("24시이후", "24시간대이후") for c in df.columns]
        df = df[~df["수송일자"].between(*BAD_DATES)]
        hours = [c for c in df.columns if "시간대" in c]
        am = [c for c in hours if c[:2] in ("07", "08")]
        df[hours] = df[hours].fillna(0)
        day = df[hours].sum(axis=1)
        morning = df[am].sum(axis=1)
        rows.append(pd.DataFrame({"date": df["수송일자"], "station": df["역명"].map(base_name).replace(RENAMED),
                                  "kind": df["승하차구분"], "day": day, "am": morning}))
    d = pd.concat(rows).groupby(["date", "station", "kind"], as_index=False)[["day", "am"]].sum()
    dt = pd.to_datetime(d["date"])
    d["weekend"] = (dt.dt.dayofweek >= 5) | d["date"].isin(HOLI)
    daily = d.groupby(["station", "date", "weekend"], as_index=False)["day"].sum()
    wd = daily[~daily["weekend"]].groupby("station")["day"].mean()
    we = daily[daily["weekend"]].groupby("station")["day"].mean()
    am = d[~d["weekend"]].groupby(["station", "kind"])["am"].mean().unstack()
    out = pd.DataFrame({"wd_daily": wd, "weekend_ratio": we / wd,
                        "am_in_ratio": am["하차"] / am["승차"].clip(lower=1)}).reset_index()
    return out


def classify(r) -> str:
    if r.am_in_ratio >= 2:
        return "업무지구"
    if r.am_in_ratio <= 0.5:
        return "주거지"
    if r.weekend_ratio >= 0.75:
        return "여가·상업"
    return "혼합"


def station_coords() -> pd.DataFrame:
    txt = (RAW / "station_master.json").read_text(encoding="utf-8")
    # 키에 따옴표가 없고 끝 쉼표가 있는 JS 객체라 JSON으로 못 읽는다 → 항목마다 정규식으로 뽑는다
    rows = [dict(re.findall(r'(\w+):"([^"]*)"', item)) for item in re.findall(r"\{[^{}]*BLDN_NM[^{}]*\}", txt)]
    df = pd.DataFrame(rows)[["BLDN_NM", "ROUTE", "LAT", "LOT"]]
    df["station"] = df["BLDN_NM"].map(base_name)
    df[["LAT", "LOT"]] = df[["LAT", "LOT"]].astype(float)
    # 같은 이름의 다른 역(예: 5호선 양평 ↔ 경의중앙선 양평, 53km 떨어짐)이 섞이지 않게
    # 승하차 자료와 같은 서울교통공사 1~8호선 좌표를 먼저 쓰고, 없을 때만 다른 노선 좌표를 쓴다
    df["main"] = df["ROUTE"].isin([f"{i}호선" for i in range(1, 9)])
    df = df.sort_values("main", ascending=False)
    df = df[df["main"] | ~df["station"].isin(df.loc[df["main"], "station"])]
    return df.groupby("station", as_index=False).agg(lat=("LAT", "mean"), lon=("LOT", "mean"))


def read_xlsx(path: Path) -> pd.DataFrame:
    """openpyxl 없이 첫 시트를 읽는다 (공유 문자열 + 셀 값)."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns):
                shared.append("".join(t.text or "" for t in si.iter(f"{{{ns['m']}}}t")))
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in sheet.iter(f"{{{ns['m']}}}row"):
        cells = {}
        for c in row.findall("m:c", ns):
            col = re.match(r"[A-Z]+", c.get("r")).group()
            v = c.find("m:v", ns)
            if v is None:
                t = c.find("m:is/m:t", ns)
                val = t.text if t is not None else None
            else:
                val = shared[int(v.text)] if c.get("t") == "s" else v.text
            cells[col] = val
        rows.append(cells)
    return pd.DataFrame(rows)


def bike_coords() -> pd.DataFrame:
    # 머리글이 여러 줄로 병합된 표: A=대여소 번호, E=위도, F=경도. 번호가 숫자인 행만 데이터
    x = read_xlsx(RAW / "bike_stations_2512.xlsx")
    x = x[x["A"].astype(str).str.strip().str.fullmatch(r"\d+")]
    out = pd.DataFrame({"bike_id": x["A"].str.strip().str.zfill(5),
                        "lat": pd.to_numeric(x["E"], errors="coerce"),
                        "lon": pd.to_numeric(x["F"], errors="coerce")})
    return out.dropna()


def main():
    cache = RAW / "station_profile_cache.csv"  # 원본 4개를 읽는 데 오래 걸려 한 번만 계산
    if cache.exists():
        prof = pd.read_csv(cache)
    else:
        prof = profile()
        prof.to_csv(cache, index=False)
    prof["type"] = prof.apply(classify, axis=1)
    coords = station_coords()
    info = prof.merge(coords, on="station", how="left")
    miss = info["lat"].isna()
    print("좌표 없는 역:", info.loc[miss, "station"].tolist())

    bikes = bike_coords()
    lat0 = np.radians(37.55)
    def near(r):
        if pd.isna(r.lat):
            return ""
        dy = (bikes["lat"] - r.lat) * 111_000
        dx = (bikes["lon"] - r.lon) * 111_000 * np.cos(lat0)
        return ";".join(bikes.loc[np.hypot(dx, dy) <= NEAR_M, "bike_id"])
    info["bike_ids"] = info.apply(near, axis=1)
    info = info.round({"weekend_ratio": 3, "am_in_ratio": 3, "wd_daily": 0, "lat": 5, "lon": 5})
    info.to_csv(OUT, index=False)
    print(info["type"].value_counts().to_string())
    print("따릉이 대여소 수", len(bikes), "| 300m 안에 대여소 있는 역",
          int((info["bike_ids"] != "").sum()), "/", len(info))


if __name__ == "__main__":
    main()
