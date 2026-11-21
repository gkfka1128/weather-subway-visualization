"""따릉이(공공자전거)·서울 버스 일별 원본을 서울 열린데이터광장에서 받아 가벼운 일별 집계로 줄인다.

원본 (cp949, 월별 CSV, 한 달 100~250MB → data/raw/bike/, data/raw/bus/ 에 저장, git 제외)
  - 따릉이 이용정보(일별) OA-15246 : 대여일자, 대여소번호, 대여소, 대여구분코드(정기권/일일권…), 성별, 연령대, 이용건수 …
  - 버스 노선별 정류장별 승하차(일별) OA-12912 : 사용일자, 노선번호, …, 표준버스정류장ID, 승차총승객수, 하차총승객수
출력
  - data/raw/bike_station_daily.csv : date, station_id, station, ticket, rentals   (대여소·대여구분별 일 대여 건수)
  - data/bus_daily.csv          : date, boardings, alightings                   (서울 버스 전체 일 합계)

다시 실행하면 이미 받은 원본은 건너뛴다. 사용법: python prep/fetch_mobility.py [bike|bus]
"""
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data"
URL = "https://datafile.seoul.go.kr/bigfile/iot/inf/nio_download.do?&useCache=false"

# 포털 파일 목록의 seq 번호 (2026-11-21 확인). 월 → seq
BIKE_SEQ = dict(zip(
    [f"{y}{m:02d}" for y in (23, 24, 25) for m in range(1, 13)],
    list(range(49, 61)) + list(range(61, 73)) + [73, 75, 76, 77, 78, 79, 81, 82, 83, 84, 85, 86]))
BUS_SEQ = {f"20{y}{m:02d}": 118 + (y - 23) * 12 + m - 1 for y in (23, 24, 25) for m in range(1, 13)}

SOURCES = {
    "bike": dict(inf="OA-15246", infseq=1, seqs=BIKE_SEQ, name="bike_{}.csv"),
    "bus": dict(inf="OA-12912", infseq=3, seqs=BUS_SEQ, name="bus_{}.csv"),
}


def download(kind: str, ym: str, seq: int) -> Path:
    src = SOURCES[kind]
    path = RAW / kind / src["name"].format(ym)
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with requests.post(URL, data={"infId": src["inf"], "seqNo": "", "seq": seq, "infSeq": src["infseq"]},
                       stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    head = tmp.read_bytes()[:200]
    if not any(k.encode(e) in head for k in ("대여일자", "사용일자") for e in ("cp949", "utf-8")):
        tmp.unlink()
        raise ValueError(f"{kind} {ym}: CSV가 아닌 응답 (포털 오류 페이지?) — seq 번호를 확인할 것")
    tmp.replace(path)
    return path


def read(path: Path, usecols) -> pd.DataFrame:
    return pd.read_csv(path, encoding="cp949", encoding_errors="replace", usecols=usecols,
                       dtype=str, low_memory=False)


def agg_bike(path: Path) -> pd.DataFrame:
    df = read(path, lambda c: c.strip() in ("대여일자", "대여소번호", "대여소", "대여구분코드", "이용건수"))
    df.columns = [c.strip() for c in df.columns]
    df["rentals"] = pd.to_numeric(df["이용건수"], errors="coerce").fillna(0)
    g = df.groupby(["대여일자", "대여소번호", "대여구분코드"], as_index=False).agg(
        station=("대여소", "first"), rentals=("rentals", "sum"))
    g.columns = ["date", "station_id", "ticket", "station", "rentals"]
    g["date"] = pd.to_datetime(g["date"].str.replace(".", "-")).dt.strftime("%Y-%m-%d")
    return g[["date", "station_id", "station", "ticket", "rentals"]]


def agg_bus(path: Path) -> pd.DataFrame:
    df = read(path, ["사용일자", "승차총승객수", "하차총승객수"])
    for c in ("승차총승객수", "하차총승객수"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    g = df.groupby("사용일자", as_index=False)[["승차총승객수", "하차총승객수"]].sum()
    g.columns = ["date", "boardings", "alightings"]
    g["date"] = pd.to_datetime(g["date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")
    return g


def run(kind: str):
    parts = []
    for ym, seq in SOURCES[kind]["seqs"].items():
        p = download(kind, ym, seq)
        part = (agg_bike if kind == "bike" else agg_bus)(p)
        print(kind, ym, f"{p.stat().st_size / 1e6:.0f}MB →", part.shape, flush=True)
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    out = out[(out["date"] >= "2023-01-01") & (out["date"] <= "2025-12-31")]
    # 따릉이 대여소별 일별 표는 수백만 행이라 저장소에 넣지 않고 raw/에 둔다 → prep/link_bike.py 가 더 줄인다
    path = RAW / "bike_station_daily.csv" if kind == "bike" else OUT / "bus_daily.csv"
    out.to_csv(path, index=False)
    print("saved", path, out.shape)


if __name__ == "__main__":
    for k in (sys.argv[1:] or ["bus", "bike"]):
        run(k)
