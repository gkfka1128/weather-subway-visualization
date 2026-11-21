"""따릉이 대여소별 일별 표(수백만 행)를 대시보드가 쓰는 작은 표 두 개로 줄인다.

입력  data/raw/bike_station_daily.csv (prep/fetch_mobility.py bike), data/station_info.csv (prep/stations.py)
출력  data/bike_daily.csv        date, commute, casual, total
          commute = 정기권(반복 이용 → 출퇴근·통학 성격), casual = 일일권(회원·비회원, 가끔 타는 이용)
          가족권·단체권(전체의 약 0.05%)은 성격이 섞여 있어 뺀다
          bike_near_station 은 대여가 0건인 날의 행이 없다(평균을 낼 때는 날짜를 채워서 쓸 것)
      data/bike_near_station.csv date, station, rentals   (지하철역 300m 안 대여소의 일 대여 합)
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def main():
    b = pd.read_csv(DATA / "raw" / "bike_station_daily.csv", dtype={"station_id": str})
    b["station_id"] = b["station_id"].str.zfill(5)
    kind = b["ticket"].map(lambda t: "commute" if t == "정기권" else "casual" if str(t).startswith("일일권") else None)
    daily = b.assign(kind=kind).dropna(subset=["kind"]).pivot_table(
        index="date", columns="kind", values="rentals", aggfunc="sum", fill_value=0)
    daily["total"] = daily["commute"] + daily["casual"]
    daily.astype(int).reset_index().to_csv(DATA / "bike_daily.csv", index=False)

    info = pd.read_csv(DATA / "station_info.csv", dtype={"bike_ids": str}).fillna({"bike_ids": ""})
    pairs = [(s, i) for s, ids in zip(info["station"], info["bike_ids"]) for i in ids.split(";") if i]
    link = pd.DataFrame(pairs, columns=["station", "station_id"])
    per = b.groupby(["date", "station_id"], as_index=False)["rentals"].sum()
    near = per.merge(link, on="station_id").groupby(["date", "station"], as_index=False)["rentals"].sum()
    near.to_csv(DATA / "bike_near_station.csv", index=False)
    print("bike_daily", daily.shape, "| bike_near_station", near.shape, "| 연결된 대여소", link["station_id"].nunique())


if __name__ == "__main__":
    main()
