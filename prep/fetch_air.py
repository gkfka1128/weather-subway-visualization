"""서울시 일별 평균 대기오염도(측정소 실측, 서울 열린데이터광장 OA-2218) 2023~2025년을 받아
서울 전체 측정소(도시대기 25개 구 + 도로변·배경 등 약 50곳) 평균으로 줄여 data/air_daily.csv 로 저장한다.

출력 열: date, pm10, pm2_5   (단위 ㎍/㎥, 측정소 일평균의 평균)
"""
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "air"
OUT = ROOT / "data" / "air_daily.csv"
URL = "https://datafile.seoul.go.kr/bigfile/iot/inf/nio_download.do?&useCache=false"
SEQ = {2023: 20, 2024: 22, 2025: 23}  # 포털 파일 목록 번호 (2026-11-21 확인)


def read_any(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp949"):  # 해마다 인코딩이 다르다 (2024년은 BOM 붙은 UTF-8)
        try:
            return pd.read_csv(path, encoding=enc, dtype={"측정일시": str})
        except UnicodeDecodeError:
            continue
    raise ValueError(f"인코딩을 알 수 없음: {path}")


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    parts = []
    for y, seq in SEQ.items():
        p = RAW / f"air_{y}.csv"
        if not p.exists():
            r = requests.post(URL, data={"infId": "OA-2218", "seqNo": "", "seq": seq, "infSeq": 3}, timeout=60)
            r.raise_for_status()
            if "측정일시".encode("utf-8") not in r.content[:300] and "측정일시".encode("cp949") not in r.content[:300]:
                raise ValueError(f"{y}년 파일이 CSV가 아님 (포털 오류 페이지?) — seq 번호를 확인할 것")
            p.write_bytes(r.content)
        df = read_any(p)
        df.columns = [c.strip().strip('"') for c in df.columns]
        parts.append(df[["측정일시", "측정소명", "미세먼지농도(㎍/㎥)", "초미세먼지농도(㎍/㎥)"]])
    df = pd.concat(parts, ignore_index=True)
    df.columns = ["date", "site", "pm10", "pm2_5"]
    for c in ("pm10", "pm2_5"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"].str.strip('"').str[:8], format="%Y%m%d").dt.strftime("%Y-%m-%d")
    out = df.groupby("date", as_index=False)[["pm10", "pm2_5"]].mean().round(1)
    out.to_csv(OUT, index=False)
    print("air_daily", out.shape, "측정소", df["site"].nunique(), "결측일", int(out.isna().any(axis=1).sum()))


if __name__ == "__main__":
    main()
