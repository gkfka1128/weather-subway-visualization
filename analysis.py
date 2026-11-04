"""데이터 로딩, 계절 보정 기준선, 날씨 보정 계수, 백테스트.

핵심 아이디어
- "날씨 때문에 이용량이 변했다"를 말하려면 계절·연휴 영향을 먼저 걷어내야 한다.
  그래서 같은 요일 유형·같은 시간대의 '앞뒤 4주 무강수일 평균'을 그날의 기준선(lb)으로 두고,
  실제 이용량 / 기준선 = ratio(1이면 평소와 같음)를 날씨 구간별로 비교한다.
"""
from datetime import date as _date
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
WIN = 28  # 기준선 창(일)
WEEKDAYS = ["월", "화", "수", "목", "금", "토", "일"]
DAY_TYPES = ["평일", "토요일", "일·공휴일"]
WEATHER_COLS = ["temp_c", "precip_mm", "snowfall_cm", "humidity", "wind_ms"]


# ---------- 로딩 ----------
def load_data() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "merged.csv")
    df["use"] = df["boardings"] + df["alightings"]  # 이용량 = 승차 + 하차
    return df


DF = load_data()
LINES = sorted(DF["line"].unique())
_HOUR_WEATHER = DF.groupby(["date", "hour"], as_index=False)[WEATHER_COLS + ["day_type", "weekday"]].first()


def _load_passenger_types() -> pd.DataFrame:
    s = pd.read_csv(ROOT / "data" / "subway_hourly.csv")
    s["use"] = s["boardings"] + s["alightings"]
    g = s.groupby(["date", "hour", "passenger_type"], as_index=False)["use"].sum()
    return g.merge(_HOUR_WEATHER, on=["date", "hour"], how="inner", validate="many_to_one")


def _load_stations() -> pd.DataFrame:
    s = pd.read_csv(ROOT / "data" / "subway_station.csv")
    s["use"] = s["boardings"] + s["alightings"]
    day_type = DF.drop_duplicates("date").set_index("date")["day_type"]
    s["day_type"] = s["date"].map(day_type)
    return s


PT = _load_passenger_types()
ST = _load_stations()
PTYPES = ["일반", "우대권", "청소년", "어린이", "직원"]  # 중고생은 전체의 0.00%라 제외


# ---------- 날씨 구간 ----------
def rain_level(mm: float) -> str:
    return "무강수" if mm <= 0.1 else "약한 비(<1mm)" if mm < 1 else "비(1~5mm)" if mm < 5 else "강한 비(5mm+)"


def temp_band(c: float) -> str:
    return "0℃ 이하" if c <= 0 else "0~10℃" if c <= 10 else "10~20℃" if c <= 20 else "20~28℃" if c <= 28 else "28℃ 초과"


RAIN_ORDER = ["무강수", "약한 비(<1mm)", "비(1~5mm)", "강한 비(5mm+)"]
TEMP_ORDER = ["0℃ 이하", "0~10℃", "10~20℃", "20~28℃", "28℃ 초과"]


# ---------- 기준선 ----------
def _baseline(g: pd.DataFrame, past_only: bool) -> np.ndarray:
    """g: 한 (요일유형, 시간) 묶음, 날짜순. 무강수일 평균을 앞뒤(또는 이전) WIN일 창에서 구한다."""
    d = g["d"].to_numpy()
    u = g["use"].to_numpy(dtype=float)
    dry = g["dry"].to_numpy()
    out = np.full(len(g), np.nan)
    w = np.timedelta64(WIN, "D")
    for i in range(len(g)):
        hi = d[i] if past_only else d[i] + w
        m = (d >= d[i] - w) & (d < hi) & dry
        m[i] = False
        if m.sum() >= 2:
            out[i] = u[m].mean()
    return out


def make_t(df: pd.DataFrame, past: bool = False) -> pd.DataFrame:
    """(날짜, 시간, 요일유형) 단위 표. use=이용량 합, lb=앞뒤 4주 무강수일 평균(ratio=use/lb).

    past=True면 기준선을 '이전 4주'만으로 만든다(백테스트용: 미래 정보를 쓰지 않는다).
    """
    t = df.groupby(["date", "hour", "day_type"], as_index=False).agg(
        use=("use", "sum"), **{c: (c, "first") for c in WEATHER_COLS})
    t["d"] = pd.to_datetime(t["date"])
    t["dry"] = (t["precip_mm"] == 0) & (t["snowfall_cm"] == 0)
    t = t.sort_values(["day_type", "hour", "d"]).reset_index(drop=True)
    parts = [pd.Series(_baseline(g, past), index=g.index) for _, g in t.groupby(["day_type", "hour"])]
    t["lb"] = pd.concat(parts).sort_index()
    t["ratio"] = t["use"] / t["lb"]
    t["rain_lv"] = t["precip_mm"].map(rain_level)
    t["temp_band"] = t["temp_c"].map(temp_band)
    t["snow"] = t["snowfall_cm"] > 0
    return t


@lru_cache(maxsize=None)
def t_for(line: str = "전체", dtype: str = "전체") -> pd.DataFrame:
    sub = DF if line == "전체" else DF[DF["line"] == line]
    if dtype != "전체":
        sub = sub[sub["day_type"] == dtype]
    return make_t(sub)


# ---------- 날씨 보정 계수 ----------
def compute_factors(t: pd.DataFrame):
    ok = t.dropna(subset=["ratio"])
    rain = ok[~ok["snow"]].groupby("rain_lv")["ratio"].mean().to_dict()          # 눈 아닌 시간: 강수 구간별
    temp = ok[ok["dry"]].groupby("temp_band")["ratio"].mean().to_dict()           # 무강수 시간: 기온 구간별
    snow = float(ok[ok["snow"]]["ratio"].mean())
    return rain, temp, snow


T = t_for("전체", "전체")
RAIN_F, TEMP_F, SNOW_F = compute_factors(T)


def weather_factor(temp_c, precip_mm, snow_cm) -> float:
    """강수·눈 보정 × 기온 보정 (눈이 오면 눈 보정이 강수 보정을 대신한다)."""
    precip_f = SNOW_F if snow_cm > 0 else RAIN_F.get(rain_level(precip_mm), 1.0)
    return precip_f * TEMP_F.get(temp_band(temp_c), 1.0)


def group_stats(t: pd.DataFrame, col: str, order: list, mask=None) -> pd.DataFrame:
    """구간별 평균 ratio, 95% 신뢰구간 반폭, 표본(시간 수)."""
    ok = t.dropna(subset=["ratio"])
    if mask is not None:
        ok = ok[mask(ok)]
    g = ok.groupby(col)["ratio"].agg(["mean", "std", "count"]).reindex(order)
    g["ci"] = 1.96 * g["std"] / np.sqrt(g["count"])
    return g


# ---------- 기준 이용량 (예측용) ----------
@lru_cache(maxsize=None)
def usual_by_hour(line: str, dtype: str, anchor: str) -> pd.Series:
    """2025년에서 anchor('MM-DD')와 앞뒤 4주 안에 있는 같은 요일유형 무강수일의 시간대별 평균."""
    try:
        center = pd.Timestamp(f"2025-{anchor}")
    except ValueError:  # 2/29
        center = pd.Timestamp("2025-02-28")
    sub = DF if line == "전체" else DF[DF["line"] == line]
    sub = sub[(sub["day_type"] == dtype) & (sub["precip_mm"] == 0) & (sub["snowfall_cm"] == 0)]
    tot = sub.groupby(["date", "hour"])["use"].sum().reset_index()
    gap = (pd.to_datetime(tot["date"]) - center).abs().dt.days
    tot = tot[gap <= WIN]
    if tot["date"].nunique() < 3:  # 창에 표본이 너무 적으면 한 해 전체 평균으로 대체
        tot = sub.groupby(["date", "hour"])["use"].sum().reset_index()
    return tot.groupby("hour")["use"].mean()


# ---------- 2026 공휴일 (예보일 요일유형 판정용) ----------
HOLIDAYS_2026 = {
    "2026-01-01", "2026-02-16", "2026-02-17", "2026-02-18", "2026-03-02", "2026-05-05", "2026-05-25",
    "2026-06-03", "2026-08-17", "2026-09-24", "2026-09-25", "2026-09-26", "2026-10-05", "2026-10-09",
    "2026-12-25",
}


def day_type_of(d) -> str:
    if f"{d:%Y-%m-%d}" in HOLIDAYS_2026:
        return "일·공휴일"
    return "평일" if d.weekday() < 5 else "토요일" if d.weekday() == 5 else "일·공휴일"


# ---------- 백테스트 ----------
CUT = "2025-09-01"


@lru_cache(maxsize=None)
def backtest():
    """앞(1~8월)으로 계수를 만들고 뒤(9~12월)를 맞혀 본다. 세 가지 방법을 비교한다.

    A 연평균: 학습기간 무강수일 평균 / B 최근 4주 평균 / C B × 날씨 보정(학습기간 계수)
    """
    tr_t = make_t(DF[DF["date"] < CUT])
    rain_f, temp_f, snow_f = compute_factors(tr_t)
    base_a = tr_t[tr_t["dry"]].groupby(["day_type", "hour"])["use"].mean().rename("a")

    full = make_t(DF, past=True)
    te = full[full["date"] >= CUT].join(base_a, on=["day_type", "hour"]).dropna(subset=["lb", "a"]).copy()
    te["b"] = te["lb"]
    f = [(snow_f if s else rain_f.get(rain_level(p), 1.0)) * temp_f.get(temp_band(c), 1.0)
         for s, p, c in zip(te["snow"], te["precip_mm"], te["temp_c"])]
    te["c"] = te["b"] * np.array(f)

    def mape(x, y):
        return float((abs(x - y) / x).mean() * 100)

    daily = te.groupby("date")[["use", "a", "b", "c"]].sum()
    wet = te[~te["dry"]]
    metrics = pd.DataFrame({
        "시간 단위": [mape(te["use"], te[k]) for k in "abc"],
        "하루 합계": [mape(daily["use"], daily[k]) for k in "abc"],
        "비·눈 온 시간만": [mape(wet["use"], wet[k]) for k in "abc"],
    }, index=["A 연평균", "B 최근 4주", "C B+날씨"])
    return daily, metrics, len(daily), len(wet)
