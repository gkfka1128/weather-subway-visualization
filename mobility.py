"""이야기 2~3단계: 다른 교통수단과의 대비(버스·따릉이), 전환(역 주변 따릉이), 원인(역 성격), 미세먼지.

모든 비교는 같은 방법을 쓴다: 그날 값 / (같은 요일 앞뒤 4주 '보통날' 중앙값) - 1.
보통날 = 공휴일·징검다리·근로자의 날·비·눈·미세먼지 나쁨이 아닌 날.
"""
from functools import lru_cache

import numpy as np
import pandas as pd

from analysis import _HOUR_WEATHER, DF, ROOT, ST

DATA = ROOT / "data"
HOLIDAYS = pd.read_csv(DATA / "holidays.csv").set_index("date")["name"].to_dict()
RENAMED = {"당고개": "불암산", "자양": "뚝섬유원지"}  # prep/stations.py 와 같은 역 이름 통일
WX_CATS = ["약한 비", "많은 비 (10mm↑)", "눈", "한파 (최저 -10℃↓)", "폭염 (최고 33℃↑)", "미세먼지 나쁨"]
TYPES = ["업무지구", "혼합", "주거지", "여가·상업"]
T95 = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26, 15: 2.14, 20: 2.09, 30: 2.05}


def t95(n: int) -> float:
    """표본 n개의 95% 신뢰구간 t값 (scipy 없이 근사표)."""
    return next((v for k, v in sorted(T95.items()) if n <= k), 1.96)


def same_weekday_base(values: pd.Series, skip: pd.Series, k: int = 4) -> pd.Series:
    """같은 요일 앞뒤 k주(자기 자신·skip인 날 제외, 보통날 3일 이상)의 중앙값. values는 같은 요일끼리 날짜순."""
    v = values.where(~skip)

    def med(k):
        nb = pd.concat([v.shift(i) for i in range(-k, k + 1) if i], axis=1)
        return nb.median(axis=1).where(nb.notna().sum(axis=1) >= 3)
    return med(k).fillna(med(2 * k))  # 연말·연휴가 몰린 때는 보통날이 모자라 앞뒤 8주로 넓힌다


@lru_cache(maxsize=1)
def days() -> pd.DataFrame:
    """날짜별 날씨·미세먼지·달력 구분. index=날짜(Timestamp)."""
    w = _HOUR_WEATHER.groupby("date").agg(precip=("precip_mm", "sum"), snow=("snowfall_cm", "sum"),
                                         tmax=("temp_c", "max"), tmin=("temp_c", "min"))
    air = pd.read_csv(DATA / "air_daily.csv").set_index("date")
    g = w.join(air)
    g.index = pd.to_datetime(g.index)
    g["wd"] = g.index.weekday
    g["hol"] = g.index.strftime("%Y-%m-%d").map(HOLIDAYS)
    g["labor"] = g.index.strftime("%m-%d") == "05-01"
    off = g["hol"].notna() | (g["wd"] >= 5) | g["labor"]
    prev_off = off.reindex(g.index - pd.Timedelta(days=1), fill_value=False).to_numpy()
    next_off = off.reindex(g.index + pd.Timedelta(days=1), fill_value=False).to_numpy()
    g["bridge"] = (g["wd"] < 5) & ~off & prev_off & next_off
    g["plain"] = g["hol"].isna() & ~g["bridge"] & ~g["labor"]            # 달력상 보통날
    g["pm_bad"] = (g["pm10"] >= 81) | (g["pm2_5"] >= 36)                   # 환경부 예보 '나쁨' 기준
    g["wx"] = np.select(
        [g["snow"] > 0, g["precip"] >= 10, g["precip"] > 0.1,
         g["tmin"] <= -10, g["tmax"] >= 33, g["pm_bad"]],
        ["눈", "많은 비 (10mm↑)", "약한 비", "한파 (최저 -10℃↓)", "폭염 (최고 33℃↑)", "미세먼지 나쁨"], "맑은 보통날")
    g["usual"] = g["plain"] & (g["wx"] == "맑은 보통날")
    g["weekend"] = (g["wd"] >= 5) | g["hol"].notna()
    return g


def effect(series: pd.Series) -> pd.Series:
    """날짜별 값 → 같은 요일 보통날 대비 변화(%). series.index는 'YYYY-MM-DD' 문자열 또는 Timestamp."""
    s = series.copy()
    s.index = pd.to_datetime(s.index)
    g = days().reindex(s.index)
    df = pd.DataFrame({"v": s, "wd": g["wd"], "skip": ~g["usual"].fillna(False).astype(bool)}).sort_index()
    base = pd.concat([same_weekday_base(x["v"], x["skip"]) for _, x in df.groupby("wd")])
    return (df["v"] / base.reindex(df.index) - 1) * 100


def mean_ci(e: pd.Series):
    e = e.dropna()
    n = len(e)
    ci = t95(n) * e.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return e.mean(), ci, n


# ---------- 교통수단별 하루 합계 ----------
@lru_cache(maxsize=1)
def modes() -> pd.DataFrame:
    """date × [지하철, 버스, 따릉이, 따릉이 정기권, 따릉이 일일권] 하루 합계. 세 수단이 모두 있는 날만."""
    sub = DF.groupby("date")["use"].sum()
    bus = pd.read_csv(DATA / "bus_daily.csv").set_index("date")
    bike = pd.read_csv(DATA / "bike_daily.csv").set_index("date")
    m = pd.DataFrame({"지하철": sub, "버스": bus["boardings"] + bus["alightings"],
                      "따릉이": bike["total"], "따릉이 정기권": bike["commute"], "따릉이 일일권": bike["casual"]})
    return m.dropna()


@lru_cache(maxsize=1)
def mode_effects() -> pd.DataFrame:
    """date × 수단별 보통날 대비 %."""
    return pd.DataFrame({c: effect(s) for c, s in modes().items()})


def mode_table(cols=("지하철", "버스", "따릉이"), weekday_only=True) -> pd.DataFrame:
    """날씨 구분 × 수단: 평균 %, 95% CI, 일수. 달력상 보통날만(공휴일 등 제외)."""
    e = mode_effects()
    g = days().reindex(e.index)
    keep = g["plain"] & (~g["weekend"] if weekday_only else True)
    rows = []
    for cat in WX_CATS:
        for c in cols:
            m, ci, n = mean_ci(e.loc[keep & (g["wx"] == cat), c])
            rows.append((cat, c, m, ci, n))
    return pd.DataFrame(rows, columns=["wx", "mode", "m", "ci", "n"])


# ---------- 역 성격 ----------
@lru_cache(maxsize=1)
def station_info() -> pd.DataFrame:
    return pd.read_csv(DATA / "station_info.csv", dtype={"bike_ids": str}).fillna({"bike_ids": ""})


@lru_cache(maxsize=1)
def station_daily() -> pd.DataFrame:
    """date × 역(이름 기준, 여러 호선 합) 하루 이용량."""
    # prep/stations.py 의 base_name 과 같은 규칙: 괄호 부기명 제거, 끝의 '역' 제거(서울역 → 서울), 옛 이름 통일
    name = ST["station"].str.replace(r"\(.*?\)", "", regex=True).str.strip()
    name = name.str.replace(r"(?<=..)역$", "", regex=True).replace(RENAMED)
    sd = ST.assign(station=name).groupby(["date", "station"])["use"].sum().unstack()
    # 기간 중간에 새로 생기거나 거의 기록이 없는 역은 합계 수준을 바꾸므로 뺀다 (전체 날짜의 98% 이상 있는 역만)
    return sd.loc[:, sd.notna().mean() >= .98]


@lru_cache(maxsize=1)
def type_effects() -> pd.DataFrame:
    """date × 역 유형별 보통날 대비 % (유형에 속한 역들의 하루 합으로 계산)."""
    sd = station_daily()
    info = station_info().set_index("station")
    out = {}
    for t in TYPES:
        cols = [s for s in info.index[info["type"] == t] if s in sd.columns]
        out[t] = effect(sd[cols].sum(axis=1))
    return pd.DataFrame(out)


def type_table(cat: str = "rain") -> pd.DataFrame:
    """역 유형 × (평일, 주말): 비 오는 날(약한 비+많은 비) 평균 %, CI, 일수."""
    e = type_effects()
    g = days().reindex(e.index)
    wet = g["wx"].isin(WX_CATS[:2])
    rows = []
    for t in TYPES:
        for lab, m in (("평일", g["plain"] & ~g["weekend"]), ("주말", g["plain"] & (g["wd"] >= 5))):
            mm, ci, n = mean_ci(e.loc[wet & m, t])
            rows.append((t, lab, mm, ci, n))
    return pd.DataFrame(rows, columns=["type", "day", "m", "ci", "n"])


@lru_cache(maxsize=1)
def station_rain() -> pd.DataFrame:
    """역별: 평일 비 오는 날 지하철 변화 %, 역 주변 따릉이 변화 %, 따릉이 의존도(주변 대여 / 지하철 이용, 맑은 평일 평균)."""
    sd = station_daily()
    near = pd.read_csv(DATA / "bike_near_station.csv").pivot(index="date", columns="station", values="rentals")
    near = near.reindex(sorted(set(near.index) | set(sd.index))).fillna(0)  # 행이 없는 날 = 대여 0건
    g = days()
    info = station_info().set_index("station")
    rows = []
    for st in info.index:
        if st not in sd.columns or st not in near.columns:
            continue
        sub, bk = sd[st].dropna(), near[st].dropna()
        sub.index, bk.index = pd.to_datetime(sub.index), pd.to_datetime(bk.index)
        if sub.mean() < 3000 or bk.mean() < 20:  # 너무 작은 역·대여소는 비율이 쉽게 튄다
            continue
        es, eb = effect(sub), effect(bk)
        gg = g.reindex(es.index)
        wet = gg["plain"] & ~gg["weekend"] & gg["wx"].isin(WX_CATS[:2])
        dry = gg["usual"] & ~gg["weekend"]
        share = bk.reindex(es.index)[dry].mean() / sub.reindex(es.index)[dry].mean() * 100
        rows.append((st, info.at[st, "type"], info.at[st, "lat"], info.at[st, "lon"], sub.mean(),
                     es[wet].mean(), eb.reindex(es.index)[wet].mean(), share))
    return pd.DataFrame(rows, columns=["station", "type", "lat", "lon", "use", "sub_rain", "bike_rain", "bike_share"])


@lru_cache(maxsize=1)
def lost_trips() -> pd.DataFrame:
    """날씨 구분 × 수단: 비·눈 오는 평일 하루에 '보통날이었다면' 대비 늘거나 준 이동 (만 건).
    지하철·버스 이용량은 승차+하차라 2로 나눠 승차(이동) 기준으로, 따릉이는 대여 건수 그대로."""
    v = modes().copy()
    v.index = pd.to_datetime(v.index)
    e = mode_effects()
    g = days().reindex(e.index)
    out = {}
    for cat in WX_CATS[:3]:
        k = g["plain"] & (g["wd"] < 5) & (g["wx"] == cat)
        diff = (v[k] - v[k] / (1 + e[k] / 100)).mean() / 1e4
        diff[["지하철", "버스"]] /= 2
        out[cat] = diff
    return pd.DataFrame(out).T
