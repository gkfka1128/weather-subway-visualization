"""대시보드 그래프 23개 + 오늘 예측 그래프. 설명 문장의 숫자는 그때그때 데이터에서 계산한다."""
import base64
import io
import re
import threading
from functools import lru_cache

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpld3
import numpy as np
import pandas as pd
import seaborn as sns
from markupsafe import Markup, escape

from analysis import (_HOUR_WEATHER, CUT, DF, LINES, ROOT, PT, PTYPES, RAIN_ORDER, ST, TEMP_ORDER, WEEKDAYS, backtest, group_stats,
                      make_t, t_for)

import mobility as mob
from matplotlib import font_manager as _fm
from matplotlib.colors import to_rgba
from matplotlib.ticker import FuncFormatter
import matplotlib.patheffects as pe

# Windows는 맑은 고딕, Docker(리눅스)는 나눔고딕 — 실제로 설치된 첫 한글 폰트를 쓴다.
_installed = {f.name for f in _fm.fontManager.ttflist}
KFONT = next((f for f in ["Malgun Gothic", "NanumGothic", "AppleGothic"] if f in _installed), "DejaVu Sans")
sns.set_theme(style="whitegrid", font=KFONT)  # set_theme이 font.family를 덮어쓰므로 rcParams는 그 뒤에
plt.rcParams.update({"axes.unicode_minus": False, "grid.color": "#ebe8e1", "axes.edgecolor": "#d8d4ca",
                     "text.color": "#1b1b1a", "axes.labelcolor": "#55534e",
                     "xtick.color": "#55534e", "ytick.color": "#55534e"})

_lock = threading.Lock()  # pyplot은 전역 상태를 쓰므로 직렬화
# 화면(style.css)과 맞춘 차분한 색: 비=남색, 눈=주황, 평소=회색
C_BLUE, C_ORANGE, C_GRAY, C_RED, C_GREEN = "#2f5d8a", "#c8742b", "#a9a59b", "#b8432f", "#3d7a4f"
# 서울 지하철 노선 고유색 (style.css의 --l1~--l8과 같은 값)
LINE_COLORS = {"1호선": "#0052a4", "2호선": "#00a84d", "3호선": "#ef7c1c", "4호선": "#00a5de",
               "5호선": "#996cac", "6호선": "#cd7c2f", "7호선": "#747f00", "8호선": "#e6186c"}
WEATHER_LABELS = {"temp_c": "평균 기온", "precip_mm": "강수량 합", "snowfall_cm": "적설 합", "humidity": "평균 습도",
                  "wind_ms": "평균 풍속"}
COMMUTE = [7, 8, 9, 17, 18, 19]


# ---------- 공통 ----------
def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    return f'<img class="png" alt="" src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode("ascii")}">'


def _out(fig, kind: str) -> str:
    html = mpld3.fig_to_html(fig) if kind == "html" else _png(fig)
    plt.close(fig)
    return html


def _hour_x(h):
    """0시(자정 이후)는 24로 취급해 그래프 맨 오른쪽에 둔다."""
    return np.where(np.asarray(h) < 1, 24, np.asarray(h))


def _pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def _effect(t: pd.DataFrame, mask) -> float:
    """mask에 해당하는 시간들의 (실제 합 / 기준선 합) - 1. 표본이 없으면 NaN."""
    ok = t.dropna(subset=["lb"])
    ok = ok[mask(ok)]
    return float(ok["use"].sum() / ok["lb"].sum() - 1) if len(ok) else float("nan")


def _daily(t: pd.DataFrame) -> pd.DataFrame:
    ok = t.dropna(subset=["lb"])
    g = ok.groupby("date")
    d = pd.DataFrame({"use": g["use"].sum(), "lb": g["lb"].sum(), "day_type": g["day_type"].first(),
                      "temp_c": g["temp_c"].mean(), "precip_mm": g["precip_mm"].sum(),
                      "snowfall_cm": g["snowfall_cm"].sum(), "humidity": g["humidity"].mean(),
                      "wind_ms": g["wind_ms"].mean()})
    d["ratio"] = d["use"] / d["lb"]                                         # 계절 보정 후
    d["raw"] = d["use"] / d.groupby("day_type")["use"].transform("mean")   # 요일 유형만 보정
    return d


def _sub(line, dtype):
    sub = DF if line == "전체" else DF[DF["line"] == line]
    return sub if dtype == "전체" else sub[sub["day_type"] == dtype]


def _desc(key: str, *paras) -> dict:
    """그래프 설명: key=핵심 메시지 한 줄, paras=(라벨, 문장). None은 건너뛴다. 문장 속 **굵게** 표시를 쓸 수 있다."""
    return {"key": key, "paras": [p for p in paras if p and p[1]]}


def _scope(line, dtype) -> str:
    return " · ".join(x for x in (None if line == "전체" else line, None if dtype == "전체" else dtype) if x) or "전체"


# ---------- 그래프 ----------
def forecast_html(out) -> str:
    with _lock:
        out = out.assign(x=_hour_x(out["hour"])).sort_values("x")
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.plot(out["x"], out["base"] / 1000, color=C_GRAY, ls="--", label="평소 (같은 계절 무강수일 평균)")
        ax.plot(out["x"], out["pred"] / 1000, color="#0f6b5c", lw=2.2, label="날씨 반영 예측")
        ax.fill_between(out["x"], out["pred"] / 1000, color="#0f6b5c", alpha=.15)
        ax.set(title="오늘의 시간대별 이용량 예측", xlabel="시간 (24=자정 이후)", ylabel="이용량 (천 명)")
        ax.legend()
        return _out(fig, "html")


def c_heat(line, dtype):
    sub = _sub(line, dtype)
    pv = (sub.groupby(["date", "weekday", "hour"])["use"].sum().groupby(["weekday", "hour"]).mean()
          .unstack("hour"))
    pv = pv[[c for c in pv.columns if c != 0] + ([0] if 0 in pv.columns else [])]
    pv.index = [WEEKDAYS[i] for i in pv.index]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    sns.heatmap(pv / 1000, cmap="YlOrRd", ax=ax, cbar_kws={"label": "시간당 평균 이용량 (천 명)"})
    ax.set(title=f"요일×시간대 이용량 ({_scope(line, dtype)})", xlabel="시간", ylabel="요일")
    ax.tick_params(axis="y", rotation=0)
    flat = pv.stack()
    (dy, dh), top = flat.idxmax(), flat.max()
    wk = pv.loc[["월", "화", "수", "목", "금"]].mean().max() if set("월화수목금") <= set(pv.index) else np.nan
    desc = _desc(f"가장 붐비는 때는 **{dy}요일 {dh}시**(시간당 평균 {top / 1000:,.0f}천 명)입니다.",
                 ("읽는 법", "색이 진할수록 이용량이 많습니다. 평일은 **출근(7~9시)·퇴근(17~19시) 두 봉우리**, 주말은 낮 시간대의 낮고 넓은 봉우리입니다."),
                 ("왜 보나", "날씨를 보기 전에, 이용량이 원래 시간·요일에 따라 크게 달라진다는 점을 먼저 확인합니다."))
    return _out(fig, "png"), desc


def c_lines(line, dtype):
    sub = DF if dtype == "전체" else DF[DF["day_type"] == dtype]
    n = sub["date"].nunique()
    s = (sub.groupby("line")["use"].sum() / n / 10000).reindex(LINES)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(range(len(s)), s.to_numpy(),
           color=[to_rgba(LINE_COLORS[l], 1 if line in ("전체", l) else .3) for l in s.index])
    ax.set_xticks(range(len(s)))
    ax.set_xticklabels(list(s.index))
    ax.set(title=f"호선별 하루 평균 이용량 ({dtype})", xlabel="호선", ylabel="하루 평균 이용량 (만 명)")
    top, low = s.idxmax(), s.idxmin()
    desc = _desc(f"**{top}**이 하루 평균 {s.max():,.0f}만 명으로 가장 많고, {low}은 {s.min():,.0f}만 명으로 가장 적습니다.",
                 ("읽는 법", "1일 이용량 = 승차+하차. 막대 색은 호선 고유색이고, 호선을 고르면 그 호선만 진하게 표시됩니다."),
                 ("왜 보나", "호선별 규모 차이가 커서, 날씨 영향은 사람 수가 아니라 **비율(%)**로 비교해야 공정합니다."))
    return _out(fig, "html"), desc


def c_station(line, dtype):
    sub = ST if line == "전체" else ST[ST["line"] == line]
    sub = sub if dtype == "전체" else sub[sub["day_type"] == dtype]
    n = sub["date"].nunique()
    s = (sub.groupby(["station", "line"])["use"].sum() / n / 10000).nlargest(15).iloc[::-1]
    # 역 이름 뒤 괄호 부기명(예: 잠실(송파구청))은 빼고, 호선은 막대 색으로도 보여 준다
    names = [f"{a.split('(')[0]} · {b}" for a, b in s.index]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.barh(range(len(s)), s.to_numpy(), color=[LINE_COLORS.get(b, C_BLUE) for _, b in s.index], height=.7)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set(title=f"이용량 많은 역 15곳 ({_scope(line, dtype)})", xlabel="하루 평균 이용량 (만 명)")
    # mpld3는 bbox_inches="tight"가 없어 긴 라벨이 잘린다 → 가장 긴 라벨 길이만큼 왼쪽 여백을 직접 잡는다
    fig.subplots_adjust(left=min(.45, .06 + .024 * max(len(x) for x in names)), right=.97, top=.92, bottom=.11)
    a, b = s.index[-1]
    desc = _desc(f"**{a.split('(')[0]}역({b})**이 하루 평균 {s.iloc[-1]:,.1f}만 명으로 1위입니다.",
                 ("읽는 법", "막대 색은 호선 고유색입니다. 역 이름 뒤 괄호 부기명은 줄였습니다."),
                 ("참고", "역별 자료는 시간대 구분이 없는 일 단위라서, 예측은 호선 단위로 하고 역은 규모 확인용으로 씁니다."))
    return _out(fig, "html"), desc


def c_ptype_share():
    wk = PT[PT["day_type"] == "평일"]
    pv = wk.groupby(["hour", "passenger_type"])["use"].sum().unstack("passenger_type")[PTYPES]
    pv = pv.div(pv.sum(axis=1), axis=0) * 100
    pv.index = _hour_x(pv.index)
    pv = pv.sort_index()
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.stackplot(pv.index, pv.T.to_numpy(), labels=PTYPES, alpha=.9,
                 colors=sns.color_palette("Set2", len(PTYPES)))
    ax.set(title="평일 시간대별 승객 유형 구성 (전 호선, 2025년)", xlabel="시간 (24=자정 이후)", ylabel="비율 (%)", xlim=(5, 24), ylim=(0, 100))
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    mid = pv.loc[11:15, "우대권"].mean()
    rush = pv.loc[[8, 18], "우대권"].mean()
    desc = _desc(f"'우대권'(어르신·장애인 등) 비중은 출퇴근 피크(8·18시)에 **{rush:.0f}%**, 낮 시간대(11~15시)에는 **{mid:.0f}%**까지 올라갑니다.",
                 ("읽는 법", "시간대별 승객 유형 비중입니다. 대부분은 '일반'이고, 중고생 유형은 0.00%라 뺐습니다."),
                 ("왜 보나", "시간대마다 승객 구성이 달라서 날씨 반응도 유형마다 다를 수 있습니다(아래 '승객 유형별 반응')."),
                 ("주의", "호선·요일 선택과 무관하게 평일 전체 기준이며, 유형 구분이 있는 2025년 원본만 씁니다."))
    return _out(fig, "png"), desc


def c_corr(line, dtype):
    d = _daily(t_for(line, dtype))
    cols = list(WEATHER_LABELS)
    corr = pd.DataFrame({
        "요일 유형만 보정": [d["raw"].corr(d[c]) for c in cols],
        "계절·연휴까지 보정": [d["ratio"].corr(d[c]) for c in cols]}, index=[WEATHER_LABELS[c] for c in cols])
    fig, ax = plt.subplots(figsize=(6, 3.8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax, cbar_kws={"label": "상관계수"})
    ax.set(title=f"날씨 변수와 하루 이용량의 상관계수 ({_scope(line, dtype)})")
    ax.tick_params(axis="x", rotation=0)
    adj = corr["계절·연휴까지 보정"]
    k = adj.abs().idxmax()
    desc = _desc(f"보정 후 상관계수는 모두 **|r| ≤ {adj.abs().max():.2f}**로, 날씨와 이용량의 직선 관계는 약합니다.",
                 ("읽는 법", f"하루 단위 상관계수(−1~+1)입니다. 왼쪽은 요일 유형만 보정, 오른쪽은 계절·연휴까지 걷어낸 값이며, 보정 후 가장 큰 변수는 '{k}'(r={adj[k]:+.2f})입니다."),
                 ("해석", "왼쪽과 오른쪽 차이가 클수록, 그만큼은 **날씨가 아니라 계절 때문**이었다는 뜻입니다."))
    return _out(fig, "png"), desc


def c_scatter(line, dtype):
    d = _daily(t_for(line, dtype))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, col, ttl, ylab in ((axes[0], "raw", "보정 전 (요일 유형만 보정)", "하루 이용량 / 요일유형 평균"),
                               (axes[1], "ratio", "보정 후 (계절·연휴 제거)", "실제 / 기준선")):
        ax.scatter(d["temp_c"], d[col], s=12, alpha=.45, color=C_GRAY)
        bins = pd.cut(d["temp_c"], bins=np.arange(-15, 40, 5))
        m = d.groupby(bins, observed=True).agg(x=("temp_c", "mean"), y=(col, "mean"), n=(col, "size"))
        m = m[m["n"] >= 3]
        ax.plot(m["x"], m["y"], marker="o", color=C_RED, label="5℃ 구간 평균")
        ax.set(title=ttl, xlabel="하루 평균 기온 (℃)", ylabel=ylab)
        ax.legend(fontsize=8)
    raw_rng = d.groupby(pd.cut(d["temp_c"], np.arange(-15, 40, 5)), observed=True)["raw"].mean()
    adj_rng = d.groupby(pd.cut(d["temp_c"], np.arange(-15, 40, 5)), observed=True)["ratio"].mean()
    desc = _desc(f"기온에 따른 차이가 보정 전 **{(raw_rng.max() - raw_rng.min()) * 100:.0f}%p**에서 보정 후 **{(adj_rng.max() - adj_rng.min()) * 100:.0f}%p**로 줄어듭니다.",
                 ("읽는 법", f"점 하나가 하루, 빨간 선은 5℃ 구간 평균입니다. 구간 평균은 보정 전 {raw_rng.min():.2f}~{raw_rng.max():.2f}, 보정 후 {adj_rng.min():.2f}~{adj_rng.max():.2f}입니다."),
                 ("해석", "'추운 날 이용량이 적다'는 현상의 상당 부분은 기온 자체가 아니라 **계절·연휴 요인**이었습니다."))
    fig.tight_layout()
    return _out(fig, "png"), desc


def c_rain_commute(line, dtype):
    # 평일 출퇴근 전용 그래프: 주말 유형을 골라도 비우지 않고 그 호선의 평일 자료를 보여 준다.
    weekend = dtype not in ("전체", "평일")
    dtype = "평일"
    sub = _sub(line, dtype)
    t = sub.groupby(["date", "hour", "day_type"], as_index=False).agg(
        use=("use", "sum"), precip_mm=("precip_mm", "first"), snowfall_cm=("snowfall_cm", "first"))
    t = t[(t["day_type"] == "평일") & t["hour"].isin(COMMUTE)].copy()
    t["날씨"] = ["눈" if s > 0 else "비" if p > 0.1 else "맑음" for p, s in zip(t.precip_mm, t.snowfall_cm)]
    g = t.groupby(["hour", "날씨"])["use"].mean().unstack("날씨") / 1000
    fig, ax = plt.subplots(figsize=(7, 3.8))
    g.plot(kind="bar", ax=ax, color={"맑음": C_GRAY, "비": C_BLUE, "눈": C_ORANGE})
    ax.set(title=f"평일 출퇴근 시간대: 맑은 날 vs 비·눈 오는 날 ({_scope(line, dtype)})", xlabel="시간",
           ylabel="시간당 평균 이용량 (천 명)")
    ax.tick_params(axis="x", rotation=0)
    cnt = t.groupby("날씨")["date"].nunique()
    if "비" in g and "맑음" in g:
        diff = g["비"].sum() / g["맑음"].sum() - 1
        key = f"출퇴근 시간대에 비 오는 날은 맑은 날보다 **{diff * 100:+.1f}%**입니다. 출퇴근은 날씨가 나빠도 가야 해서 차이가 작습니다."
    else:
        key = "출퇴근은 날씨가 나빠도 가야 하는 이동이라 차이가 작습니다."
    desc = _desc(key,
                 ("참고", "출퇴근 비교라 요일 유형 선택과 관계없이 평일 자료만 씁니다.") if weekend else None,
                 ("주의", f"출퇴근 6개 시간대(7~9시, 17~19시)의 단순 평균 비교라 계절이 섞여 있고, 비·눈 온 날이 적습니다(비 {cnt.get('비', 0)}일, 눈 {cnt.get('눈', 0)}일). "
                         "정밀한 비교는 아래 **'날씨 구간별 영향'**을 보세요."))
    return _out(fig, "html"), desc


def c_factors(line, dtype):
    t = t_for(line, dtype)
    rain = group_stats(t, "rain_lv", RAIN_ORDER, mask=lambda x: ~x["snow"])
    snow = t.dropna(subset=["ratio"])
    snow = snow[snow["snow"]]["ratio"]
    temp = group_stats(t, "temp_band", TEMP_ORDER, mask=lambda x: x["dry"])
    labels1 = RAIN_ORDER + ["눈"]
    mean1 = list(rain["mean"]) + [snow.mean() if len(snow) else np.nan]
    ci1 = list(rain["ci"]) + [1.96 * snow.std() / np.sqrt(len(snow)) if len(snow) > 1 else np.nan]
    n1 = list(rain["count"]) + [len(snow)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, labels, mean, ci, n, ttl in ((axes[0], labels1, mean1, ci1, n1, "강수·적설 구간 (무강수=기준)"),
                                         (axes[1], TEMP_ORDER, list(temp["mean"]), list(temp["ci"]),
                                          list(temp["count"]), "기온 구간 (무강수 시간만)")):
        y = (np.array(mean, dtype=float) - 1) * 100
        e = np.array(ci, dtype=float) * 100
        ax.bar(range(len(labels)), np.nan_to_num(y), yerr=np.nan_to_num(e), capsize=4,
               color=[C_RED if v < 0 else C_GREEN for v in np.nan_to_num(y)], alpha=.85)
        ax.axhline(0, color="black", lw=.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([f"{l}\n(n={int(c)})" for l, c in zip(labels, n)], fontsize=8)
        ax.set_title(ttl)
    axes[0].set_ylabel("평소 대비 이용량 변화 (%)")
    fig.suptitle(f"날씨 구간별 이용량 변화 — 막대=평균, 선=95% 신뢰구간 ({_scope(line, dtype)})")
    fig.tight_layout()
    worst = min(zip(labels1, mean1), key=lambda p: p[1] if p[1] == p[1] else 9)
    wide = [l for l, m, c in zip(labels1, mean1, ci1) if l != "무강수" and m == m and c == c and abs(m - 1) < c]
    desc = _desc(f"가장 크게 줄어드는 구간은 **'{worst[0]}' {(worst[1] - 1) * 100:+.1f}%**이고, 기온 구간은 대부분 ±2% 안쪽입니다.",
                 ("읽는 법", "각 시간대 실제 이용량 ÷ '같은 계절·요일 유형·시간대의 무강수일 평균'의 평균입니다. 이 비율이 **오늘 예측에 곱해지는 보정 계수**입니다."),
                 ("주의", f"신뢰구간이 0을 포함하는 구간({', '.join(wide)})은 '차이 없음'과 구분되지 않습니다.") if wide else None)
    return _out(fig, "png"), desc


@lru_cache(maxsize=None)
def _ptype_t(p):
    return make_t(PT[PT["passenger_type"] == p])


def c_ptype_rain():
    rows = []
    for p in PTYPES:
        t = _ptype_t(p)
        rows.append((p, _effect(t, lambda x: (~x["snow"]) & (x["precip_mm"] > 0.1)) * 100,
                     _effect(t, lambda x: x["snow"]) * 100))
    df = pd.DataFrame(rows, columns=["유형", "비", "눈"]).set_index("유형")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    df.plot(kind="bar", ax=ax, color={"비": C_BLUE, "눈": C_ORANGE})
    ax.plot([-.5, len(df) - .5], [0, 0], color="black", lw=.8)
    ax.set(title="승객 유형별 비·눈 오는 날 이용량 변화 (전 호선, 2025년)", xlabel="", ylabel="평소 대비 변화 (%)")
    ax.tick_params(axis="x", rotation=0)
    k = df["비"].idxmin()
    j = df["비"].idxmax()
    desc = _desc(f"비 오는 날 **'{k}' {df.loc[k, '비']:+.1f}%**로 가장 크게 줄고, '{j}'는 {df.loc[j, '비']:+.1f}%로 가장 덜 줄어듭니다.",
                 ("해석", "우대권·어린이처럼 이용을 미룰 수 있는 승객일수록 날씨에 민감해 보이지만, 이 자료만으로 원인까지 확정할 수는 없습니다."),
                 ("주의", "어린이·직원은 전체의 1% 안팎이라 변동이 큽니다. 유형 구분이 있는 2025년 1년치로 계산했습니다."))
    return _out(fig, "html"), desc


def c_line_rain():
    rows = []
    for ln in LINES:
        t = t_for(ln, "전체")
        rows.append((ln, _effect(t, lambda x: (~x["snow"]) & (x["precip_mm"] > 0.1)) * 100,
                     _effect(t, lambda x: x["snow"]) * 100))
    df = pd.DataFrame(rows, columns=["호선", "비", "눈"]).set_index("호선")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    df.plot(kind="bar", ax=ax, color={"비": C_BLUE, "눈": C_ORANGE})
    ax.plot([-.5, len(df) - .5], [0, 0], color="black", lw=.8)
    ax.set(title="호선별 비·눈 오는 날 이용량 변화", xlabel="", ylabel="평소 대비 변화 (%)")
    ax.tick_params(axis="x", rotation=0)
    k, j = df["비"].idxmin(), df["비"].idxmax()
    desc = _desc(f"비 오는 날 **{k} {df.loc[k, '비']:+.1f}%**로 가장 크게, {j}은 {df.loc[j, '비']:+.1f}%로 가장 적게 줄었습니다.",
                 ("해석", "호선마다 이용 목적(출퇴근·여가 등)이 달라 반응이 다를 수 있으나, 원인까지 확정하긴 어렵습니다."),
                 ("참고", "호선 선택과 무관하게 전체 호선을 비교합니다."))
    return _out(fig, "html"), desc


def c_backtest():
    daily, metrics, ndays, nwet = backtest()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    x = pd.to_datetime(daily.index)
    ax.plot(x, daily["use"] / 1e6, color="black", lw=1.4, label="실제")
    ax.plot(x, daily["a"] / 1e6, color=C_GRAY, lw=1, label="A 연평균")
    ax.plot(x, daily["b"] / 1e6, color=C_BLUE, lw=1, label="B 최근 4주")
    ax.plot(x, daily["c"] / 1e6, color=C_ORANGE, lw=1, label="C B+날씨")
    ax.set(title=f"하루 이용량: 실제 vs 예측 (검증 {ndays}일)", ylabel="하루 이용량 (백만 명)")
    ax.legend(fontsize=8, ncol=4, loc="lower left")
    ax.tick_params(axis="x", rotation=30)
    m = metrics.T
    m.plot(kind="bar", ax=axes[1], color=[C_GRAY, C_BLUE, C_ORANGE])
    axes[1].set(title="평균 절대 오차율 (낮을수록 좋음)", ylabel="MAPE (%)")
    axes[1].tick_params(axis="x", rotation=0, labelsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    b, c = metrics.loc["B 최근 4주"], metrics.loc["C B+날씨"]
    desc = _desc(f"날씨 보정(C)을 더해도 하루 오차는 **{b['하루 합계']:.1f}% → {c['하루 합계']:.1f}%**로 거의 그대로입니다.",
                 ("검증 방법", f"{CUT[:4]}년 이전 데이터로 계수를 만들고, 학습에 쓰지 않은 {CUT[:4]}년 1년({ndays}일)을 맞혀 봤습니다. 한 해 전체라 계절이 모두 들어갑니다."),
                 ("결과", f"하루 합계 오차는 연평균(A) {metrics.loc['A 연평균', '하루 합계']:.1f}% → 최근 4주(B) {b['하루 합계']:.1f}% → B+날씨(C) {c['하루 합계']:.1f}%입니다. "
                         f"비·눈 온 {nwet}개 시간대만 보면 B {b['비·눈 온 시간만']:.1f}% → C {c['비·눈 온 시간만']:.1f}%로 약간 개선됩니다. "
                         "A는 이전 해 평균이라 해마다의 수준 변화까지 오차에 들어갑니다."),
                 ("결론", "날씨 효과는 방향은 일관되지만 평소 변동보다 작아 정확도를 크게 올리진 못합니다. "
                         "그래서 서비스는 **'평소 대비 몇 % 차이'**로 안내합니다."))
    return _out(fig, "png"), desc


# ---------- 날씨 개요 (B 담당 그래프) ----------
def _wx_daily() -> pd.DataFrame:
    w = _HOUR_WEATHER
    g = w.groupby("date")
    d = pd.DataFrame({"temp": g["temp_c"].mean(), "tmax": g["temp_c"].max(), "tmin": g["temp_c"].min(),
                      "precip": g["precip_mm"].sum(), "snow": g["snowfall_cm"].sum()})
    d.index = pd.to_datetime(d.index)
    return d


def c_weather_month():
    d = _wx_daily()
    yrs = sorted(d.index.year.unique())
    by_year = d.groupby([d.index.year, d.index.month])["precip"].sum().unstack(0)  # 행=월, 열=해
    m = d.groupby(d.index.month).agg(temp=("temp", "mean"), tmax=("tmax", "mean"), tmin=("tmin", "mean"))
    m["precip"] = by_year.mean(axis=1)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax2 = ax.twinx()
    ax2.bar(m.index, m["precip"], color=C_BLUE, alpha=.28, width=.6, label=f"월 강수량 {len(yrs)}년 평균 (mm)")
    ax2.scatter(np.repeat(m.index, len(yrs)), by_year.to_numpy().ravel(), s=9, color=C_BLUE, alpha=.7,
                label="해마다 월 강수량", zorder=3)
    ax2.set_ylabel("월 강수량 (mm)")
    ax2.grid(False)
    ax.fill_between(m.index, m["tmin"], m["tmax"], color=C_ORANGE, alpha=.15, label="일 최저~최고 평균")
    ax.plot(m.index, m["temp"], color=C_ORANGE, marker="o", lw=2, label="월 평균 기온 (℃)")
    ax.axhline(0, color=C_GRAY, lw=.8)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set(title=f"{yrs[0]}~{yrs[-1]}년 서울 월별 기온과 강수량", xlabel="월", ylabel="기온 (℃)", xticks=range(1, 13))
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    wet_m, cold_m, hot_m = m["precip"].idxmax(), m["temp"].idxmin(), m["temp"].idxmax()
    share = m.loc[[6, 7, 8], "precip"].sum() / m["precip"].sum()
    swing = (by_year.max(axis=1) / by_year.min(axis=1).clip(lower=1)).loc[[6, 7, 8, 9]].max()
    desc = _desc(f"한 해 비의 **{share:.0%}가 6~8월**에 몰려 있어, 그냥 비교하면 '비 효과'에 '여름 효과'가 섞입니다.",
                 ("읽는 법", f"{yrs[0]}~{yrs[-1]}년 {len(yrs)}년 평균입니다. 선은 월 평균 기온, 띠는 하루 최저~최고 평균, 막대는 월 강수량 평균, 점은 해마다의 월 강수량입니다."),
                 ("수치", f"가장 추운 달은 {cold_m}월({m.loc[cold_m, 'temp']:.1f}℃), 가장 더운 달은 {hot_m}월({m.loc[hot_m, 'temp']:.1f}℃), "
                         f"비가 가장 많은 달은 {wet_m}월({m.loc[wet_m, 'precip']:,.0f}mm)입니다. 여름 강수량은 같은 달이라도 해마다 최대 {swing:.0f}배 차이가 납니다."),
                 ("그래서", "이 대시보드는 **같은 시기의 맑은 날**과 비교하는 기준선을 씁니다."))
    return _out(fig, "png"), desc


def c_temp_hist():
    t = _HOUR_WEATHER["temp_c"].dropna()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    sns.histplot(t, bins=np.arange(np.floor(t.min()), t.max() + 1.5, 1.5), kde=True, ax=ax,
                 color=C_ORANGE, edgecolor="white", alpha=.55, line_kws={"lw": 2})
    for x, lab in ((0, "한파 기준 0℃"), (28, "폭염 기준 28℃")):
        ax.axvline(x, color=C_RED if x else C_BLUE, ls="--", lw=1.2)
        ax.text(x, ax.get_ylim()[1] * .95, f" {lab}", color=C_RED if x else C_BLUE, fontsize=8, va="top")
    yrs = f"{_HOUR_WEATHER['date'].min()[:4]}~{_HOUR_WEATHER['date'].max()[:4]}"
    ax.set(title=f"{yrs}년 서울 시간별 기온 분포", xlabel="기온 (℃)", ylabel="시간 수")
    cold, hot = (t <= 0).mean(), (t > 28).mean()
    desc = _desc(f"기온이 **{t.min():.1f}℃ ~ {t.max():.1f}℃**로 넓게 퍼져 있어, 평균 하나보다 구간(한파·보통·폭염)으로 나눠 봐야 합니다.",
                 ("읽는 법", f"지하철 운행 시간대 {len(t):,}개 시간의 기온을 1.5℃ 간격으로 센 히스토그램이고, 곡선은 분포를 부드럽게 그린 KDE입니다."),
                 ("수치", f"평균 {t.mean():.1f}℃이고, 0℃ 이하가 전체 시간의 {cold:.1%}, 28℃ 초과가 {hot:.1%}입니다."))
    return _out(fig, "png"), desc


def _box_verdict(med, iqr, order) -> str:
    """상자그림 설명의 결론 문장: 날씨별 중앙값 차이를 맑은 날 상자 폭과 실제로 비교해서 고른다."""
    base = order[0]
    if base not in med.index or not np.isfinite(iqr.get(base, np.nan)):
        return "맑은 날 표본이 부족해 날마다 흔들림과 비교하기 어렵습니다."
    gap = max((abs(med[k] - med[base]) for k in order[1:] if k in med.index), default=0.0)
    w = iqr[base]
    if gap <= w:
        return (f"날씨별 중앙값 차이(최대 **{gap:.1f}%p**)가 맑은 날끼리의 흔들림(상자 폭 {w:.1f}%p)보다 작아서, "
                "날씨 차이가 평소 날마다 흔들림에 묻힙니다.")
    return (f"날씨별 중앙값 차이(최대 **{gap:.1f}%p**)가 맑은 날 상자 폭({w:.1f}%p)보다 커서, "
            "이 범위에서는 날씨 효과가 날마다 흔들림을 넘어섭니다.")


def c_weather_box(line, dtype):
    d = _daily(t_for(line, dtype))
    order = ["맑음 (강수 0)", "약한 비 (<10mm)", "많은 비 (10mm↑)", "눈"]
    kind = np.select([d["snowfall_cm"] > 0, d["precip_mm"] >= 10, d["precip_mm"] > 0.1],
                     [order[3], order[2], order[1]], order[0])
    d = d.assign(kind=pd.Categorical(kind, order), pct=(d["ratio"] - 1) * 100)
    pal = [C_GRAY, "#7fa3c6", C_BLUE, "#7a5fb5"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    sns.boxplot(data=d, x="kind", y="pct", hue="kind", order=order, palette=pal, legend=False, ax=ax,
                width=.55, fliersize=3, linewidth=1)
    ax.axhline(0, color=C_GRAY, ls="--", lw=1)
    # 명절·연휴 앞뒤처럼 ±30%가 넘는 날 몇 개 때문에 상자가 납작해지지 않게 축을 1~99% 범위로 자른다
    lo, hi = d["pct"].quantile([.01, .99])
    pad = (hi - lo) * .15
    ax.set_ylim(lo - pad, hi + pad)
    cut = int(((d["pct"] < lo - pad) | (d["pct"] > hi + pad)).sum())
    n = d["kind"].value_counts()
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{k}\n{n.get(k, 0)}일" for k in order])
    ax.set(title=f"날씨 유형별 하루 이용량 (평소 대비, {_scope(line, dtype)})", xlabel="", ylabel="평소 대비 (%)")
    med = d.groupby("kind", observed=True)["pct"].median()
    iqr = d.groupby("kind", observed=True)["pct"].quantile(.75) - d.groupby("kind", observed=True)["pct"].quantile(.25)
    short = dict(zip(order, ["맑음", "약한 비", "많은 비", "눈"]))
    parts = ", ".join(f"{short[k]} {med[k]:+.1f}%({n.get(k, 0)}일)" for k in order if k in med.index)
    desc = _desc(_box_verdict(med, iqr, order),
                 ("읽는 법", f"하루 이용량을 같은 시기 맑은 날 기준선과 비교한 값(%)입니다. 상자는 가운데 50%, 가운데 선은 중앙값이며, 중앙값은 {parts}입니다."),
                 ("주의", "눈·많은 비는 날짜 수가 적습니다."
                         + (f" 연휴 앞뒤처럼 극단적인 {cut}일은 세로축 밖이라 생략했습니다." if cut else "")))
    return _out(fig, "png"), desc


# ---------- 다른 요인과 비교 (달력·행사) ----------
# 공휴일 목록·신뢰구간 t값·같은 요일 기준선은 교통수단 비교(mobility.py)와 같은 것을 쓴다
HOLIDAYS = mob.HOLIDAYS
_t95 = mob.t95
_same_weekday_base = mob.same_weekday_base
DOW = "월화수목금토일"


@lru_cache(maxsize=1)
def _day_effects() -> pd.DataFrame:
    """하루 전체 이용량을 '같은 요일 앞뒤 4주 보통날' 중앙값과 비교한 %. 날씨·달력 구분을 붙인다."""
    w = _HOUR_WEATHER.groupby("date").agg(precip=("precip_mm", "sum"), snow=("snowfall_cm", "sum"),
                                         tmax=("temp_c", "max"), tmin=("temp_c", "min"))
    g = DF.groupby("date")["use"].sum().to_frame().join(w)
    g.index = pd.to_datetime(g.index)
    g["wd"] = g.index.weekday
    g["hol"] = g.index.strftime("%Y-%m-%d").map(HOLIDAYS)
    g["labor"] = g.index.strftime("%m-%d") == "05-01"  # 근로자의 날: 공휴일은 아니지만 많은 회사가 쉰다
    # 징검다리: 공휴일 아닌 평일인데 전날과 다음날이 모두 쉬는 날(주말·공휴일·근로자의 날)
    off = g["hol"].notna() | (g["wd"] >= 5) | g["labor"]
    # 빠진 날짜(2024년 4월)가 있어 행 이동 대신 실제 전날·다음날 날짜로 찾는다 (없는 날은 쉬는 날로 보지 않음)
    prev_off = off.reindex(g.index - pd.Timedelta(days=1), fill_value=False).to_numpy()
    next_off = off.reindex(g.index + pd.Timedelta(days=1), fill_value=False).to_numpy()
    g["bridge"] = (g["wd"] < 5) & ~off & prev_off & next_off
    unusual = g["hol"].notna() | g["bridge"] | g["labor"] | (g["precip"] > 0.1) | (g["snow"] > 0)
    base = pd.concat([_same_weekday_base(x["use"], unusual.loc[x.index]) for _, x in g.groupby("wd")])
    g["eff"] = (g["use"] / base.reindex(g.index) - 1) * 100
    plain = g["hol"].isna() & ~g["bridge"] & ~g["labor"]
    g["cat"] = np.select(
        [g["hol"].fillna("").str.contains("설|추석") & (g["wd"] < 5),
         g["hol"].notna() & (g["wd"] < 5), g["labor"], g["bridge"],
         plain & (g["snow"] > 0), plain & (g["precip"] >= 10), plain & (g["precip"] > 0.1),
         plain & (g["tmin"] <= -10), plain & (g["tmax"] >= 33)],
        ["설·추석 연휴", "그 밖의 공휴일", "근로자의 날", "징검다리 평일",
         "눈", "많은 비 (10mm↑)", "약한 비", "한파 (최저 -10℃↓)", "폭염 (최고 33℃↑)"], "")
    return g


def c_factor_size():
    g = _day_effects()
    cal = ["설·추석 연휴", "그 밖의 공휴일", "근로자의 날", "징검다리 평일"]
    wx = ["많은 비 (10mm↑)", "눈", "한파 (최저 -10℃↓)", "폭염 (최고 33℃↑)", "약한 비"]
    rows = []
    for c in cal + wx:
        e = g.loc[g["cat"] == c, "eff"].dropna()
        ci = _t95(len(e)) * e.std(ddof=1) / np.sqrt(len(e)) if len(e) > 1 else 0
        rows.append((c, e.mean(), ci, len(e)))
    r = pd.DataFrame(rows, columns=["cat", "m", "ci", "n"])
    fig, ax = plt.subplots(figsize=(9, 4.2))
    y = np.arange(len(r))[::-1]
    ax.barh(y, r["m"], xerr=r["ci"], color=[C_RED] * len(cal) + [C_BLUE] * len(wx), alpha=.85,
            error_kw={"ecolor": "#55534e", "lw": 1, "capsize": 3}, height=.62)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c} ({n}일{', 참고' if n < 2 else ''})" for c, n in zip(r["cat"], r["n"])])
    for yy, m, ci in zip(y, r["m"], r["ci"]):
        ax.text(m - ci - 1, yy, f"{m:+.1f}%", va="center", ha="right", fontsize=8.5, color="#1b1b1a")
    ax.axvline(0, color="#55534e", lw=.8)
    ax.axhline(len(wx) - .5, color=C_GRAY, ls=":", lw=1)
    ax.set_xlim((r["m"] - r["ci"]).min() - 9, 3)
    ax.set(title="하루 이용량을 바꾸는 요인 — 같은 요일 보통날 대비 (1~8호선 전체)", xlabel="평균 변화 (%, 선=95% 신뢰구간)")
    big = r.set_index("cat")
    wmax = big.loc[wx, "m"].abs().max()
    desc = _desc(f"이용량을 크게 흔드는 것은 날씨(최대 {wmax:.1f}%)가 아니라 **'그날 출근·등교를 하느냐'**입니다.",
                 ("읽는 법", "각 날의 하루 이용량을 같은 요일 앞뒤 4주 보통날(공휴일·징검다리·근로자의 날·비·눈이 아닌 날) 중앙값과 비교했습니다. 빨강은 달력, 파랑은 날씨입니다."),
                 ("수치", f"설·추석 연휴 **{big.loc[cal[0], 'm']:+.0f}%**, 그 밖의 공휴일 **{big.loc[cal[1], 'm']:+.0f}%**, "
                         f"근로자의 날 {big.loc[cal[2], 'm']:+.0f}%, 징검다리 평일 {big.loc[cal[3], 'm']:+.0f}%입니다. 날씨는 가장 큰 것도 {wmax:.1f}% 안팎입니다."),
                 ("해석", "날씨 효과가 작은 것은 실패가 아니라, 지하철 이동 대부분이 날씨와 상관없이 해야 하는 이동이라는 결과입니다."))
    return _out(fig, "png"), desc


@lru_cache(maxsize=1)
def _station_spikes() -> pd.DataFrame:
    s = ST.groupby(["line", "station", "date"], as_index=False)["use"].sum()
    s = s[s.groupby(["line", "station"])["use"].transform("mean") >= 5000]  # 너무 작은 역은 비율이 쉽게 튄다
    s["d"] = pd.to_datetime(s["date"])
    s["wd"] = s["d"].dt.weekday
    s = s.sort_values(["line", "station", "wd", "d"]).reset_index(drop=True)
    skip = s["date"].isin(HOLIDAYS.keys())
    s["base"] = pd.concat([_same_weekday_base(x["use"], skip.loc[x.index])
                           for _, x in s.groupby(["line", "station", "wd"])])
    s["r"] = s["use"] / s["base"]
    return s.dropna(subset=["r"])


def c_station_spikes():
    s = _station_spikes()
    s = s[~s["date"].isin(HOLIDAYS.keys())]  # 공휴일 자체의 나들이 증가는 '행사'와 섞이지 않게 뺀다
    top = s.sort_values("r", ascending=False).drop_duplicates(["line", "station"]).head(10)
    many = s[s["r"] >= 3].groupby(["line", "station"]).size().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    y = np.arange(len(top))[::-1]
    ax.barh(y, top["r"], color=[LINE_COLORS[ln] for ln in top["line"]], height=.62)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{st} ({ln}) {d[5:]}({DOW[w]})" for st, ln, d, w in
                        zip(top["station"], top["line"], top["date"], top["wd"])])
    for yy, v in zip(y, top["r"]):
        ax.text(v + .1, yy, f"{v:.1f}배", va="center", fontsize=8.5)
    ax.axvline(1, color="#55534e", lw=.8)
    ax.set(title="평소보다 가장 붐빈 날 — 역별 1위 (같은 요일 대비)", xlabel="같은 요일 앞뒤 4주 중앙값 대비 (배)")
    ax.set_xlim(0, top["r"].max() * 1.15)
    first = top.iloc[0]
    desc = _desc("노선 전체로는 날씨 효과가 몇 %지만, 역 하나만 보면 **행사 하나가 이용량을 몇 배로** 바꿉니다.",
                 ("읽는 법", "역마다 하루 이용량을 같은 요일 앞뒤 4주 중앙값과 비교해, 가장 크게 튄 날을 하나씩 뽑은 상위 10곳입니다(하루 평균 5천 명 이상 역)."),
                 ("수치", f"1위는 {first['station']}({first['line']}) {first['date']}로 **평소의 {first['r']:.1f}배**입니다."
                         + (f" {many.index[0][1]}역은 평소의 3배 이상인 날이 {many.iloc[0]}일입니다." if len(many) else "")),
                 ("주의", "경기장·한강공원 근처 역이 많아 경기·공연·축제가 원인으로 추정되지만, 날짜별 행사는 따로 확인해야 합니다."))
    return _out(fig, "png"), desc


# ---------- 다른 교통수단·역 성격 (mobility.py) ----------
MODE_COLORS = {"지하철": "#0f6b5c", "버스": C_BLUE, "따릉이": C_ORANGE,
               "따릉이 정기권": "#a35a1c", "따릉이 일일권": "#e0a35f"}
TYPE_COLORS = {"업무지구": "#2f5d8a", "혼합": "#a9a59b", "주거지": "#3d7a4f", "여가·상업": "#c8742b"}


def _trips() -> pd.DataFrame:
    """수단별 하루 이동 건수: 지하철·버스는 승차+하차의 절반(승차 기준), 따릉이는 대여 건수."""
    m = mob.modes()[["지하철", "버스", "따릉이"]].copy()
    m[["지하철", "버스"]] /= 2
    m.index = pd.to_datetime(m.index)
    return m


def c_mode_month():
    m = _trips()
    mon = m.resample("MS").mean()
    idx = mon / mon.mean() * 100
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for c in idx:
        ax.plot(idx.index, idx[c], marker="o", ms=3.5, lw=2, color=MODE_COLORS[c], label=c)
    ax.axhline(100, color=C_GRAY, ls="--", lw=1)
    ax.set(title="교통수단별 월별 하루 이용 (각 수단의 3년 평균 = 100)", ylabel="이용 지수")
    ax.legend(ncol=3, fontsize=9, loc="lower center")
    rng = (idx.max() - idx.min()).round(0)
    avg = m.mean() / 1e4
    bk = idx["따릉이"]
    by_m = bk.groupby(bk.index.month).mean()
    desc = _desc(f"달마다 이용 수준의 차이(최고−최저)가 지하철 {rng['지하철']:.0f}·버스 {rng['버스']:.0f}포인트인데, 따릉이는 **{rng['따릉이']:.0f}포인트**로 약 {rng['따릉이'] / rng[['지하철', '버스']].mean():.0f}배입니다.",
                 ("읽는 법", "달마다 하루 평균 이동 건수를 각 수단의 3년 평균과 비교한 지수입니다(100 = 평균). 규모가 크게 달라 지수로 맞췄습니다. "
                            f"하루 평균은 지하철 약 {avg['지하철']:.0f}만, 버스 {avg['버스']:.0f}만(둘 다 승차 기준), 따릉이 {avg['따릉이']:.0f}만 건입니다."),
                 ("수치", f"따릉이는 평균적으로 {by_m.idxmax()}월이 가장 많고({by_m.max():.0f}) {by_m.idxmin()}월이 가장 적습니다({by_m.min():.0f}). 2024년 4월은 지하철 원본 오류로 빠져 있습니다."),
                 ("왜 보나", "날씨를 그대로 맞는 수단일수록 계절을 탑니다. 이 차이가 아래 **질문 ②(교통수단 대비)**의 출발점입니다."))
    return _out(fig, "html"), desc


def c_mode_week():
    m = _trips()
    g = mob.days().reindex(m.index)
    m = m[g["hol"].isna() & ~g["labor"] & ~g["bridge"]]  # 공휴일·징검다리 빼고 요일 모양만
    wk = m.groupby(m.index.weekday).mean()
    idx = wk / wk.loc[0:4].mean() * 100
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    w = .26
    for i, c in enumerate(idx):
        ax.bar(np.arange(7) + (i - 1) * w, idx[c], w, color=MODE_COLORS[c], label=c, alpha=.9)
    ax.set_xticks(range(7))
    ax.set_xticklabels([f"{d}요일" for d in DOW])
    ax.axhline(100, color="#55534e", lw=.8)
    ax.set(title="요일별 하루 이용 (각 수단의 평일 평균 = 100)", ylabel="이용 지수", ylim=(0, max(125, idx.max().max() + 10)))
    ax.legend(ncol=3, fontsize=9, loc="upper left")
    sun = idx.loc[6]
    desc = _desc(f"일요일 이용은 평일의 지하철 **{sun['지하철']:.0f}%**, 버스 {sun['버스']:.0f}%, 따릉이 **{sun['따릉이']:.0f}%**입니다. 따릉이는 주말에도 덜 줄어 **여가 이동 비중이 큽니다**.",
                 ("읽는 법", "공휴일·징검다리를 뺀 날의 요일별 하루 평균을, 수단마다 평일(월~금) 평균 = 100으로 맞춘 지수입니다."),
                 ("왜 보나", "지하철·버스는 평일(출퇴근·통학) 중심의 수단이라는 점을 먼저 확인합니다. 아래 **질문 ③**에서 이 '필수 이동'이 날씨에 잘 흔들리지 않는다는 것을 봅니다."))
    return _out(fig, "html"), desc


def c_modes():
    r = mob.mode_table()
    cats, modes = mob.WX_CATS, ["지하철", "버스", "따릉이"]
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    w = .26
    for i, m in enumerate(modes):
        s = r[r["mode"] == m].set_index("wx").reindex(cats)
        x = np.arange(len(cats)) + (i - 1) * w
        ax.bar(x, s["m"], w, yerr=s["ci"], color=MODE_COLORS[m], label=m, alpha=.9,
               error_kw={"ecolor": "#55534e", "lw": .8, "capsize": 2})
    n = r.drop_duplicates("wx").set_index("wx")["n"].reindex(cats)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([f"{c}\n({int(k)}일)" for c, k in zip(cats, n)], fontsize=8.5)
    ax.axhline(0, color="#55534e", lw=.8)
    ax.set(title="같은 날씨, 다른 반응 — 교통수단별 평일 이용 변화 (같은 요일 보통날 대비)", ylabel="평균 변화 (%, 선=95% 신뢰구간)")
    ax.set_ylim(top=max(18, float((r["m"] + r["ci"]).max()) + 4))
    ax.legend(ncol=3, fontsize=9, loc="upper left")
    t = r.set_index(["wx", "mode"])["m"]
    heavy = "많은 비 (10mm↑)"
    desc = _desc(f"많은 비가 오는 평일, 따릉이는 **{t[(heavy, '따릉이')]:+.0f}%**로 급감하지만 지하철은 **{t[(heavy, '지하철')]:+.1f}%**, 버스는 {t[(heavy, '버스')]:+.1f}%에 그칩니다.",
                 ("읽는 법", "평일(공휴일·징검다리 제외) 하루 이용량을 같은 요일 앞뒤 4주 '맑은 보통날' 중앙값과 비교했습니다. 세 수단 모두 **같은 방법·같은 날짜**로 계산했습니다."),
                 ("수치", f"약한 비 지하철 {t[('약한 비', '지하철')]:+.1f}% · 따릉이 {t[('약한 비', '따릉이')]:+.0f}%, "
                         f"눈 지하철 {t[('눈', '지하철')]:+.1f}% · 따릉이 {t[('눈', '따릉이')]:+.0f}%, "
                         f"미세먼지 나쁨 지하철 {t[('미세먼지 나쁨', '지하철')]:+.1f}% · 따릉이 {t[('미세먼지 나쁨', '따릉이')]:+.0f}%."),
                 ("해석", "날씨에 그대로 노출되는 자전거는 크게 줄고, 지붕이 있는 지하철·버스는 거의 그대로입니다. "
                         "**지하철의 작은 감소는 '날씨 영향이 없다'가 아니라 '날씨를 피할 수 있는 수단'이라는 뜻**입니다."),
                 ("자료", "버스: 서울시 버스 정류장별 일별 승하차(서울 열린데이터광장), 따릉이: 공공자전거 이용정보(일별), "
                         "미세먼지: 서울시 일별 평균 대기오염도(25개 구 측정소 평균, PM10 81↑ 또는 PM2.5 36↑을 '나쁨'으로)."))
    return _out(fig, "png"), desc


def c_dumbbell():
    e = mob.mode_effects()
    g = mob.days().reindex(e.index)
    wet = g["wx"].isin(mob.WX_CATS[:2])
    wk, we = g["plain"] & (g["wd"] < 5), g["plain"] & (g["wd"] >= 5)
    order = ["지하철", "버스", "따릉이 정기권", "따릉이 일일권"]
    a = [e.loc[wet & wk, m].mean() for m in order]
    b = [e.loc[wet & we, m].mean() for m in order]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    y = np.arange(len(order))[::-1]
    for yy, x0, x1 in zip(y, a, b):
        ax.plot([x0, x1], [yy, yy], color="#cfcac0", lw=3, zorder=1)
    ax.scatter(a, y, s=70, color="#0f6b5c", label="평일", zorder=2)
    ax.scatter(b, y, s=70, color=C_ORANGE, label="주말", zorder=2)
    for yy, x0, x1 in zip(y, a, b):  # 더 작은(왼쪽) 값은 점 왼쪽에, 큰 값은 점 오른쪽에 적어 겹치지 않게
        for x, col in ((x0, "#0f6b5c"), (x1, C_ORANGE)):
            left = x == min(x0, x1)
            ax.text(x + (-.7 if left else .7), yy, f"{x:+.1f}%", fontsize=8.5, color=col,
                    ha="right" if left else "left", va="center")
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.axvline(0, color="#55534e", lw=.8)
    ax.set(title="비 오는 날 변화: 평일 vs 주말 (같은 요일 보통날 대비)", xlabel="평균 변화 (%)",
           xlim=(min(a + b) - 6, 2.5), ylim=(-.6, len(order) - .4))
    ax.legend(fontsize=9, loc="upper left", ncol=2)
    d = dict(zip(order, zip(a, b)))
    n_wk, n_we = int((wet & wk).sum()), int((wet & we).sum())
    desc = _desc(f"같은 비라도 **주말에 더 많이 줄어듭니다** — 지하철 평일 {d['지하철'][0]:+.1f}% → 주말 {d['지하철'][1]:+.1f}%.",
                 ("읽는 법", f"점 두 개 사이가 멀수록 '꼭 가야 하는 이동(평일)'과 '안 가도 되는 이동(주말)'의 차이가 큽니다. 비 오는 평일 {n_wk}일, 주말 {n_we}일(공휴일 제외)."),
                 ("수치", f"따릉이 정기권(반복 이용 → 출퇴근·통학) 평일 {d['따릉이 정기권'][0]:+.0f}% · 주말 {d['따릉이 정기권'][1]:+.0f}%, "
                         f"일일권(가끔 타는 이용) 평일 {d['따릉이 일일권'][0]:+.0f}% · 주말 {d['따릉이 일일권'][1]:+.0f}%. 자전거는 비를 그대로 맞아 평일에도 크게 줄어듭니다."),
                 ("해석", "평일 출퇴근은 날씨와 상관없이 해야 하는 **필수 이동**이라 거의 줄지 않고, 주말 나들이 같은 **선택 이동**이 날씨에 따라 취소됩니다. "
                         "지하철 이용 대부분이 필수 이동이라 전체 효과가 작게 나옵니다."))
    return _out(fig, "png"), desc


def c_station_map():
    info = mob.station_info().dropna(subset=["lat"])
    info = info[(info["lat"].between(37.4, 37.72)) & (info["lon"].between(126.75, 127.2))]  # 서울 밖 몇 역 제외
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    for t in mob.TYPES:
        s = info[info["type"] == t]
        ax.scatter(s["lon"], s["lat"], s=np.sqrt(s["wd_daily"]) / 2.2, color=TYPE_COLORS[t], alpha=.75,
                   edgecolor="white", lw=.5, label=f"{t} ({len(s)})")
    # 강남 일대는 큰 역이 몰려 있어 상위 N개를 다 적으면 겹친다 → 지역별 대표 역만 적는다
    for name in ("서울", "강남", "홍대입구", "잠실", "구로디지털단지", "노원", "여의도", "수유", "김포공항"):
        r = info[info["station"] == name]
        if len(r):
            ax.annotate(name, (r["lon"].iat[0], r["lat"].iat[0]), xytext=(5, 4), textcoords="offset points", fontsize=8.5,
                        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
    ax.set_aspect(1 / np.cos(np.radians(37.55)))
    ax.set(title="서울 지하철역 성격 지도 (원 크기 = 평일 하루 이용량)", xlabel="경도", ylabel="위도")
    ax.legend(fontsize=8.5, loc="lower right", markerscale=.6)
    cnt = info["type"].value_counts()
    desc = _desc("평일 아침 승하차 방향과 주말 이용 비율만으로 역을 나누면, **업무지구는 도심·강남에, 주거지는 외곽에** 모입니다.",
                 ("읽는 법", "점 하나가 역 하나이고 위치는 실제 좌표(서울시 역사마스터 정보)입니다. 원이 클수록 평일 이용량이 많습니다."),
                 ("분류 기준", "평일 7~9시 하차가 승차의 2배 이상이면 **업무지구**(일하러 오는 역), 승차가 하차의 2배 이상이면 **주거지**(집에서 나가는 역), "
                              "주말 하루 이용이 평일의 75% 이상이면 **여가·상업**, 나머지는 혼합입니다."),
                 ("수치", ", ".join(f"{t} {cnt.get(t, 0)}곳" for t in mob.TYPES) + ". 이름은 데이터로 붙인 것이라 실제 용도와 다를 수 있습니다."))
    return _out(fig, "png"), desc


def c_type_rain():
    r = mob.type_table()
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    w = .38
    for i, (lab, col) in enumerate((("평일", "#0f6b5c"), ("주말", C_ORANGE))):
        s = r[r["day"] == lab].set_index("type").reindex(mob.TYPES)
        ax.bar(np.arange(len(mob.TYPES)) + (i - .5) * w, s["m"], w, yerr=s["ci"], color=col, label=lab, alpha=.9,
               error_kw={"ecolor": "#55534e", "lw": .8, "capsize": 3})
    ax.set_xticks(range(len(mob.TYPES)))
    ax.set_xticklabels(mob.TYPES)
    ax.axhline(0, color="#55534e", lw=.8)
    ax.set(title="역 성격별 비 오는 날 이용 변화 (같은 요일 보통날 대비)", ylabel="평균 변화 (%, 선=95% 신뢰구간)")
    ax.set_ylim(top=max(2.2, float((r["m"] + r["ci"]).max()) + 1))
    ax.legend(fontsize=9, loc="upper right", ncol=2)
    t = r.set_index(["type", "day"])["m"]
    wk = t.xs("평일", level="day")
    we = t.xs("주말", level="day")
    desc = _desc(f"평일에는 역 성격과 상관없이 {wk.min():+.1f}~{wk.max():+.1f}%로 비슷하지만, 주말에는 **{we.idxmin()} {we.min():+.1f}%**로 가장 크게 줄어듭니다.",
                 ("읽는 법", "같은 성격의 역들을 합친 하루 이용량을 같은 요일 앞뒤 4주 맑은 보통날과 비교했습니다. 비 = 하루 강수 0.1mm 초과(눈 오는 날 제외)."),
                 ("해석", "평일에는 어디서 타든 출퇴근이라 차이가 작습니다. "
                         + (f"주말에 업무지구 역이 가장 많이 줄어드는 것은, 쉬는 날 도심·강남에 가는 이동(쇼핑·약속)이 비 오는 날 미루기 쉬운 이동이기 때문으로 보입니다."
                            if we.idxmin() == "업무지구" else f"주말에는 {we.idxmin()} 역이 가장 많이 줄어듭니다.")
                         + (f" 여가·상업 역({we['여가·상업']:+.1f}%)은 실내 상권·행사가 있어 상대적으로 덜 줄어듭니다." if we.idxmax() == "여가·상업" else "")),
                 ("주의", f"역 성격은 위 지도의 데이터 기준 분류입니다. 여가·상업 역은 {int((mob.station_info()['type'] == '여가·상업').sum())}곳으로 적습니다."))
    return _out(fig, "png"), desc


def c_transfer():
    lost = mob.lost_trips()
    s = mob.station_rain()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})
    cats, modes = mob.WX_CATS[:2], ["지하철", "버스", "따릉이"]
    y = np.arange(len(modes))[::-1]
    for i, (c, col) in enumerate(zip(cats, ("#7fa3c6", C_BLUE))):
        a1.barh(y + (i - .5) * .38, [lost.loc[c, m] for m in modes], .38, color=col, label=c)
    a1.set_yticks(y)
    a1.set_yticklabels(modes)
    a1.axvline(0, color="#55534e", lw=.8)
    a1.set(title="① 비 오는 평일에 줄어든 이동 (하루, 만 건)", xlabel="보통날 대비 증감 (만 건)")
    a1.legend(fontsize=8, loc="lower left")
    for t in mob.TYPES:
        x = s[s["type"] == t]
        a2.scatter(x["bike_share"], x["sub_rain"], s=np.sqrt(x["use"]) / 3, color=TYPE_COLORS[t], alpha=.7,
                   edgecolor="white", lw=.5, label=t)
    xs = np.log10(s["bike_share"])
    k, b0 = np.polyfit(xs, s["sub_rain"], 1)
    gx = np.linspace(xs.min(), xs.max(), 50)
    a2.plot(10 ** gx, k * gx + b0, color=C_RED, lw=1.6, label="추세선")
    a2.set_xscale("log")
    a2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))  # 10^-1 같은 수식 눈금 대신 0.1
    a2.axhline(0, color="#55534e", lw=.8)
    a2.set(title="② 역 주변 따릉이 비율 vs 비 오는 평일 지하철 변화", xlabel="역 300m 안 따릉이 대여 ÷ 지하철 이용 (%, 로그)",
           ylabel="지하철 변화 (%)")
    a2.legend(fontsize=7.5, ncol=3, loc="lower right")
    fig.tight_layout()
    r = np.corrcoef(xs, s["sub_rain"])[0, 1]
    h = mob.WX_CATS[1]
    tot = sum(lost.loc[h, m] for m in modes)
    all_down = all(lost.loc[h, m] < 0 for m in modes)
    share_max = s["bike_share"].quantile(.9)
    desc = _desc((f"비 오는 날 자전거에서 빠진 사람이 지하철로 옮겨 탄 흔적은 **보이지 않습니다** — 세 수단이 **모두** 줄어, 많은 비 오는 평일 하루 약 **{-tot:.0f}만 건**의 이동이 사라집니다."
                  if all_down and abs(r) < .2 else
                  f"많은 비 오는 평일, 세 수단 합계로 하루 약 {-tot:.0f}만 건의 이동이 바뀝니다(역 단위 r = {r:+.2f})."),
                 ("읽는 법", "① 수단별로 '보통날이었다면'의 값과 실제 값의 차이(지하철·버스는 승차 인원, 따릉이는 대여 건수). "
                            f"② 점 하나가 역 하나({len(s)}곳)이고, 자전거를 많이 타는 동네의 역일수록 비 오는 날 지하철이 덜 줄어드는지(=옮겨 타는지) 본 것입니다."),
                 ("수치", f"많은 비: 지하철 {lost.loc[h, '지하철']:+.0f}만, 버스 {lost.loc[h, '버스']:+.0f}만, 따릉이 {lost.loc[h, '따릉이']:+.1f}만 건. "
                         f"② 상관계수 r = {r:+.2f}. 역 주변 따릉이는 지하철 이용의 중앙값 {s['bike_share'].median():.2f}%에 불과합니다."),
                 ("해석", f"역 주변 따릉이 대여는 지하철 이용의 상위 10% 역에서도 {share_max:.1f}% 수준이라, 다 옮겨 타도 지하철 숫자는 거의 변하지 않습니다. "
                         "줄어든 이동은 다른 수단으로 옮겨 갔다기보다 **미뤄지거나 취소된 이동**(또는 자료에 없는 택시·자가용)으로 보는 것이 맞습니다."),
                 ("주의", "택시·자가용 자료는 없어 확인하지 못했습니다. 대여소 위치는 2025년 12월 기준입니다."))
    return _out(fig, "png"), desc


# ---------- 조립 ----------
def _card(cid, title, fn, *args):
    html, desc = fn(*args)
    return {"id": cid, "title": title, "html": html, "desc": _render_desc(desc)}


def _md(text: str) -> Markup:
    """문장을 HTML로 이스케이프한 뒤 **굵게**만 <b>로 바꾼다."""
    return Markup(re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", str(escape(text))))


def _render_desc(d: dict) -> dict:
    return {"key": _md(d["key"]), "paras": [(lab, _md(txt)) for lab, txt in d["paras"]]}


@lru_cache(maxsize=64)
def sections(line: str = "전체", dtype: str = "전체"):
    with _lock:
        return [
            {"tab": "데이터 개요", "title": "1. 데이터 개요 — 서울 시민은 언제, 무엇을 타고 이동하는가", "cards": [
                _card("mode-month", "교통수단별 월별 이용 추이", c_mode_month),
                _card("mode-week", "교통수단별 요일 패턴", c_mode_week),
                _card("heatmap", "요일×시간대 이용량", c_heat, line, dtype),
                _card("lines", "호선별 하루 평균 이용량", c_lines, line, dtype),
                _card("stations", "이용량 많은 역 Top 15", c_station, line, dtype),
                _card("ptype", "승객 유형별 시간대 구성", c_ptype_share)]},
            {"tab": "서울의 날씨", "title": "2. 2023~2025년 서울의 날씨 — 무엇과 비교하는가", "cards": [
                _card("wmonth", "월별 기온과 강수량", c_weather_month),
                _card("thist", "시간별 기온 분포", c_temp_hist)]},
            {"tab": "질문① 지하철", "title": "3. 질문 ① 날씨가 나쁘면 지하철을 덜 타는가 — 상관과 크기", "cards": [
                _card("corr", "상관계수 히트맵", c_corr, line, dtype),
                _card("scatter", "기온과 이용량 (보정 전·후)", c_scatter, line, dtype),
                _card("wbox", "날씨 유형별 하루 이용량 분포", c_weather_box, line, dtype),
                _card("factors", "날씨 구간별 영향 (신뢰구간)", c_factors, line, dtype)]},
            {"tab": "질문② 수단 비교", "title": "4. 질문 ② 다른 교통수단은 어떤가 — 지하철·버스·따릉이 대비", "cards": [
                _card("modes", "교통수단별 날씨 반응 (미세먼지 포함)", c_modes),
                _card("transfer", "줄어든 이동은 어디로 갔나", c_transfer)]},
            {"tab": "질문③ 누가·어디서", "title": "5. 질문 ③ 누가, 어디서 줄이는가 — 필수 이동과 선택 이동", "cards": [
                _card("dumbbell", "평일 vs 주말: 같은 비, 다른 반응", c_dumbbell),
                _card("commute", "출퇴근 시간대: 맑음 vs 비·눈", c_rain_commute, line, dtype),
                _card("station-map", "역 성격 지도", c_station_map),
                _card("type-rain", "역 성격별 비 오는 날 변화", c_type_rain),
                _card("ptype-rain", "승객 유형별 반응", c_ptype_rain),
                _card("line-rain", "호선별 반응", c_line_rain)]},
            {"tab": "다른 요인", "title": "6. 다른 요인과 비교 — 날씨는 얼마나 큰가", "cards": [
                _card("factor-size", "하루 이용량을 바꾸는 요인 크기", c_factor_size),
                _card("spikes", "역 단위로 보면: 평소보다 가장 붐빈 날", c_station_spikes)]},
            {"tab": "예측 검증", "title": "7. 그래서 — 예측하고 검증한다", "cards": [
                _card("backtest", "백테스트: 실제 vs 예측", c_backtest)]},
        ]


@lru_cache(maxsize=1)
def headline():
    """화면 맨 위 핵심 숫자 4개."""
    t = t_for("전체", "전체")
    rain = _effect(t, lambda x: (~x["snow"]) & (x["precip_mm"] > 0.1))
    snow = _effect(t, lambda x: x["snow"])
    cold = _effect(t, lambda x: x["dry"] & (x["temp_c"] <= 0))
    hot = _effect(t, lambda x: x["dry"] & (x["temp_c"] > 28))
    d = _daily(t)
    r = max((d["ratio"].corr(d[c]) for c in WEATHER_LABELS), key=abs)
    _, m, _, _ = backtest()
    mt = mob.mode_table().set_index(["wx", "mode"])["m"]
    bike = mt[("많은 비 (10mm↑)", "따릉이")]
    return [
        {"value": _pct(rain), "label": "비 오는 날 지하철", "icon": "rain", "tone": "rain"},
        {"value": _pct(snow), "label": "눈 오는 날 지하철", "icon": "snow", "tone": "snow"},
        {"value": f"{bike:+.0f}%", "label": "많은 비 오는 평일 따릉이", "icon": "bike", "tone": "bike"},
        {"value": f"{_pct(cold)} / {_pct(hot)}", "label": "한파(0℃↓) / 폭염(28℃↑) 지하철", "icon": "thermo", "tone": "temp"},
        {"value": f"|r| ≤ {abs(r):.2f}", "label": "날씨와 지하철 하루 이용량의 상관계수", "icon": "scatter", "tone": "corr"},
    ]


def warm():
    """첫 접속이 느리지 않게 미리 계산해 둔다."""
    try:
        sections("전체", "전체")
        headline()
    except Exception:
        pass


threading.Thread(target=warm, daemon=True).start()
