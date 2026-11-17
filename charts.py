"""대시보드 그래프 16개 + 오늘 예측 그래프. 설명 문장의 숫자는 그때그때 데이터에서 계산한다."""
import base64
import io
import threading
from functools import lru_cache

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpld3
import numpy as np
import pandas as pd
import seaborn as sns

from analysis import (_HOUR_WEATHER, CUT, DF, LINES, ROOT, PT, PTYPES, RAIN_ORDER, ST, TEMP_ORDER, WEEKDAYS, backtest, group_stats,
                      make_t, t_for)

from matplotlib import font_manager as _fm
from matplotlib.colors import to_rgba

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
    desc = (f"색이 진할수록 이용량이 많습니다. 가장 붐비는 칸은 {dy}요일 {dh}시(시간당 평균 {top / 1000:,.0f}천 명)이며, "
            "평일에는 출근(7~9시)과 퇴근(17~19시) 두 개의 봉우리가, 주말에는 낮 시간대의 낮고 넓은 봉우리가 보입니다. "
            "날씨 영향을 보기 전에 먼저 '이용량이 원래 시간·요일에 따라 크게 달라진다'는 점을 확인하는 그래프입니다.")
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
    desc = (f"{top}이 하루 평균 {s.max():,.0f}만 명으로 가장 많고 {low}이 {s.min():,.0f}만 명으로 가장 적습니다"
            f"(1일 이용량 = 승차+하차). 호선별 규모 차이가 커서, 날씨 영향도 '비율'로 비교해야 공정합니다. "
            "막대 색은 각 호선의 고유색이고, 호선을 고르면 그 호선만 진하게 표시됩니다.")
    return _out(fig, "html"), desc


def c_station(line, dtype):
    sub = ST if line == "전체" else ST[ST["line"] == line]
    sub = sub if dtype == "전체" else sub[sub["day_type"] == dtype]
    n = sub["date"].nunique()
    s = (sub.groupby(["station", "line"])["use"].sum() / n / 10000).nlargest(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.barh(range(len(s)), s.to_numpy(), color=C_BLUE)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels([f"{a}({b})" for a, b in s.index])
    ax.set(title=f"이용량 많은 역 15곳 ({_scope(line, dtype)})", xlabel="하루 평균 이용량 (만 명)")
    a, b = s.index[-1]
    desc = (f"{a}역({b})이 하루 평균 {s.iloc[-1]:,.1f}만 명으로 1위입니다. 역별 자료는 시간대 구분이 없는 일 단위라서, "
            "예측은 호선 단위로 하고 역은 규모를 확인하는 용도로 씁니다.")
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
    desc = (f"승객 유형별 비중입니다(중고생 유형은 전체의 0.00%라 제외). 대부분은 '일반'이지만 '우대권'(어르신·장애인 등 우대 대상)은 "
            f"출퇴근 피크(8·18시)에 평균 {rush:.0f}%로 낮다가 낮 시간대(11~15시)에는 {mid:.0f}%까지 올라갑니다. "
            "시간대마다 승객 구성이 다르므로 날씨에 반응하는 정도도 유형마다 다를 수 있습니다(아래 '승객 유형별 반응' 참고). "
            "이 그래프는 호선·요일 선택과 무관하게 평일 전체 기준이고, 승객 유형 구분이 있는 2025년 원본만 씁니다.")
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
    desc = (f"하루 단위 상관계수(−1~+1)입니다. 왼쪽은 요일 유형만 보정한 값, 오른쪽은 계절·연휴 영향까지 걷어낸 값입니다. "
            f"보정 후 가장 큰 변수는 '{k}'(r={adj[k]:+.2f})이고, 모든 값의 절댓값이 {adj.abs().max():.2f} 이하라서 "
            "날씨와 이용량의 직선 관계는 약합니다. 왼쪽과 오른쪽 차이가 크면 그만큼이 '날씨가 아니라 계절 때문'이었다는 뜻입니다.")
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
    desc = (f"기온과 하루 이용량의 관계입니다. 보정 전에는 구간 평균이 {raw_rng.min():.2f}~{raw_rng.max():.2f}로 "
            f"{(raw_rng.max() - raw_rng.min()) * 100:.0f}%p 벌어지지만, 보정 후에는 {adj_rng.min():.2f}~{adj_rng.max():.2f}"
            f"({(adj_rng.max() - adj_rng.min()) * 100:.0f}%p)로 좁아집니다. 즉 '추운 날 이용량이 적다'는 현상의 상당 부분은 "
            "기온 자체가 아니라 계절·연휴 요인이었습니다.")
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
        txt = f"출퇴근 6개 시간대 합계로 비 오는 날은 맑은 날보다 {diff * 100:+.1f}%입니다. "
    else:
        txt = ""
    if weekend:
        txt = "출퇴근 비교라 요일 유형 선택과 관계없이 평일 자료만 씁니다. " + txt
    desc = (txt + f"다만 단순 평균 비교라 계절이 섞여 있고, 비·눈이 온 날 수가 적습니다(비 {cnt.get('비', 0)}일, 눈 {cnt.get('눈', 0)}일). "
            "출퇴근은 날씨가 나빠도 가야 하므로 감소폭이 작습니다. 정밀한 비교는 아래 '날씨 구간별 영향' 그래프를 보세요.")
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
    desc = (f"각 시간대의 실제 이용량을 '같은 계절·요일 유형·시간대의 무강수일 평균'으로 나눈 비율의 평균입니다. "
            f"가장 크게 줄어드는 구간은 '{worst[0]}'({(worst[1] - 1) * 100:+.1f}%)이고, 기온 구간의 차이는 대부분 ±2% 안쪽입니다. "
            + (f"신뢰구간이 0을 포함하는 구간({', '.join(wide)})은 '차이가 없다'와 구분되지 않으므로 해석에 주의하세요. " if wide else "")
            + "이 비율이 오늘 예측에 곱해지는 보정 계수입니다.")
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
    desc = (f"비 오는 날 가장 크게 줄어드는 유형은 '{k}'({df.loc[k, '비']:+.1f}%), 가장 덜 줄어드는 유형은 '{j}'({df.loc[j, '비']:+.1f}%)입니다. "
            "우대권·어린이 이용자는 일반 이용자보다 비 오는 날 더 많이 줄고, 직원은 거의 줄지 않습니다. "
            "이용을 미룰 수 있는 승객일수록 날씨에 민감하다고 해석할 수 있지만, 이 자료만으로 원인까지 확정할 수는 없습니다. "
            "어린이·직원은 전체의 1% 안팎이라 변동이 큽니다. 승객 유형 구분은 2025년 원본에만 있어 2025년 1년치로 계산했습니다.")
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
    desc = (f"비 오는 날 {k}이 {df.loc[k, '비']:+.1f}%로 가장 크게, {j}이 {df.loc[j, '비']:+.1f}%로 가장 적게 줄었습니다. "
            "호선마다 이용 목적(출퇴근·여가 등)이 달라 반응 정도가 다를 수 있으나, 이 자료만으로 원인까지 확정하긴 어렵습니다. "
            "호선 선택 콤보박스와 무관하게 전체 호선을 비교합니다.")
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
    desc = (f"{CUT[:4]}년 이전 데이터로 계수를 만들고 {CUT[:4]}년 1년({ndays}일)을 맞혀 본 결과입니다(학습에 쓰지 않은 한 해 전체라 계절이 모두 들어갑니다). 하루 합계 오차는 연평균(A, 이전 해들의 평균이라 해마다의 수준 변화까지 오차에 들어감) {metrics.loc['A 연평균', '하루 합계']:.1f}% → "
            f"최근 4주(B) {b['하루 합계']:.1f}%로 줄지만, 날씨 보정(C)을 더해도 {c['하루 합계']:.1f}%로 거의 달라지지 않습니다. "
            f"비·눈 온 {nwet}개 시간대만 보면 B {b['비·눈 온 시간만']:.1f}% → C {c['비·눈 온 시간만']:.1f}%로 약간 개선됩니다. "
            "결론: 날씨는 방향은 일관되지만 효과(±5% 안팎)가 평소 변동(약 6%)보다 작아서, 예측 정확도를 크게 올리진 못합니다. "
            "그래서 서비스는 '날씨 때문에 붐빈다/한산하다'가 아니라 '평소 대비 몇 % 차이'로 안내합니다.")
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
    desc = (f"{yrs[0]}~{yrs[-1]}년 {len(yrs)}년 평균입니다. 선은 월 평균 기온, 띠는 하루 최저~최고 기온의 평균, 막대는 월 강수량 평균이고 "
            f"점은 해마다의 월 강수량입니다. 여름 강수량은 같은 달이라도 해마다 최대 {swing:.0f}배 차이가 납니다. "
            f"가장 추운 달은 {cold_m}월({m.loc[cold_m, 'temp']:.1f}℃), 가장 더운 달은 {hot_m}월({m.loc[hot_m, 'temp']:.1f}℃)이고, "
            f"비는 평균적으로 {wet_m}월({m.loc[wet_m, 'precip']:,.0f}mm)에 가장 많이 오고 6~8월이 한 해 강수량의 {share:.0%}를 차지합니다. "
            "비 오는 날이 여름에 몰려 있어서, 그냥 비교하면 '비 효과'에 '여름 효과'가 섞입니다. "
            "그래서 이 대시보드는 같은 시기의 맑은 날과 비교하는 기준선을 씁니다.")
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
    desc = (f"지하철 운행 시간대 {len(t):,}개 시간의 기온을 1.5℃ 간격으로 센 히스토그램이고, 곡선은 분포를 부드럽게 그린 KDE입니다. "
            f"평균 {t.mean():.1f}℃, 가장 낮은 {t.min():.1f}℃부터 가장 높은 {t.max():.1f}℃까지 퍼져 있습니다. "
            f"0℃ 이하는 전체 시간의 {cold:.1%}, 28℃ 초과는 {hot:.1%}입니다. "
            "분포가 영하부터 30℃대까지 넓게 퍼져 있어서, 기온은 평균 하나보다 구간(한파·보통·폭염)으로 나눠 영향을 보는 것이 맞습니다.")
    return _out(fig, "png"), desc


def _box_verdict(med, iqr, order) -> str:
    """상자그림 설명의 결론 문장: 날씨별 중앙값 차이를 맑은 날 상자 폭과 실제로 비교해서 고른다."""
    base = order[0]
    if base not in med.index or not np.isfinite(iqr.get(base, np.nan)):
        return "맑은 날 표본이 부족해 날마다 흔들림과 비교하기 어렵습니다."
    gap = max((abs(med[k] - med[base]) for k in order[1:] if k in med.index), default=0.0)
    w = iqr[base]
    if gap <= w:
        return (f"맑은 날끼리도 상자 폭이 {w:.1f}%p인데 날씨별 중앙값 차이는 가장 커도 {gap:.1f}%p라서, "
                "날씨 때문에 생기는 차이가 평소 날마다 흔들림보다 작다는 것을 보여 줍니다.")
    return (f"날씨별 중앙값 차이가 가장 큰 경우 {gap:.1f}%p로 맑은 날 상자 폭({w:.1f}%p)보다 커서, "
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
    desc = (f"하루 이용량을 같은 시기 맑은 날 기준선과 비교한 값(%)의 상자그림입니다. 상자는 가운데 50%, 가운데 선은 중앙값입니다. "
            f"중앙값은 {parts}입니다. "
            + _box_verdict(med, iqr, order) + " 눈·많은 비는 날짜 수가 적어 해석에 주의해야 합니다."
            + (f" (연휴 앞뒤처럼 극단적인 {cut}일은 세로축 밖이라 생략했습니다.)" if cut else ""))
    return _out(fig, "png"), desc


# ---------- 다른 요인과 비교 (달력·행사) ----------
HOLIDAYS = pd.read_csv(ROOT / "data" / "holidays.csv").set_index("date")["name"].to_dict()
T95 = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26, 15: 2.14, 20: 2.09, 30: 2.05}


def _t95(n: int) -> float:
    """표본 n개의 95% 신뢰구간 t값 (scipy 없이 근사표)."""
    return next((v for k, v in sorted(T95.items()) if n <= k), 1.96)
DOW = "월화수목금토일"


def _same_weekday_base(values: pd.Series, skip: pd.Series, k: int = 4) -> pd.Series:
    """같은 요일 앞뒤 k주(자기 자신·skip인 날 제외, 보통날 3일 이상)의 중앙값. values는 같은 요일끼리 날짜순으로 정렬돼 있어야 한다."""
    v = values.where(~skip)

    def med(k):
        nb = pd.concat([v.shift(i) for i in range(-k, k + 1) if i], axis=1)
        return nb.median(axis=1).where(nb.notna().sum(axis=1) >= 3)
    return med(k).fillna(med(2 * k))  # 연말·연휴가 몰린 때는 보통날이 모자라 앞뒤 8주로 넓힌다


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
    desc = (f"각 날의 하루 이용량을 같은 요일 앞뒤 4주의 보통날(공휴일·징검다리·근로자의 날·비·눈이 아닌 날) 중앙값과 비교했습니다. "
            f"빨간 막대(달력)는 설·추석 연휴 {big.loc[cal[0], 'm']:+.0f}%, 그 밖의 공휴일 {big.loc[cal[1], 'm']:+.0f}%, "
            f"근로자의 날 {big.loc[cal[2], 'm']:+.0f}%, 징검다리 평일 {big.loc[cal[3], 'm']:+.0f}%로 줄고, "
            f"파란 막대(날씨)는 가장 큰 것도 {wmax:.1f}% 안팎입니다. "
            f"지하철 이용량을 가장 크게 흔드는 것은 날씨가 아니라 '그날 출근·등교를 하느냐'입니다. "
            "날씨 효과가 작은 것은 실패가 아니라, 지하철 이동 대부분이 날씨와 상관없이 해야 하는 이동이라는 결과입니다.")
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
    desc = (f"역마다 하루 이용량을 같은 요일 앞뒤 4주 중앙값과 비교해, 가장 크게 튄 날을 역별로 하나씩 뽑은 상위 10곳입니다(하루 평균 5천 명 이상 역). "
            f"1위는 {first['station']}({first['line']}) {first['date']}로 평소의 {first['r']:.1f}배입니다. "
            + (f"{many.index[0][1]}역은 평소의 3배 이상인 날이 {many.iloc[0]}일이나 됩니다. " if len(many) else "")
            + "경기장·한강공원 근처 역이 많아, 경기·공연·축제 같은 행사가 원인으로 추정됩니다(날짜별 행사는 별도 확인 필요). "
            "노선 전체로 보면 날씨 효과는 몇 %지만, 역 하나만 보면 행사 하나가 이용량을 몇 배로 바꿉니다.")
    return _out(fig, "png"), desc


# ---------- 조립 ----------
def _card(cid, title, fn, *args):
    html, desc = fn(*args)
    return {"id": cid, "title": title, "html": html, "desc": desc}


@lru_cache(maxsize=64)
def sections(line: str = "전체", dtype: str = "전체"):
    with _lock:
        return [
            {"title": "1. 데이터 개요 — 이용량은 언제, 어디서, 누가 많이 타는가", "cards": [
                _card("heatmap", "요일×시간대 이용량", c_heat, line, dtype),
                _card("lines", "호선별 하루 평균 이용량", c_lines, line, dtype),
                _card("stations", "이용량 많은 역 Top 15", c_station, line, dtype),
                _card("ptype", "승객 유형별 시간대 구성", c_ptype_share)]},
            {"title": "2. 2023~2025년 서울의 날씨 — 무엇과 비교하는가", "cards": [
                _card("wmonth", "월별 기온과 강수량", c_weather_month),
                _card("thist", "시간별 기온 분포", c_temp_hist)]},
            {"title": "3. 날씨와 이용량의 관계 — 상관관계가 있는가", "cards": [
                _card("corr", "상관계수 히트맵", c_corr, line, dtype),
                _card("scatter", "기온과 이용량 (보정 전·후)", c_scatter, line, dtype),
                _card("wbox", "날씨 유형별 하루 이용량 분포", c_weather_box, line, dtype),
                _card("commute", "출퇴근 시간대: 맑음 vs 비·눈", c_rain_commute, line, dtype)]},
            {"title": "4. 영향의 크기 — 얼마나, 누구에게", "cards": [
                _card("factors", "날씨 구간별 영향 (신뢰구간)", c_factors, line, dtype),
                _card("ptype-rain", "승객 유형별 반응", c_ptype_rain),
                _card("line-rain", "호선별 반응", c_line_rain)]},
            {"title": "5. 예측과 검증 — 믿을 만한가", "cards": [
                _card("backtest", "백테스트: 실제 vs 예측", c_backtest)]},
            {"title": "6. 다른 요인과 비교 — 날씨는 얼마나 큰가", "cards": [
                _card("factor-size", "하루 이용량을 바꾸는 요인 크기", c_factor_size),
                _card("spikes", "역 단위로 보면: 평소보다 가장 붐빈 날", c_station_spikes)]},
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
    return [
        {"value": _pct(rain), "label": "비 오는 날 이용량", "icon": "rain", "tone": "rain"},
        {"value": _pct(snow), "label": "눈 오는 날 이용량", "icon": "snow", "tone": "snow"},
        {"value": f"{_pct(cold)} / {_pct(hot)}", "label": "한파(0℃↓) / 폭염(28℃↑)", "icon": "thermo", "tone": "temp"},
        {"value": f"|r| ≤ {abs(r):.2f}", "label": "날씨와 하루 이용량의 상관계수", "icon": "scatter", "tone": "corr"},
    ]


def warm():
    """첫 접속이 느리지 않게 미리 계산해 둔다."""
    try:
        sections("전체", "전체")
        headline()
    except Exception:
        pass


threading.Thread(target=warm, daemon=True).start()
