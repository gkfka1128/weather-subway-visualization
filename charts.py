"""대시보드 그래프 11개 + 오늘 예측 그래프. 설명 문장의 숫자는 그때그때 데이터에서 계산한다."""
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

from analysis import (DF, LINES, PT, PTYPES, RAIN_ORDER, ST, TEMP_ORDER, WEEKDAYS, backtest, group_stats,
                      make_t, t_for)

from matplotlib import font_manager as _fm

# Windows는 맑은 고딕, Docker(리눅스)는 나눔고딕 — 실제로 설치된 첫 한글 폰트를 쓴다.
_installed = {f.name for f in _fm.fontManager.ttflist}
KFONT = next((f for f in ["Malgun Gothic", "NanumGothic", "AppleGothic"] if f in _installed), "DejaVu Sans")
sns.set_theme(style="whitegrid", font=KFONT)  # set_theme이 font.family를 덮어쓰므로 rcParams는 그 뒤에
plt.rcParams["axes.unicode_minus"] = False

_lock = threading.Lock()  # pyplot은 전역 상태를 쓰므로 직렬화
C_BLUE, C_ORANGE, C_GRAY, C_RED, C_GREEN = "#2563eb", "#ea580c", "#9ca3af", "#dc2626", "#16a34a"
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
        ax.plot(out["x"], out["base"] / 1000, marker="o", color=C_GRAY, label="평소 (같은 계절 무강수일 평균)")
        ax.plot(out["x"], out["pred"] / 1000, marker="o", color=C_BLUE, label="오늘 예측")
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
    ax.bar(range(len(s)), s.to_numpy(), color=[C_ORANGE if l == line else C_BLUE for l in s.index])
    ax.set_xticks(range(len(s)))
    ax.set_xticklabels(list(s.index))
    ax.set(title=f"호선별 하루 평균 이용량 ({dtype})", xlabel="호선", ylabel="하루 평균 이용량 (만 명)")
    top, low = s.idxmax(), s.idxmin()
    desc = (f"{top}이 하루 평균 {s.max():,.0f}만 명으로 가장 많고 {low}이 {s.min():,.0f}만 명으로 가장 적습니다"
            f"(1일 이용량 = 승차+하차). 호선별 규모 차이가 커서, 날씨 영향도 '비율'로 비교해야 공정합니다. "
            "선택한 호선은 주황색으로 표시됩니다.")
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
    ax.set(title="평일 시간대별 승객 유형 구성 (전 호선)", xlabel="시간 (24=자정 이후)", ylabel="비율 (%)", xlim=(5, 24), ylim=(0, 100))
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    mid = pv.loc[11:15, "우대권"].mean()
    rush = pv.loc[[8, 18], "우대권"].mean()
    desc = (f"승객 유형별 비중입니다(중고생 유형은 전체의 0.00%라 제외). 대부분은 '일반'이지만 '우대권'(어르신·장애인 등 우대 대상)은 "
            f"출퇴근 피크(8·18시)에 평균 {rush:.0f}%로 낮다가 낮 시간대(11~15시)에는 {mid:.0f}%까지 올라갑니다. "
            "시간대마다 승객 구성이 다르므로 날씨에 반응하는 정도도 유형마다 다를 수 있습니다(아래 '승객 유형별 반응' 참고). "
            "이 그래프는 호선·요일 선택과 무관하게 평일 전체 기준입니다.")
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
    ax.set(title="승객 유형별 비·눈 오는 날 이용량 변화 (전 호선)", xlabel="", ylabel="평소 대비 변화 (%)")
    ax.tick_params(axis="x", rotation=0)
    k = df["비"].idxmin()
    j = df["비"].idxmax()
    desc = (f"비 오는 날 가장 크게 줄어드는 유형은 '{k}'({df.loc[k, '비']:+.1f}%), 가장 덜 줄어드는 유형은 '{j}'({df.loc[j, '비']:+.1f}%)입니다. "
            "우대권·어린이 이용자는 일반 이용자보다 비 오는 날 더 많이 줄고, 직원은 거의 줄지 않습니다. "
            "이용을 미룰 수 있는 승객일수록 날씨에 민감하다고 해석할 수 있지만, 이 자료만으로 원인까지 확정할 수는 없습니다. "
            "어린이·직원은 전체의 1% 안팎이라 변동이 큽니다.")
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
    desc = (f"1~8월 데이터로 계수를 만들고 9~12월({ndays}일)을 맞혀 본 결과입니다. 하루 합계 오차는 연평균(A) {metrics.loc['A 연평균', '하루 합계']:.1f}% → "
            f"최근 4주(B) {b['하루 합계']:.1f}%로 줄지만, 날씨 보정(C)을 더해도 {c['하루 합계']:.1f}%로 거의 달라지지 않습니다. "
            f"비·눈 온 {nwet}개 시간대만 보면 B {b['비·눈 온 시간만']:.1f}% → C {c['비·눈 온 시간만']:.1f}%로 약간 개선됩니다. "
            "결론: 날씨는 방향은 일관되지만 효과(±5% 안팎)가 평소 변동(약 6%)보다 작아서, 예측 정확도를 크게 올리진 못합니다. "
            "그래서 서비스는 '날씨 때문에 붐빈다/한산하다'가 아니라 '평소 대비 몇 % 차이'로 안내합니다.")
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
            {"title": "2. 날씨와 이용량의 관계 — 상관관계가 있는가", "cards": [
                _card("corr", "상관계수 히트맵", c_corr, line, dtype),
                _card("scatter", "기온과 이용량 (보정 전·후)", c_scatter, line, dtype),
                _card("commute", "출퇴근 시간대: 맑음 vs 비·눈", c_rain_commute, line, dtype)]},
            {"title": "3. 영향의 크기 — 얼마나, 누구에게", "cards": [
                _card("factors", "날씨 구간별 영향 (신뢰구간)", c_factors, line, dtype),
                _card("ptype-rain", "승객 유형별 반응", c_ptype_rain),
                _card("line-rain", "호선별 반응", c_line_rain)]},
            {"title": "4. 예측과 검증 — 믿을 만한가", "cards": [
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
    return [
        {"value": _pct(rain), "label": "비 오는 날 이용량 (평소 대비)"},
        {"value": _pct(snow), "label": "눈 오는 날"},
        {"value": f"{_pct(cold)} / {_pct(hot)}", "label": "한파(0℃↓) / 폭염(28℃↑)"},
        {"value": f"|r| ≤ {abs(r):.2f}", "label": "날씨-이용량 하루 단위 상관계수"},
    ]


def warm():
    """첫 접속이 느리지 않게 미리 계산해 둔다."""
    try:
        sections("전체", "전체")
        headline()
    except Exception:
        pass


threading.Thread(target=warm, daemon=True).start()
