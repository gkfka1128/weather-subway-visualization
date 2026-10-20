"""데이터 시각화 과제 - Flask + matplotlib/seaborn + mpld3 대시보드 (뼈대).

데이터가 정해지면 load_data()와 CHARTS 목록만 채우면 된다.
"""
import base64
import io
import threading

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpld3
import numpy as np
import pandas as pd
import seaborn as sns
from flask import Flask, render_template

plt.rcParams["font.family"] = ["Malgun Gothic", "NanumGothic", "AppleGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font=plt.rcParams["font.family"][0])

app = Flask(__name__)
_plot_lock = threading.Lock()  # pyplot은 전역 상태를 쓰므로 요청 간 직렬화


def load_data() -> pd.DataFrame:
    """TODO: 데이터셋 확정 후 교체 (data/ 폴더의 CSV 또는 sns.load_dataset)."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({"x": np.arange(1, 6), "y": rng.integers(1, 10, 5) * 2})


def to_png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def chart_sample(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["x"], df["y"], marker="o", label="y")
    ax.set(title="Sample", xlabel="X", ylabel="Y")
    ax.legend()
    return fig


# id, 제목, 빌더, 설명(수업 개념 + 해석)
CHARTS = [
    {"id": "sample", "title": "샘플 그래프", "build": chart_sample,
     "desc": "뼈대 확인용 그래프입니다. 데이터가 정해지면 교체합니다."},
]


@app.route("/")
def index():
    df = load_data()
    cards = []
    with _plot_lock:
        for c in CHARTS:
            fig = c["build"](df)
            cards.append({"id": c["id"], "title": c["title"], "desc": c["desc"],
                          "html": mpld3.fig_to_html(fig)})
            plt.close(fig)
    return render_template("index.html", cards=cards)


if __name__ == "__main__":
    app.run(debug=True)
