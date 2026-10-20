# 데이터 시각화 대시보드 (Flask + matplotlib/seaborn + mpld3)

과제 요구: 그래프 약 10개 + 그래프별 설명, 웹페이지 게시, 컨트롤(콤보박스/버튼), Docker, GitHub.
데이터셋은 미정 — `app.py`의 `load_data()`와 `CHARTS`를 채우면 된다.

## 실행

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py      # http://127.0.0.1:5000
```

## Docker

```bash
docker build -t dataviz-dashboard .
docker run --rm -p 8000:8000 dataviz-dashboard   # http://localhost:8000
```

## 구조

- `app.py` — Flask 앱, 데이터 로딩, 차트 빌더 목록
- `templates/index.html`, `static/style.css` — 페이지
- `data/` — 데이터 파일, `report/` — 보고서 작업물
