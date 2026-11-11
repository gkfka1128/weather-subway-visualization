# 날씨가 서울 지하철 이용량에 미치는 영향 (Flask + matplotlib/seaborn + mpld3)

서울교통공사 1~8호선 2025년 승하차 데이터와 Open-Meteo 날씨를 합쳐
(1) 날씨와 이용량의 관계를 그래프 11개로 보여 주는 **발표용 대시보드**(포트 5000)와
(2) 오늘~6일 뒤 예보로 "지하철, 이 시간에 붐빌까?"를 알려 주는 **이용자용 서비스**(포트 5001)를 제공한다.

## 실행

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py              # 대시보드 http://127.0.0.1:5000
.\.venv\Scripts\python.exe service/serve.py    # 서비스   http://127.0.0.1:5001
```

예보(Open-Meteo, 키 불필요)를 인터넷에서 받아 오므로 오프라인이면 예측 부분만 안내 문구로 대체된다.

## Docker

```bash
docker build -t subway-weather .
docker run --rm -p 8000:8000 subway-weather   # 대시보드 http://localhost:8000
```

서비스 사이트(사용자용):

```bash
docker run --rm -p 8001:8001 subway-weather gunicorn -b 0.0.0.0:8001 --chdir service serve:app   # http://localhost:8001
```

2026-11-11 Docker Desktop 29.9에서 빌드·실행 확인(대시보드·`/api/predict`·서비스 모두 200, 그래프 한글은 나눔고딕).
첫 요청은 그래프를 만드느라 수십 초 걸릴 수 있다.

## 구조

- `analysis.py` — 데이터 로딩, 계절 보정 기준선, 날씨 보정 계수, 백테스트
- `charts.py` — 그래프 11개 + 설명 문장(숫자는 데이터에서 계산), 화면 상단 핵심 숫자
- `app.py` — 날씨 예보 → 오늘 예측, 대시보드 라우트, `/api/predict?line=2호선`(JSON)
- `service/` — 이용자용 사이트 (`serve.py`, `templates/home.html`, `static/service.css`)
- `prep/` — 데이터 준비 스크립트 (`aggregate_subway.py`, `fetch_weather.py`, `merge.py`)
- `data/` — 집계·병합 CSV (원본 `data/raw/`는 git 제외), `TEAM_PLAN.md`, `report/`

## 방법 요약

1. 같은 요일 유형·시간대의 **앞뒤 4주 무강수일 평균**을 그날의 기준선으로 두고 실제/기준선 비율을 날씨 구간별로 비교(계절·연휴 효과 제거).
2. 예측 = 예보일과 같은 계절의 무강수일 평균 × (강수·적설 계수 × 기온 계수).
3. 백테스트로 검증: 날씨 보정은 효과가 평소 변동보다 작아 정확도를 크게 올리지 못한다(대시보드 4번 섹션).

## 한계

- 이용량은 승하차 인원이며 열차 안 혼잡도가 아니다.
- 데이터는 2025년 1년치. 강한 비·눈 표본이 적다. 2026 공휴일은 `analysis.py`에 수동 입력.
- 예보일이 멀수록 오차가 커진다.
