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

## 배포 1 — GitHub Pages (정적, 무료)

Pages는 서버를 돌릴 수 없어서 `build_static.py`가 정적 사이트(`site/`)를 만든다.
- 대시보드: 호선 9 × 요일 유형 4 = 36개 화면을 미리 HTML로 렌더링(그래프는 Flask 버전과 동일). 콤보박스는 화면 사이를 이동한다.
- 오늘 예측·서비스(`today.html`): 평소 이용량·보정 계수를 `model.json`으로 내보내고, 브라우저가 Open-Meteo 예보를 직접 받아 계산한다(`static_site/predict.js`).
  평소 이용량은 7일 간격 기준일 중 가장 가까운 날을 써서 Flask 버전과 0.1% 안팎 차이가 난다.

main에 push하면 `.github/workflows/pages.yml`이 빌드·배포한다(저장소 Settings → Pages → Source: GitHub Actions).
로컬 확인: `python build_static.py` 후 `python -m http.server 8090 -d site`.

## 배포 2 — Render (Flask 그대로, 무료)

`render.yaml`에 두 사이트가 정의돼 있다. GitHub에 올린 뒤 Render → **New → Blueprint** → 저장소를 고르면 끝이다.
컨테이너는 `PORT` 환경변수를 따른다. 워커 1개 + 스레드 4개로 띄워 메모리를 무료 한도(512MB) 안으로 유지한다(워커 2개일 때 대시보드 약 305MB).
무료 플랜은 15분간 요청이 없으면 잠들고, 깨어날 때 1분 안팎 걸린다. 발표 직전에 한 번 열어 두면 된다.

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
