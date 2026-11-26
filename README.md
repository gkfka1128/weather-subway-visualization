# 날씨는 서울 시민의 이동을 어떻게 바꾸는가 — 지하철·버스·따릉이 (Flask + matplotlib/seaborn + mpld3 + Plotly)

서울교통공사 1~8호선 2023~2025년(3년) 승하차 데이터와 Open-Meteo 날씨를 중심으로,
같은 기간 서울 버스·따릉이(공공자전거)·미세먼지·역 좌표를 함께 엮어
(1) 날씨와 이용량의 관계를 그래프 25개로 보여 주는 **발표용 대시보드**(포트 5000)와
(2) 오늘~6일 뒤 예보로 "지하철, 이 시간에 붐빌까?"를 알려 주는 **이용자용 서비스**(포트 5001)를 제공한다.

## 실행

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py              # 대시보드 http://127.0.0.1:5000
.\.venv\Scripts\python.exe service/serve.py    # 서비스   http://127.0.0.1:5001
```

예보를 인터넷에서 받아 오므로 오프라인이면 예측 부분만 안내 문구로 대체된다.

### 날씨 예보 API

- **기상청 단기예보**(공공데이터포털 「기상청_단기예보 조회서비스」): `.env.example`을 `.env`로 복사해 `KMA_SERVICE_KEY=`에 인증키를 넣으면 쓴다.
  오늘~3일 뒤를 기상청 값으로 채우고, 화면의 날씨 칸에 "기상청 단기예보 · ○시 발표"로 표시된다.
- **Open-Meteo**(키 불필요): 기상청이 주지 않는 4~6일 뒤, 그리고 키가 없거나 기상청 호출이 실패했을 때 쓴다.
- GitHub Pages 판은 인증키를 숨길 서버가 없어 Open-Meteo만 쓴다.

## Docker

```bash
docker build -t subway-weather .
docker run --rm -p 8000:8000 subway-weather   # 대시보드 http://localhost:8000
docker run --rm -p 8000:8000 --env-file .env subway-weather   # 기상청 인증키(.env)를 넘겨 실행
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
- `charts.py` — 그래프 25개(matplotlib/seaborn 19개 + 마우스를 올리면 값이 뜨는 Plotly 6개) + 설명 문장(숫자는 데이터에서 계산), 화면 상단 핵심 숫자, 결론 탭의 통념 검증 표(`verdicts()`)
- `mobility.py` — 지하철·버스·따릉이 비교, 역 성격(유형)별 효과, 미세먼지 (같은 요일 보통날 대비 %)
- `app.py` — 날씨 예보(기상청 단기예보 + Open-Meteo) → 오늘 예측, 대시보드 라우트, `/api/predict?line=2호선`(JSON)
- `service/` — 이용자용 사이트 (`serve.py`, `templates/home.html`, `static/service.css`)
- `prep/` — 데이터 준비 스크립트 (`aggregate_subway.py`, `fetch_weather.py`, `merge.py`, `fetch_mobility.py`, `fetch_air.py`, `stations.py`, `link_bike.py`)
- `data/` — 집계·병합 CSV (원본 `data/raw/`는 git 제외), `TEAM_PLAN.md`
- `report/` — 보고서용 화면 캡처. `node report/capture.mjs`가 게시된 사이트를 헤드리스 Chrome으로 찍어 `report/captures/`에 넣는다

### 데이터 다시 만들기

원본(약 210MB, git 제외)을 `data/raw/`에 아래 이름으로 받은 뒤 `prep/` 스크립트를 순서대로 실행한다.

| 파일 | 출처 |
|---|---|
| `subway_1.csv`, `subway_2.csv` (2025년 상·하반기, 승객유형 포함) | 서울 열린데이터광장 OA-22470 |
| `subway_2023.csv`, `subway_2024.csv` (승객유형 없음) | 공공데이터포털 15048032 「주기성 과거 데이터」의 20231231·20241231 파일 |

```bash
python prep/aggregate_subway.py   # → subway_line_hourly / subway_hourly(2025 승객유형) / subway_station
python prep/fetch_weather.py      # → weather_hourly (Open-Meteo 2023~2025), holidays
python prep/merge.py              # → merged.csv
```

버스·따릉이·미세먼지·역 좌표는 스크립트가 서울 열린데이터광장에서 직접 받는다(원본 약 12GB, `data/raw/`, git·Docker 제외).

| 결과 파일 | 출처 | 스크립트 |
|---|---|---|
| `bus_daily.csv` 서울 버스 하루 승하차 합 | OA-12912 버스노선별 정류장별 승하차 인원(일별) | `prep/fetch_mobility.py bus` |
| `bike_daily.csv`, `bike_near_station.csv` 따릉이 하루 대여(정기권/일일권), 역 300m 안 대여 | OA-15246 공공자전거 이용정보(일별), OA-13252 대여소 정보 | `prep/fetch_mobility.py bike` → `prep/link_bike.py` |
| `air_daily.csv` PM10·PM2.5 일평균 | OA-2218 서울시 일별 평균 대기오염도(측정소 실측) | `prep/fetch_air.py` |
| `station_info.csv` 역 좌표·성격(업무지구/주거지/여가·상업/혼합)·주변 대여소 | OA-21232 역사마스터 정보 + 지하철 원본 | `prep/stations.py` (`link_bike.py` 전에 실행) |

## 방법 요약

1. 같은 요일 유형·시간대의 **앞뒤 4주 무강수일 평균**을 그날의 기준선으로 두고 실제/기준선 비율을 날씨 구간별로 비교(계절·연휴 효과 제거).
2. 예측 = 예보일과 같은 계절의 무강수일 평균 × (강수·적설 계수 × 기온 계수).
3. 백테스트로 검증: 날씨 보정은 효과가 평소 변동보다 작아 정확도를 크게 올리지 못한다(대시보드 7번 섹션).
4. 버스·따릉이·역 성격 비교는 하루 단위로, 같은 요일 앞뒤 4주 '맑은 보통날'(공휴일·징검다리·비·눈·미세먼지 나쁨 제외) 중앙값 대비 %로 같은 방법을 쓴다.

## 이야기 구조

화면은 섹션 탭(오늘 예측 · 01~07 · 결론)으로 나뉘어 한 번에 한 섹션만 보인다. 선택한 탭은 주소 끝(`#s3` 등)에 남아 호선·요일을 바꿔도 유지되고, 섹션 끝의 '다음' 버튼으로 이야기 순서대로 넘어간다.


질문 ① 날씨가 나쁘면 지하철을 덜 타는가 → 조금(많은 비 평일 −5.5%) ·
질문 ② 다른 수단은? → 따릉이는 −57%, 버스는 지하철과 비슷. 옮겨 타기보다 이동 자체가 줄어든다 ·
질문 ③ 누가 줄이나? → 평일 출퇴근(필수 이동)은 그대로, 주말(선택 이동)이 2~3배 더 줄어든다 ·
그래서 → 날씨는 '상대적 수준'으로 안내하는 예측 서비스.

## 한계

- 이용량은 승하차 인원이며 열차 안 혼잡도가 아니다.
- 버스·따릉이는 하루 단위라 시간대 비교가 안 된다. 택시·자가용 자료가 없어 '줄어든 이동이 어디로 갔는지'는 일부만 확인된다. 역 성격은 승하차 패턴으로 붙인 데이터 기준 분류다.
- 데이터는 2023~2025년 3년치(승객 유형은 2025년만). 2024년 4월은 원본 파일의 시간대 칸이 밀려 있어 통째로 제외(`prep/aggregate_subway.py`의 `BAD_DATES`). 강한 비 141시간대·눈 305시간대로 여전히 많지 않다. 예측의 평소 수준은 최근 1년(2025) 기준. 2026 공휴일은 `analysis.py`에 수동 입력.
- 예보일이 멀수록 오차가 커진다.
