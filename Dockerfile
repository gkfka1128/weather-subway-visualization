FROM python:3.12-slim

# matplotlib 한글 표시용 폰트
RUN apt-get update && apt-get install -y --no-install-recommends fonts-nanum \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 8000
# 호스팅(Render 등)은 PORT 환경변수로 포트를 정해 준다. 없으면 8000.
# 첫 요청에서 그래프를 만드느라 오래 걸릴 수 있어 timeout을 넉넉히 둔다.
CMD ["sh", "-c", "exec gunicorn -b 0.0.0.0:${PORT:-8000} -w 1 --threads 4 --timeout 180 app:app"]
