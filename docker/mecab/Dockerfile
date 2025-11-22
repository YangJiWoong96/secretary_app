FROM python:3.11-slim

# 시스템 패키지 업데이트 & mecab 설치
RUN apt-get update && apt-get install -y \
    mecab \
    libmecab-dev \
    mecab-ipadic-utf8 \
    git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# mecab-ko / mecab-ko-dic / mecab-python3 설치
RUN pip install --no-cache-dir mecab-python3 mecab-ko mecab-ko-dic

# 테스트 코드 복사
COPY test_mecab.py /app/test_mecab.py
WORKDIR /app

CMD ["python", "test_mecab.py"]