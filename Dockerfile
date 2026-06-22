FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs

RUN uv pip install --system .

COPY tests ./tests

CMD ["python", "-m", "atari_rl.train", "--config", "configs/ppo.yaml"]
