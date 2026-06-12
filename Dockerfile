# syntax=docker/dockerfile:1.7
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FTV_APP_HOST=0.0.0.0 \
    FTV_APP_PORT=8765

WORKDIR /app

ARG FTV_INSTALL_GPU=0

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libvulkan1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md constraints.txt ./
COPY src ./src

RUN python -m pip install --no-cache-dir --constraint constraints.txt setuptools wheel \
    && if [ "$FTV_INSTALL_GPU" = "1" ]; then \
        python -m pip install --no-cache-dir --constraint constraints.txt --no-build-isolation -e ".[gpu]"; \
    else \
        python -m pip install --no-cache-dir --constraint constraints.txt --no-build-isolation -e .; \
    fi

EXPOSE 8765

CMD ["ftv-app"]
