FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install --yes --no-install-recommends bash coreutils git ripgrep tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --create-home --home-dir /home/tars --shell /usr/sbin/nologin tars

COPY src/tars_agent/sandbox/worker.py /opt/tars/worker.py

ENV HOME=/home/tars \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1

USER 10001:10001
WORKDIR /workspace
ENTRYPOINT ["/usr/bin/tini", "--", "sleep", "infinity"]
