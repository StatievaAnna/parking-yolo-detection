FROM bluenviron/mediamtx:latest AS mediamtx_bin

FROM python:3.9.6

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        netcat-openbsd \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY cvpipeline/requirements.txt ./cvpipeline/requirements.txt
RUN pip install --no-cache-dir -r cvpipeline/requirements.txt

COPY . .
COPY --from=mediamtx_bin /mediamtx /usr/local/bin/mediamtx

RUN chmod +x /usr/local/bin/mediamtx /app/start_pipeline.sh

CMD ["bash", "/app/start_pipeline.sh"]
