ARG VLLM_BASE_IMAGE=m.daocloud.io/docker.io/vllm/vllm-openai:v0.16.0
FROM ${VLLM_BASE_IMAGE}

# Prefer the host driver libraries mounted by NVIDIA Container Toolkit.  The
# cuda-compat copy bundled in vLLM 0.16 can return CUDA error 803 on R580/R590.
ENV LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH} \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    PIP_TRUSTED_HOST=mirrors.aliyun.com \
    PIP_EXTRA_INDEX_URL=

RUN set -eux; \
    for source_file in /etc/apt/sources.list /etc/apt/sources.list.d/ubuntu.sources; do \
        if [ -f "${source_file}" ]; then \
            sed -Ei \
                -e 's#https?://([a-zA-Z0-9.-]+\.)?archive\.ubuntu\.com/ubuntu/?#https://mirrors.aliyun.com/ubuntu/#g' \
                -e 's#https?://security\.ubuntu\.com/ubuntu/?#https://mirrors.aliyun.com/ubuntu/#g' \
                "${source_file}"; \
        fi; \
    done; \
    for source_file in /etc/apt/sources.list.d/*; do \
        if [ -f "${source_file}" ] && grep -q 'developer.download.nvidia.com' "${source_file}"; then \
            rm -f "${source_file}"; \
        fi; \
    done; \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ffmpeg \
        libsndfile1 \
        libsm6 \
        libxext6 \
        && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY requirements.txt overrides.txt ./
RUN pip install --no-cache-dir --break-system-packages \
    --index-url https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com \
    uv && \
    uv pip install --system \
    --default-index https://mirrors.aliyun.com/pypi/simple/ \
    --allow-insecure-host mirrors.aliyun.com \
    --override overrides.txt \
    -r requirements.txt

COPY indextts /app/indextts
COPY tools /app/tools
COPY patch_vllm.py /app/patch_vllm.py
COPY api_server.py /app/api_server.py
COPY api_server_v2.py /app/api_server_v2.py
COPY convert_hf_format.py /app/convert_hf_format.py
COPY convert_hf_format.sh /app/convert_hf_format.sh
COPY entrypoint.sh /app/entrypoint.sh

ENTRYPOINT ["bash", "/app/entrypoint.sh"]
