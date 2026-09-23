FROM lmsysorg/sglang@sha256:06e4f2ed21afde4ff513cda65070124e727ba23ccaeff7712b8c40e1097d611f
LABEL org.opencontainers.image.source="https://github.com/latent-to/cacheon"       org.opencontainers.image.title="Cacheon Qwen3.6-35B H100 qualification worker"
# The reference scoring request needs 1.89 GiB per 2,048-position logits chunk and the lane has 1.23 GiB free
# (2026-09-20); the launcher passes no engine environment, so the chunk size rides in the image for every arm.
ENV SGLANG_LOGPROB_CHUNK_SIZE=256
COPY src /opt/cacheon-source
RUN /opt/sglang/bin/python -m pip install --no-deps --no-build-isolation --no-index /opt/cacheon-source  && rmdir /usr/local/lib/python3.12/dist-packages  && mv /opt/sglang/lib/python3.12/site-packages /usr/local/lib/python3.12/dist-packages  && ln -s /usr/local/lib/python3.12/dist-packages /opt/sglang/lib/python3.12/site-packages  && printf 'import cacheon.bootstrap\n' > /usr/local/lib/python3.12/dist-packages/cacheon.pth
COPY moecfg/ /sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_1/
RUN /usr/bin/python3.12 -I -c 'import cacheon, cacheon.bootstrap, sglang, torch; print(cacheon.__file__, sglang.__version__, torch.__version__)'
