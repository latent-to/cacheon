ARG SGLANG_IMAGE=lmsysorg/sglang:dev-muse-spark-open@sha256:8cfc5a72cb296771f5f8a4a4688cb6d05685610a830cbb616fa4ab0eb7bb2b0a
FROM ${SGLANG_IMAGE}
LABEL org.opencontainers.image.source="https://github.com/latent-to/cacheon" org.opencontainers.image.title="Cacheon Muse-Spark-Open B300 TP4 qualification worker"
# The day-0 image's python3 is the /opt/sglang venv (SGLang installed editable from
# /sgl-workspace/sglang-muse-spark/python). The relocation is the same as the DeepSeek-V4.1
# worker's and stays above COPY so a source-only rebuild reuses the layer.
RUN rmdir /usr/local/lib/python3.12/dist-packages \
 && mv /opt/sglang/lib/python3.12/site-packages /usr/local/lib/python3.12/dist-packages \
 && ln -s /usr/local/lib/python3.12/dist-packages /opt/sglang/lib/python3.12/site-packages
COPY src /opt/cacheon-source
RUN /opt/sglang/bin/python -m pip install --no-deps --no-build-isolation --no-index /opt/cacheon-source \
 && printf 'import cacheon.bootstrap\n' > /usr/local/lib/python3.12/dist-packages/cacheon.pth
# Private archive extraction can leave copied fork directories root-only.
RUN chmod -R a+rX /sgl-workspace/sglang-muse-spark/python/sglang
USER 65534:65534
RUN HOME=/tmp /usr/bin/python3.12 -I -c 'import cacheon, cacheon.bootstrap, sglang, torch; from sglang.srt.environ import envs; print(cacheon.__file__, sglang.__version__, torch.__version__)'
USER 0:0
