ARG SGLANG_IMAGE
FROM ${SGLANG_IMAGE}
LABEL org.opencontainers.image.source="https://github.com/latent-to/cacheon" org.opencontainers.image.title="Cacheon DeepSeek-V4.1-Flash PD B300 qualification worker"
# The dependency relocation is a 20 GB layer that never changes between source
# revisions; it stays above COPY so a source-only rebuild reuses it (measured
# 2026-10-09: 468 s install plus 808 s export when it sat below the copy).
RUN rmdir /usr/local/lib/python3.12/dist-packages \
 && mv /opt/sglang/lib/python3.12/site-packages /usr/local/lib/python3.12/dist-packages \
 && ln -s /usr/local/lib/python3.12/dist-packages /opt/sglang/lib/python3.12/site-packages
COPY src /opt/cacheon-source
RUN /opt/sglang/bin/python -m pip install --no-deps --no-build-isolation --no-index /opt/cacheon-source \
 && printf 'import cacheon.bootstrap\n' > /usr/local/lib/python3.12/dist-packages/cacheon.pth
# Private archive extraction can leave copied fork directories root-only.
RUN chmod -R a+rX /sgl-workspace/sglang/python/sglang
USER 65534:65534
RUN HOME=/tmp /usr/bin/python3.12 -I -c 'import cacheon, cacheon.bootstrap, sglang, torch; from sglang.srt.environ import envs; print(cacheon.__file__, sglang.__version__, torch.__version__)'
USER 0:0
