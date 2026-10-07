FROM lmsysorg/sglang@sha256:b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469
LABEL org.opencontainers.image.source="https://github.com/latent-to/cacheon" org.opencontainers.image.title="Cacheon DeepSeek-V4.1-Flash B300 qualification worker"
COPY src /opt/cacheon-source
RUN /opt/sglang/bin/python -m pip install --no-deps --no-build-isolation --no-index /opt/cacheon-source \
 && rmdir /usr/local/lib/python3.12/dist-packages \
 && mv /opt/sglang/lib/python3.12/site-packages /usr/local/lib/python3.12/dist-packages \
 && ln -s /usr/local/lib/python3.12/dist-packages /opt/sglang/lib/python3.12/site-packages \
 && printf 'import cacheon.bootstrap\n' > /usr/local/lib/python3.12/dist-packages/cacheon.pth
RUN /usr/bin/python3.12 -I -c 'import cacheon, cacheon.bootstrap, sglang, torch; print(cacheon.__file__, sglang.__version__, torch.__version__)'
