FROM lmsysorg/sglang@sha256:06e4f2ed21afde4ff513cda65070124e727ba23ccaeff7712b8c40e1097d611f
LABEL org.opencontainers.image.source="https://github.com/latent-to/cacheon" org.opencontainers.image.title="Cacheon GLM-5.3 B300 native-MTP qualification worker"
COPY src /opt/cacheon-source
RUN cd /sgl-workspace/sglang \
 && git apply --unidiff-zero /opt/cacheon-source/examples/arena_inputs/glm53/memcpy-triton-bounds.patch
RUN /opt/sglang/bin/python -m pip install --no-deps --no-build-isolation --no-index /opt/cacheon-source \
 && rmdir /usr/local/lib/python3.12/dist-packages \
 && mv /opt/sglang/lib/python3.12/site-packages /usr/local/lib/python3.12/dist-packages \
 && ln -s /usr/local/lib/python3.12/dist-packages /opt/sglang/lib/python3.12/site-packages \
 && printf 'import cacheon.bootstrap\n' > /usr/local/lib/python3.12/dist-packages/cacheon.pth
RUN /usr/bin/python3.12 -I -c 'import cacheon, cacheon.bootstrap, sglang, torch; print(cacheon.__file__, sglang.__version__, torch.__version__)'
