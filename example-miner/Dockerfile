FROM vllm/vllm-openai:latest

EXPOSE 8000

COPY start.sh /start.sh
RUN chmod +x /start.sh
ENTRYPOINT ["/start.sh"]
