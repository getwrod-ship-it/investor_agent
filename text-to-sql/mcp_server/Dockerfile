FROM python:3.12-slim

WORKDIR /app

COPY mcp_server/server.py .

RUN pip install --no-cache-dir "mcp>=1.0" "requests>=2.31"

CMD ["python", "server.py"]
