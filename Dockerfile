FROM python:3.12-slim
WORKDIR /app
COPY . .

# Install runtime dependencies (add xvfb for headless 3D viz)
RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -e ".[api]"
RUN useradd --create-home twin && chown -R twin:twin /app
USER twin
ENV DISPLAY=:99
EXPOSE 8000
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
