FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer-cached until requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# LLM_PROVIDER defaults to local_hf; override at runtime:
#   docker run -e LLM_PROVIDER=openai -e OPENAI_API_KEY=sk-xxx ...
ENV LLM_PROVIDER=local_hf

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
