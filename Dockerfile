# apps/learning-style-ai/Dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 8501

# Render and Cloud Run inject $PORT; fall back to 8501 everywhere else.
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"]
