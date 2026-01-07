FROM python:3.13-slim

# 2. System Deps (git zaroori hai kuch libraries ke liye)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# 3. Install Python Depandencies
RUN pip install --no-cache-dir -r requirements.txt

# (Ye line ensure karegi ki ingest kiya hua data image mein chala jaye)
COPY . .

# 5. Env Vars
ENV HF_HUB_DISABLE_SYMLINKS=1
ENV PORT=8000
# IMPORTANT: Container ke andar Qdrant ka URL set karein
ENV QDRANT_URL=http://qdrant_vector_db:6333

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]