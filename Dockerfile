FROM python:3.11-slim

# set working directory
WORKDIR /app

# keep Python output unbuffered
ENV PYTHONUNBUFFERED=1

# Install minimal OS deps for psycopg2
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libpq-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . /app

# Install Python deps
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "-u", "-m", "src.webapp.app"]