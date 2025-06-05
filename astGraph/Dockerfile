FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/output

RUN chmod -R 755 /app

EXPOSE 8000

CMD ["gunicorn", "cfg:app", "--bind", "0.0.0.0:8000"]